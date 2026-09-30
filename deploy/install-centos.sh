#!/usr/bin/env bash
#
# Kokoro TTS installer for CentOS Stream 9 (also works on Rocky/Alma 9, RHEL 9).
#
#   sudo bash deploy/install-centos.sh
#
# Idempotent: safe to re-run. Installs to /opt/kokoro, runs as the unprivileged
# `kokoro` user under systemd.

set -euo pipefail

APP_USER="kokoro"
APP_DIR="/opt/kokoro"
STATE_DIR="/var/lib/kokoro"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${KOKORO_PORT:-8000}"
PY_VER="${KOKORO_PYTHON:-3.12}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"

# ---------------------------------------------------------------- preflight
log "Checking platform compatibility"
GLIBC="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
echo "    glibc ${GLIBC:-unknown}"
if command -v rpm &>/dev/null; then
  . /etc/os-release
  echo "    ${PRETTY_NAME:-unknown}"
fi
# torch publishes manylinux_2_28 wheels only, which need glibc >= 2.28.
if [[ -n "$GLIBC" ]] && awk "BEGIN{exit !($GLIBC < 2.28)}"; then
  die "glibc ${GLIBC} is too old. torch requires >= 2.28 (CentOS Stream/RHEL 8+).
      CentOS 7 cannot run this. Migrate to Rocky/Alma 9 or use a container."
fi

[[ -f "${SRC_DIR}/app/engine.py" ]] || die "run this from inside the project checkout"

log "Installing system packages"
# openssl generates the API key and shadow-utils provides useradd; neither is
# guaranteed on a minimal Stream 9 image, and `set -e` would abort on a missing
# one. Deliberately no rsync - the code copy below uses coreutils only.
dnf install -y curl ca-certificates tar openssl shadow-utils
# ffmpeg is optional: only needed for format=mp3. Stream 9 ships it in CRB.
if ! dnf install -y ffmpeg-free >/dev/null 2>&1; then
  dnf install -y https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E '%{rhel}').noarch.rpm >/dev/null 2>&1 || true
  dnf install -y ffmpeg-free >/dev/null 2>&1 || warn "ffmpeg unavailable; WAV still works"
fi
command -v ffmpeg >/dev/null && echo "    ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"

# -------------------------------------------------------------------- user
log "Creating service account"
if ! id -u "${APP_USER}" &>/dev/null; then
  useradd --system --create-home --home-dir "/home/${APP_USER}" \
          --shell /sbin/nologin "${APP_USER}"
fi
install -d -m 0755 -o root -g root "${APP_DIR}"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "${STATE_DIR}"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "${STATE_DIR}/huggingface" "${STATE_DIR}/output"
install -d -m 0750 -o "${APP_USER}" -g "${APP_USER}" "/etc/kokoro"

# --------------------------------------------------------------------- code
log "Installing application code into ${APP_DIR}"
# Only a fixed, known set of paths is managed here, so removing each one and
# re-copying is equivalent to `rsync --delete` while depending on nothing
# beyond coreutils. The venv and all runtime state are safe because they live
# outside this list: the venv is created after this step, and state is under
# ${STATE_DIR}.
for d in app deploy tests; do
  rm -rf "${APP_DIR:?}/${d}"
  cp -a "${SRC_DIR}/${d}" "${APP_DIR}/${d}"
done
for f in README.md DEPLOY.md pyproject.toml .gitignore; do
  if [[ -f "${SRC_DIR}/${f}" ]]; then
    cp -a "${SRC_DIR}/${f}" "${APP_DIR}/${f}"
  fi
done
find "${APP_DIR}" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
chown -R root:root "${APP_DIR}"
chmod -R go-w "${APP_DIR}"

# --------------------------------------------------------------------- uv
if ! command -v uv &>/dev/null; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
uv --version | sed 's/^/    /'

# ------------------------------------------------------------------ python
log "Building virtualenv (Python ${PY_VER})"
cd "${APP_DIR}"
uv python install "${PY_VER}"
uv venv --python "${PY_VER}" "${APP_DIR}/.venv"
VPY="${APP_DIR}/.venv/bin/python"

log "Installing torch (CPU build)"
# CPU-only index keeps this to ~180 MB instead of pulling the CUDA runtime.
uv pip install --python "${VPY}" torch \
  --index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match

log "Installing application dependencies"
uv pip install --python "${VPY}" \
  "kokoro>=0.9.4" "misaki[en]>=0.9.4" soundfile espeakng-loader \
  fastapi "uvicorn[standard]" python-multipart numpy

# The unit file mounts /opt read-only, so precompile .pyc now while we still
# can write. Without this every start re-parses the source tree.
"${VPY}" -m compileall -q "${APP_DIR}/app" >/dev/null

# Fail loudly here rather than on the first systemd start.
"${VPY}" - <<'PYCHECK'
import sys
import app.engine  # noqa: F401
print("    import check: ok", file=sys.stderr)
PYCHECK

chown -R root:root "${APP_DIR}/.venv"

# ------------------------------------------------------------------- model
log "Pre-downloading Kokoro-82M weights (~330 MB, one time)"
# Run once as the service user so the cache is created with correct ownership
# and systemd's hardening never needs a network fetch at boot.
sudo -u "${APP_USER}" env HF_HOME="${STATE_DIR}/huggingface" PYTHONIOENCODING=utf-8 \
  "${VPY}" -m app.cli --warmup

# ----------------------------------------------------------------- systemd
log "Installing systemd unit"
if [[ -z "${KOKORO_API_KEY:-}" ]]; then
  API_KEY="$(openssl rand -hex 32)"
else
  API_KEY="${KOKORO_API_KEY}"
fi
sed -e "s|^Environment=KOKORO_API_KEY=.*|Environment=KOKORO_API_KEY=${API_KEY}|" \
    -e "s|--port 8000|--port ${PORT}|" \
    "${SRC_DIR}/deploy/kokoro-tts.service" > /etc/systemd/system/kokoro-tts.service
chmod 0644 /etc/systemd/system/kokoro-tts.service

# Drop the loopback default so a reverse proxy or direct LAN access can reach it.
if [[ "${KOKORO_BIND:-127.0.0.1}" != "127.0.0.1" ]]; then
  sed -i "s|--host 127.0.0.1|--host ${KOKORO_BIND}|" /etc/systemd/system/kokoro-tts.service
fi

systemctl daemon-reload
systemctl enable kokoro-tts.service >/dev/null
systemctl restart kokoro-tts.service

log "Waiting for the service to answer"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
  || { journalctl -u kokoro-tts -n 40 --no-pager; die "service did not become healthy"; }

# ---------------------------------------------------------------- firewall
if command -v firewall-cmd &>/dev/null; then
  log "Opening port ${PORT} in firewalld"
  firewall-cmd --permanent --add-port="${PORT}/tcp" >/dev/null
  firewall-cmd --reload >/dev/null
fi

# ------------------------------------------------------------------ report
echo
log "Done. Service is running."
echo
echo "  status    systemctl status kokoro-tts"
echo "  logs      journalctl -u kokoro-tts -f"
echo "  health    curl http://127.0.0.1:${PORT}/health"
echo "  docs      http://127.0.0.1:${PORT}/docs"
echo
echo "  API key (send as X-API-Key or Authorization: Bearer):"
echo "    ${API_KEY}"
echo
echo "  Save it now:  sudo sh -c 'echo KOKORO_API_KEY=${API_KEY} > /etc/kokoro/tts.env'"
echo "  The web UI at / has no way to send a header, so serve it behind a"
echo "  reverse proxy that terminates TLS, or leave KOKORO_API_KEY unset for"
echo "  a trusted LAN. See DEPLOY.md."
echo
echo "  audio out  ${STATE_DIR}/output"
echo "  model cache ${STATE_DIR}/huggingface"
echo
