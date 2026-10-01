# Deploying to a Linux server (CentOS Stream 9)

How to move this project onto a CentOS Stream 9 box and reach it over the network.

For the local Windows workflow see [README.md](README.md).

---

## 1. Check the OS first

```bash
cat /etc/centos-release
ldd --version | head -1
```

`ldd` must report **glibc 2.28 or newer**:

| Distro | glibc | Result |
|---|---|---|
| CentOS Stream 9, Rocky 9, Alma 9 | 2.34 | works |
| CentOS Stream 8, RHEL 8 | 2.28 | works, at the minimum |
| **CentOS 7** | 2.17 | **will not work** |

CentOS 7 is the reason this table exists. PyTorch publishes `manylinux_2_28`
wheels only - there is no glibc 2.17 build on the index at any version - so a
`pip install torch` on CentOS 7 either fails outright or leaves you without a
usable wheel. CentOS 7 also went end-of-life in June 2024. If that is the box
you have, migrate to Rocky/Alma 9 or run this in a container on a newer host;
there is no in-place path.

`deploy/install-centos.sh` re-checks this and refuses to continue on an
unsupported box rather than failing later with a confusing wheel error.

---

## 2. Copy the project up

From your Windows machine, excluding the Windows virtual environment:

```powershell
# from the Kokoro folder on your PC
scp -r `
  app deploy tests `
  DEPLOY.md README.md pyproject.toml .gitignore config.json legal_preprocessor.py `
  user@SERVER_IP:kokoro-staging/
```

Then on the server:

```bash
git clone https://github.com/ch405maki/kokoro_web.git
cd kokoro_web
sudo bash deploy/install-centos.sh
```

That is the whole procedure, and it needs no `rsync`. The installer copies the
managed directories into `/opt/kokoro` using only coreutils (`rm` + `cp`), and it
resolves paths relative to its own location, so you can run it straight from the
clone in your home directory.

If you would rather stage into `/opt/kokoro` first and inspect it, `cp` works
too, and the installer will refresh it idempotently:

```bash
sudo mkdir -p /opt/kokoro
sudo cp -a app deploy tests README.md DEPLOY.md pyproject.toml config.json legal_preprocessor.py /opt/kokoro/
cd /opt/kokoro && sudo bash deploy/install-centos.sh
```

The only system packages the installer asks for are `curl`, `ca-certificates`,
`tar`, `openssl` and `shadow-utils`, plus `ffmpeg` if you want MP3 output.

### `config.json` is live configuration, not just a file

`config.json` holds the abbreviation, OCR and currency mappings used by
`POST /preprocess`. It sits at `/opt/kokoro/config.json` and is read once at
startup, so editing it needs a `sudo systemctl restart kokoro-tts`. Keep it
readable by the service user only if you care who can change how citations are
expanded. If the file is missing or malformed the server still starts and falls
back to its built-in defaults, so a bad edit degrades the formatter rather than
taking the API down.

The compiled matchers for those mappings are cached in memory, keyed on the
config object they were built from. A restart picks up an edit; running the old
process does not, which is the intended behaviour. A partial config is safe too -
an empty `abbreviations` object, a non-string value or an `editorial_markers`
list of junk all get ignored rather than raising, so you can trim the file down
while you experiment.

### The preprocessor is pure standard library

`legal_preprocessor.py` uses nothing outside the Python standard library: no C
extension, no compiler step, no `pip install` on the server, and no
platform-specific branch. The same file and the same behaviour run on Windows and
CentOS, so there is no second build to keep in sync. It is the one part of this
project that behaves identically everywhere, which matters because it runs on
every paste, synchronously, on the request path.

It has been tuned for throughput, since a pasted decision can be 130 kB. A
synthetic 130,132-character / 1,609-line decision cleans in about **38 ms**
(3.4 MB/s) against ~120 ms before the work, with byte-for-byte identical output.
Through `POST /preprocess` that is a 45 ms round trip including HTTP, and a
typical 13 kB paste comes back in 7 ms. Since the step is synchronous and
unauthenticated by default, the practical ceiling before anyone notices latency
is a few hundred kB per request; past that, put nginx in front of it and send the
document in pieces. Full method and measurements are in the README.

### Verifying the preprocessor on the server

The unit suite needs no server and no model weights, so it is the cheap check
after any `config.json` or preprocessor change:

```bash
cd /opt/kokoro
sudo -u kokoro .venv/bin/python -m unittest tests.test_preprocessor -v
```

46 tests, under a second. If you change the matcher itself, also confirm the
output did not change: run `tests/test_preprocessor.py` on both the old and new
commit and diff the results over a real decision before deploying.

### Do not copy `.venv/`

It contains Windows `.exe` launchers and Windows DLLs - `torch`, `scipy`,
`spaCy` and `soundfile` are all compiled binaries. Uploading it wastes ~2 GB and
the installer ignores it, so if you copy the folder by hand you would copy
gigabytes that then have to be deleted. The same applies to `output/`, which
accumulates generated audio.

Cloning is the cheapest transfer: `git clone` pulls ~110 KB, whereas `scp -r` of
the project directory would drag the 904 MB `.venv` and any generated audio
along with it.

---

## 3. Install

```bash
git clone https://github.com/ch405maki/kokoro_web.git
cd kokoro_web
sudo bash deploy/install-centos.sh
```

The installer is idempotent - re-run it after a code update. It will:

1. verify glibc and refuse on anything older than 2.28
2. install `curl`, `ca-certificates`, and `ffmpeg` (MP3 output only)
3. create the unprivileged `kokoro` system user
4. copy the managed directories into `/opt/kokoro` using `rm` + `cp`
5. install `uv`, then build a Python 3.12 venv
6. install **CPU-only** torch from `download.pytorch.org/whl/cpu`, then the app
7. byte-compile the tree and import-check it
8. pre-download Kokoro-82M into `/var/lib/kokoro/huggingface` as the service user
9. install and start the systemd unit
10. open the port in firewalld, then wait for `/health` to answer

It prints a generated API key when it finishes. **Save it.** The model download
is ~330 MB and takes a few minutes; every request afterwards is local.

Useful overrides:

```bash
sudo KOKORO_PORT=9000 bash deploy/install-centos.sh
sudo KOKORO_BIND=0.0.0.0 bash deploy/install-centos.sh   # see section 5 first
sudo KOKORO_API_KEY=$(openssl rand -hex 32) bash deploy/install-centos.sh
```

---

## 4. What it installs

| Path | Purpose |
|---|---|
| `/opt/kokoro` | the code, root-owned and read-only to the service |
| `/opt/kokoro/.venv` | Python 3.12, torch CPU, kokoro, misaki, FastAPI |
| `/var/lib/kokoro/huggingface` | model cache (~330 MB), the only thing worth backing up |
| `/var/lib/kokoro/output` | audio written by `save=true` |
| `/etc/kokoro/tts.env` | optional env overrides, `EnvironmentFile=-` so it may be absent |
| `/etc/systemd/system/kokoro-tts.service` | the unit |

Running as an unprivileged `kokoro` user matters: the API renders arbitrary
user-supplied text through a G2P pipeline and, with `save=true`, writes files.
You do not want that as root.

The unit is hardened - `ProtectSystem=strict`, `NoNewPrivileges`,
`PrivateTmp`, `MemoryDenyWriteExecute`, a 4 GB memory cap, and no Linux
capabilities. `ProtectSystem=strict` makes `/opt/kokoro` read-only, which is why
the installer pre-compiles bytecode: with the source tree read-only Python
cannot write `.pyc` files, and the unit sets `PYTHONDONTWRITEBYTECODE=1` so it
does not retry every boot.

`MemoryDenyWriteExecute=yes` is safe here because nothing in this stack
JIT-compiles. If you later add `torch.compile` or an ONNX runtime, set it to
`no` or the process will die with a segfault on the first inference.

---

## 5. Reaching it from the network

**The unit binds `127.0.0.1` on purpose.** Binding `0.0.0.0` on a public
interface hands an unauthenticated CPU burner to the internet, and `/speak` has
no rate limit. Pick one of these instead.

### Option A - nginx with TLS and Basic auth (recommended)

The built-in `KOKORO_API_KEY` check reads a request header, and a browser cannot
attach a custom header to a same-origin form post. So key auth protects the API
but breaks the web UI. Terminating at nginx and gating with HTTP Basic auth
protects both, because the browser handles that natively.

```bash
sudo cp /opt/kokoro/deploy/nginx-kokoro.conf /etc/nginx/conf.d/kokoro.conf
sudo sed -i 's/tts.example.com/YOUR.DOMAIN/' /etc/nginx/conf.d/kokoro.conf
sudo htpasswd -c /etc/nginx/kokoro.htpasswd kokoro
sudo nginx -t && sudo systemctl reload nginx
```

With this in place, leave `KOKORO_API_KEY` **unset** - nginx is the only gate.
The config also rate limits to 30 requests/minute per IP, keeps `/health` open
to localhost only for monitoring, and returns 404 for `/docs` and `/openapi.json`
so your API surface is not published.

Get a certificate with `sudo certbot --nginx -d YOUR.DOMAIN`.

### Option B - API key only, API clients on a LAN

For scripts that speak HTTP and do not need the browser UI:

```bash
sudo sh -c 'echo KOKORO_API_KEY=<your-key> > /etc/kokoro/tts.env'
sudo systemctl restart kokoro-tts
```

```bash
curl -H "X-API-Key: <your-key>" http://SERVER_IP:8000/voices
```

Remember the web UI at `/` will now return 401 in a browser.

### Option C - direct bind, trusted network only

```bash
sudo sed -i 's/--host 127.0.0.1/--host 0.0.0.0/' /etc/systemd/system/kokoro-tts.service
sudo systemctl restart kokoro-tts
```

Only if the host sits behind a firewall you control and you accept that anyone
who reaches the port can consume the CPU. `firewall-cmd` is already opened by
the installer, so this is reachable from anywhere the port is allowed.

---

## 6. Everyday commands

```bash
systemctl status kokoro-tts
systemctl restart kokoro-tts
journalctl -u kokoro-tts -f              # follow logs
journalctl -u kokoro-tts --since -1h     # last hour

curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

`/health` never requires the API key, so monitoring can use it unauthenticated.
It reports `ready`, `device`, `model`, `load_seconds`, and the voice count -
`load_seconds` being non-null is how you tell a warm engine from a cold one.

---

## 7. Resource notes

- **No sound card needed.** Kokoro renders to a byte array, not to ALSA/Pulse.
  That is the one thing it is better at than the system TTS stack, and it is why
  `PrivateDevices=yes` in the unit is safe. No `espeak` or ALSA packages needed
  either - `espeakng_loader` ships Linux `.so` binaries in the venv
  (`manylinux_2_17_x86_64`), which is what covers the non-English locales.
- **One worker, on purpose.** Each worker loads its own 330 MB copy of the
  model, and synthesis is serialised behind a lock inside the process anyway.
  `--workers 2` doubles memory for no throughput gain. For genuine parallelism,
  run N replicas behind nginx and let nginx balance.
- **`/preprocess` is not behind that lock.** The endpoint is a plain `def`, so
  Starlette runs it in the threadpool rather than on the event loop. Auto-format
  requests therefore overlap each other and overlap synthesis instead of
  stalling them, and the cached config is read without taking a lock. The catch is
  that it is still CPU work: a few very large documents at once will compete with
  synthesis for cores on a small box, and the default 40-thread threadpool will
  happily start more of them than the machine can serve. On a 2-core host, cap
  concurrent pastes at the nginx side if you see synthesis latency climb.
- **First request is slow.** The model loads lazily on the first `/speak` or
  `/warmup`, ~6 s on a decent CPU. The installer warms it once, but systemd
  restarts lose that, so use `Restart=always` plus a health check rather than
  assuming readiness.
- **Measure before trusting the numbers.** Throughput depends entirely on the
  CPU. Run it on the box:

  ```bash
  /opt/kokoro/.venv/bin/python tests/bench.py
  ```

---

## 8. Updating

```bash
# on the server
cd /opt/kokoro
git pull
sudo bash deploy/install-centos.sh
curl -s http://127.0.0.1:8000/health
```

The installer is idempotent, so `git pull` followed by a re-run refreshes the
code, leaves the venv and the model cache alone, and re-verifies the import
before touching systemd. For a routine code push that is all you need.

To copy changed files in from another machine instead, `scp` the managed
directories over the top and restart:

```bash
# from your PC
scp -r app deploy tests user@SERVER_IP:/tmp/kokoro-update/

# on the server
sudo cp -a /tmp/kokoro-update/. /opt/kokoro/
sudo /opt/kokoro/.venv/bin/python -m compileall -q /opt/kokoro/app
sudo systemctl restart kokoro-tts
curl -s http://127.0.0.1:8000/health
```

Replacing whole directories is safe because `/opt/kokoro` holds no runtime
state: the venv lives in `.venv/` and everything mutable lives under
`/var/lib/kokoro`. A full `bash deploy/install-centos.sh` also works and
re-verifies everything, but it
re-runs `uv python install` and the model warmup, so it is slower for a
routine code push.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `can't determine glibc version` / wheel errors | Pre-2.28 host. CentOS 7 cannot run this. See section 1. |
| `error while attempting to bind` | Something else owns the port: `ss -tlnp \| grep 8000`. |
| 401 in the browser but curl works | You set `KOKORO_API_KEY`; the UI cannot send headers. Unset it and use nginx Basic auth. |
| 500, `ffmpeg not found on PATH` | MP3 requested but ffmpeg is missing. `sudo dnf install ffmpeg-free`, or use `format=wav`. |
| 422, `needs a language pack` | Requested a `jf_*`/`zf_*` voice. See the language-pack note in README section 5. |
| First request hangs, then works | Lazy model load. Hit `/warmup` or wait out the ~6 s. |
| `403` from nginx | SELinux is denying nginx the loopback proxy. `sudo setsebool -P httpd_can_proxy_connect 1`. |
| Out of memory | Model is ~500 MB resident per worker. Raise `MemoryMax` or keep `--workers 1`. |
| Service restarts in a loop | `journalctl -u kokoro-tts -n 50`. Usually a `ProtectSystem=strict` write to a path outside `ReadWritePaths`. |

---

## 10. Container alternative

If you would rather not touch the host Python at all, the same tree runs in a
container. Note the glibc constraint still applies, because it is the base image
that matters:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "kokoro>=0.9.4" "misaki[en]>=0.9.4" \
      soundfile espeakng-loader fastapi "uvicorn[standard]" python-multipart
ENV HF_HOME=/models
VOLUME /models
EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000"]
```

Mount `/models` from a named volume or the 330 MB download is lost on every
redeploy, and mount `/app/output` if you want `save=true` to survive.
