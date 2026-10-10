"""How the HTTP service is started, and how many processes it runs.

Kept out of :mod:`app.server` so a worker does not import a launcher, and out of
``server_app.py`` so the source checkout (``run_server.bat``) and the frozen
``KokoroServer.exe`` share one implementation instead of drifting apart.

Two settings are easy to get wrong once there is more than one worker, and both
are handled here:

**Threads.** torch sizes its intra-op pool from the CPU count, so N workers
would between them ask for N x cpu_count threads. Measured on a 16-CPU box,
that oversubscription is what makes extra workers feel slower rather than
faster: two workers each holding every core finish two requests in 11.3 s where
one worker takes 14.7 s, and cutting each worker's share to half the cores is
what makes 4 workers worth starting at all. When the caller has not chosen a
thread count, the budget is divided evenly between the workers.

**Memory.** Every worker is its own interpreter with its own copy of the
weights, and the resident set is around 1.3 GB each - the 330 MB the weights
weigh on disk is not the number that matters. The worker count is therefore
capped at the CPU count, past which nothing improves and only the resident set
grows.
"""

from __future__ import annotations

import argparse
import os
import sys

WORKERS_ENV = "KOKORO_WORKERS"
THREADS_ENV = "KOKORO_THREADS"

# uvicorn only forks workers when the application is named as an import string.
# Handed an app object it exits with "You must pass the application as an import
# string to enable 'reload' or 'workers'", so this is what actually makes the
# worker count take effect - not a stylistic choice.
APP = "app.server:app"


def cpu_count() -> int:
    return os.cpu_count() or 1


def _cpus(cpus: int | None) -> int:
    """An injected CPU count wins; otherwise ask the platform, which can say None."""
    if cpus is None:
        return cpu_count()
    return max(1, cpus)


def resolve_workers(explicit=None, *, env=None, cpus=None) -> int:
    """How many server processes to start.

    ``explicit`` (a command-line value) wins over ``KOKORO_WORKERS``, and one
    worker - the behaviour every earlier version had - remains the default. A
    request larger than the CPU count is clamped rather than rejected: more
    processes than cores cannot help, and silently starting 32 of them on a
    16-core box would exhaust memory instead of failing loudly.
    """
    environ = os.environ if env is None else env
    raw = str(explicit).strip() if explicit is not None else ""
    if not raw:
        raw = environ.get(WORKERS_ENV, "").strip()
    if not raw:
        return 1
    try:
        workers = int(raw)
    except ValueError:
        raise ValueError(f"worker count must be a whole number, got {raw!r}") from None
    if workers < 1:
        raise ValueError(f"worker count must be at least 1, got {workers}")
    return min(workers, _cpus(cpus))


def apply_thread_budget(workers: int, *, env=None, cpus=None) -> int | None:
    """Divvy the CPU between workers when the operator has not.

    Exports ``KOKORO_THREADS``, which :mod:`app.engine` reads at import time, so
    this must run before the app is imported - which it does, because the
    spawned workers import it and inherit the environment of the process that
    started them.

    Returns the thread count that was exported, or ``None`` when nothing was
    changed: a single worker is left alone (its default of every core is
    correct, and capping it would only slow one user down), and an explicit
    ``KOKORO_THREADS`` always wins.
    """
    environ = os.environ if env is None else env
    if workers <= 1 or environ.get(THREADS_ENV, "").strip():
        return None
    per_worker = max(1, _cpus(cpus) // workers)
    environ[THREADS_ENV] = str(per_worker)
    return per_worker


def serve(host: str, port: int, workers: int = 1, *, log_level: str = "info",
          access_log: bool = False) -> None:
    """Start uvicorn, as ``workers`` processes sharing one listening socket.

    ``workers=1`` runs in this process, exactly as before. Anything higher binds
    the socket once and hands it to that many children, which uvicorn restarts
    if one dies.
    """
    import uvicorn

    # Idempotent: resolve_workers/apply_thread_budget may already have run in
    # the parent so it could print them, in which case this is a no-op.
    apply_thread_budget(workers)
    uvicorn.run(APP, host=host, port=port, workers=workers,
                log_level=log_level, access_log=access_log)


def main(argv: list[str] | None = None) -> int:
    """Command line used by ``run_server.bat``."""
    parser = argparse.ArgumentParser(
        prog="app.launcher",
        description="Start the Kokoro TTS web server (one worker unless asked for more).",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port to listen on (default: %(default)s)")
    parser.add_argument("--workers", default=None,
                        help=f"server processes; defaults to ${WORKERS_ENV}, or 1, "
                             f"and is capped at the CPU count")
    parser.add_argument("--no-access-log", action="store_true",
                        help="do not log one line per request")
    args = parser.parse_args(argv)

    try:
        workers = resolve_workers(args.workers)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    host = (args.host or "").strip() or "127.0.0.1"
    threads = apply_thread_budget(workers)

    # flush=True: run_server.bat is double-clicked, and a buffered banner would
    # not reach the console until the first request had already been served.
    print()
    print("  Kokoro TTS")
    print("  ---------------")
    print(f"  Web UI : http://{host}:{args.port}/")
    print(f"  API    : http://{host}:{args.port}/docs")
    if workers > 1:
        print(f"  Workers: {workers} processes"
              f"{f', {threads} threads each' if threads else ''}"
              f"  (~1.3 GB RAM each)")
    print()
    print("  First request downloads ~330 MB of weights. Ctrl+C to stop.")
    print(flush=True)

    serve(host, args.port, workers, access_log=not args.no_access_log)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
