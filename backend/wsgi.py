"""WSGI entrypoint.

Development::

    python backend/wsgi.py

Production (Linux)::

    gunicorn --chdir backend "wsgi:app" -k gevent -w 1 -b 0.0.0.0:$PORT

A single worker is intentional: the models are held in process memory, so extra
workers multiply resident memory rather than throughput. Scale with threads or a
second service, not with forks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python backend/wsgi.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.errors import TranslationAppError  # noqa: E402
from app.logging_conf import get_logger  # noqa: E402

_LOG = get_logger("wsgi")


def _build() -> object:
    """Construct the application, reporting configuration errors clearly."""
    try:
        return create_app()
    except TranslationAppError as exc:
        # Printed rather than logged: logging may not be configured yet if the
        # failure happened while reading settings.
        sys.stderr.write(f"Startup failed: {exc.message}\n")
        if exc.details:
            sys.stderr.write(f"Details: {exc.details}\n")
        raise SystemExit(2) from exc


app = _build()


def main() -> int:
    """Run a development server. Returns a process exit code."""
    settings = get_settings()

    # flask-sock needs a server that can hand over the WebSocket connection.
    # Werkzeug's threaded dev server does this; waitress does not, so it is only
    # used when explicitly requested.
    _LOG.info(
        "Starting development server",
        extra={
            "host": settings.host,
            "port": settings.port,
            "websocket": "ws://%s:%s/ws/stream" % (settings.host, settings.port),
        },
    )
    app.run(
        host=settings.host,
        port=settings.port,
        debug=False,       # never debug: it would expose the interactive console
        threaded=True,     # required so streaming does not block REST requests
        use_reloader=False,  # the reloader would load every model twice
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
