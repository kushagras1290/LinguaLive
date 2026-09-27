"""Application factory.

Builds a configured Flask application with engines, CORS, structured logging,
WebSocket streaming and centralised error handling attached.

Using a factory rather than a module-level ``app = Flask(__name__)`` means tests
can construct isolated instances with overridden settings, and no model loading
happens merely because a module was imported.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, Response, jsonify, request

from app.config import Settings, get_settings
from app.engines.registry import build_engines
from app.errors import PayloadTooLargeError, TranslationAppError
from app.logging_conf import bind_request_id, configure_logging, get_logger

__all__ = ["create_app"]

_LOG = get_logger(__name__)

__version__ = "1.0.0"


def _register_error_handlers(app: Flask) -> None:
    """Attach handlers that render every failure as the same JSON envelope."""

    @app.errorhandler(TranslationAppError)
    def handle_application_error(error: TranslationAppError) -> tuple[Response, int]:
        """Render a known application error at its declared status."""
        # 5xx indicates our fault and deserves a stack trace; 4xx is the
        # client's and would only be noise in the logs.
        if error.http_status >= 500:
            _LOG.error(
                "Request failed",
                extra={"code": error.code, "path": request.path},
                exc_info=error,
            )
        else:
            _LOG.info(
                "Request rejected",
                extra={"code": error.code, "path": request.path, "reason": error.message},
            )
        return jsonify(error.to_payload()), error.http_status

    @app.errorhandler(404)
    def handle_not_found(_: Any) -> tuple[Response, int]:
        """Render unmatched routes as JSON rather than Flask's HTML page."""
        return (
            jsonify(
                {
                    "error": {
                        "code": "not_found",
                        "message": f"No route matches {request.path!r}.",
                    }
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(_: Any) -> tuple[Response, int]:
        """Render method mismatches as JSON."""
        return (
            jsonify(
                {
                    "error": {
                        "code": "method_not_allowed",
                        "message": f"{request.method} is not allowed on {request.path!r}.",
                    }
                }
            ),
            405,
        )

    @app.errorhandler(413)
    def handle_payload_too_large(_: Any) -> tuple[Response, int]:
        """Translate Werkzeug's upload limit into our error envelope."""
        error = PayloadTooLargeError()
        return jsonify(error.to_payload()), error.http_status

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception) -> tuple[Response, int]:
        """Catch-all so an unhandled exception never leaks a stack trace.

        Werkzeug HTTP exceptions are re-raised so their own handlers run.
        """
        from werkzeug.exceptions import HTTPException  # noqa: PLC0415

        if isinstance(error, HTTPException):
            raise error

        _LOG.error("Unhandled exception", extra={"path": request.path}, exc_info=error)
        return (
            jsonify(
                {
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected internal error occurred.",
                    }
                }
            ),
            500,
        )


def _register_request_hooks(app: Flask) -> None:
    """Attach per-request correlation ids."""

    @app.before_request
    def assign_request_id() -> None:
        """Adopt the client's correlation id, or mint one."""
        bind_request_id(request.headers.get("X-Request-ID"))

    @app.after_request
    def attach_request_id(response: Response) -> Response:
        """Echo the correlation id so clients can quote it in bug reports."""
        from app.logging_conf import request_id_var  # noqa: PLC0415

        response.headers["X-Request-ID"] = request_id_var.get()
        return response


def create_app(settings: Settings | None = None) -> Flask:
    """Construct a configured Flask application.

    Args:
        settings: Settings to use. Defaults to the process-wide singleton, which
            is read from the environment and a ``.env`` file.

    Returns:
        The configured application, with ``settings`` and ``engines`` available
        on ``app.extensions``.

    Raises:
        ConfigurationError: If configuration or engine construction fails.
    """
    resolved = settings or get_settings()
    configure_logging(
        level=resolved.log_level,
        json_output=resolved.log_format == "json",
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = resolved.max_upload_bytes
    app.config["JSON_SORT_KEYS"] = False
    # Non-ASCII must survive round-tripping: most of the 202 languages are not
    # Latin-script, and escaping them makes responses unreadable.
    app.json.ensure_ascii = False

    engines = build_engines(resolved)
    app.extensions["settings"] = resolved
    app.extensions["engines"] = engines

    from flask_cors import CORS  # noqa: PLC0415

    CORS(
        app,
        resources={r"/api/*": {"origins": resolved.cors_origin_list}},
        supports_credentials=False,
        expose_headers=["X-Request-ID", "X-TTS-Engine"],
    )

    from app.api.routes import api_blueprint  # noqa: PLC0415

    app.register_blueprint(api_blueprint)

    _register_request_hooks(app)
    _register_error_handlers(app)

    from flask_sock import Sock  # noqa: PLC0415

    sock = Sock(app)
    # Ping every 25s so intermediaries do not idle out a quiet connection.
    app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}

    from app.realtime.ws import register_websocket  # noqa: PLC0415

    register_websocket(app, sock)

    if resolved.eager_load_models:
        _LOG.info("Eagerly loading models")
        engines.preload()

    _LOG.info(
        "Application ready",
        extra={
            "version": __version__,
            "environment": str(resolved.environment),
            "cache_dir": str(resolved.model_cache_dir),
        },
    )
    return app
