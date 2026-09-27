"""WebSocket endpoint for live streaming translation.

Protocol
--------
The client connects to ``/ws/stream`` and sends one JSON configuration frame::

    {"type": "config", "source_lang": "en", "target_lang": "hi", "speak": false}

The server replies with a ``ready`` event, after which the client streams **raw
little-endian PCM-16 mono** binary frames at the negotiated sample rate.

Raw PCM rather than ``MediaRecorder`` output is a deliberate choice: only the
first chunk of a WebM stream carries the container header, so subsequent chunks
are not independently decodable. Sending PCM from an ``AudioWorklet`` sidesteps
that entirely and removes per-chunk decode cost.

Text frames after configuration are treated as control messages; ``{"type":
"stop"}`` flushes any buffered speech and closes the session cleanly.
"""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, current_app

from app.audio import pcm16_bytes_to_float32
from app.errors import RequestValidationError, TranslationAppError
from app.logging_conf import bind_request_id, get_logger
from app.realtime.session import EventType, SessionConfig, StreamEvent, StreamSession

__all__ = ["register_websocket"]

_LOG = get_logger(__name__)

realtime_blueprint = Blueprint("realtime", __name__)


def _parse_config(raw: str) -> SessionConfig:
    """Parse the client's opening configuration frame.

    Raises:
        RequestValidationError: If the frame is malformed or missing a target.
    """
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestValidationError("The configuration frame is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RequestValidationError("The configuration frame must be a JSON object.")
    if payload.get("type") != "config":
        raise RequestValidationError("The first frame must have type 'config'.")

    target_lang = payload.get("target_lang")
    if not isinstance(target_lang, str) or not target_lang:
        raise RequestValidationError("'target_lang' is required.")

    source_lang = payload.get("source_lang")
    if source_lang in ("", "auto", None):
        source_lang = None
    elif not isinstance(source_lang, str):
        raise RequestValidationError("'source_lang' must be a string or null.")

    return SessionConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        speak=bool(payload.get("speak", False)),
    )


def _send(socket: Any, event: StreamEvent) -> None:
    """Send one event as JSON, ignoring a closed socket."""
    try:
        socket.send(json.dumps(event.to_dict(), ensure_ascii=False))
    except (ConnectionError, OSError):
        # The peer vanished mid-send; the receive loop will exit on its own.
        _LOG.debug("Dropped an event because the socket is closed")


def _send_error(socket: Any, message: str, code: str = "stream_error") -> None:
    """Send a structured error event."""
    _send(socket, StreamEvent(EventType.ERROR, {"code": code, "message": message}))


def register_websocket(app: Any, sock: Any) -> None:
    """Attach the ``/ws/stream`` route to the application.

    Args:
        app: The Flask application, used to push an application context so the
            handler can reach ``current_app.extensions``.
        sock: The configured :class:`flask_sock.Sock` instance.
    """

    @sock.route("/ws/stream")
    def stream(socket: Any) -> None:  # noqa: ANN401 - flask-sock supplies the type
        """Handle one live streaming connection for its whole lifetime."""
        request_id = bind_request_id()
        _LOG.info("Streaming connection opened")

        with app.app_context():
            settings = current_app.extensions["settings"]
            engines = current_app.extensions["engines"]

            # --- Handshake ------------------------------------------------- #
            try:
                opening = socket.receive(timeout=30)
            except (ConnectionError, OSError):
                _LOG.info("Client disconnected before configuring")
                return

            if opening is None:
                _LOG.info("Client sent no configuration frame")
                return
            if isinstance(opening, (bytes, bytearray)):
                _send_error(socket, "The first frame must be JSON configuration, not audio.")
                return

            try:
                config = _parse_config(opening)
                session = StreamSession(
                    settings=settings, engines=engines, config=config
                )
            except TranslationAppError as exc:
                _send_error(socket, exc.message, code=exc.code)
                return

            _send(socket, session.ready_event())
            _LOG.info(
                "Streaming session configured",
                extra={
                    "source_lang": config.source_lang,
                    "target_lang": config.target_lang,
                    "speak": config.speak,
                },
            )

            # --- Audio loop ------------------------------------------------ #
            try:
                while True:
                    if session.is_expired:
                        _send_error(
                            socket,
                            "The session exceeded its maximum duration.",
                            code="session_expired",
                        )
                        break

                    message = socket.receive(timeout=60)
                    if message is None:
                        break  # client closed, or 60s of total silence

                    if isinstance(message, str):
                        if not _handle_control(socket, session, message):
                            break
                        continue

                    try:
                        audio = pcm16_bytes_to_float32(bytes(message))
                    except TranslationAppError as exc:
                        _send_error(socket, exc.message, code=exc.code)
                        continue

                    for event in session.push(audio):
                        _send(socket, event)

            except (ConnectionError, OSError):
                _LOG.info("Streaming connection dropped")
            except TranslationAppError as exc:
                _LOG.error("Streaming session failed", extra={"error": exc.message})
                _send_error(socket, exc.message, code=exc.code)
            finally:
                # Emit anything still buffered so a mid-sentence disconnect does
                # not silently discard the user's last utterance.
                try:
                    for event in session.flush():
                        _send(socket, event)
                except TranslationAppError as exc:
                    _LOG.warning("Flush failed on close", extra={"error": exc.message})

                _LOG.info("Streaming connection closed", extra=session.stats())
                _send(socket, StreamEvent(EventType.CLOSED, session.stats()))


def _handle_control(socket: Any, session: StreamSession, raw: str) -> bool:
    """Process a text control frame.

    Returns:
        True to keep the connection open, False to close it.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        _send_error(socket, "Control frames must be valid JSON.", code="bad_control_frame")
        return True

    if not isinstance(payload, dict):
        _send_error(socket, "Control frames must be JSON objects.", code="bad_control_frame")
        return True

    match payload.get("type"):
        case "stop":
            for event in session.flush():
                _send(socket, event)
            return False
        case "flush":
            for event in session.flush():
                _send(socket, event)
            return True
        case "ping":
            _send(socket, StreamEvent(EventType.READY, {"pong": True}))
            return True
        case unknown:
            _send_error(
                socket,
                f"Unknown control frame type {unknown!r}.",
                code="unknown_control_frame",
            )
            return True
