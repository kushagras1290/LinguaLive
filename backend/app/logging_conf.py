"""Structured logging configuration.

Emits newline-delimited JSON in production so logs are queryable by any log
aggregator, and a readable coloured format in development. A per-request
correlation id is attached to every record via a :class:`contextvars.ContextVar`,
so concurrent requests remain distinguishable in interleaved output.

``print()`` must not appear anywhere in this package; use ``logging`` instead.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final

__all__ = [
    "configure_logging",
    "get_logger",
    "request_id_var",
    "new_request_id",
    "bind_request_id",
]

#: Correlation id for the in-flight request or streaming session.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

#: Attributes present on every LogRecord; anything else is treated as structured
#: context supplied by the caller via ``extra=``.
_RESERVED_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "stacklevel", "thread", "threadName",
        "taskName",
    }
)

_ANSI: Final[dict[str, str]] = {
    "DEBUG": "\033[38;5;245m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;199m",
}
_ANSI_RESET: Final[str] = "\033[0m"


def new_request_id() -> str:
    """Generate a short, collision-resistant correlation id."""
    return uuid.uuid4().hex[:12]


def bind_request_id(request_id: str | None = None) -> str:
    """Bind a correlation id to the current context.

    Args:
        request_id: An id to adopt, or ``None`` to generate one.

    Returns:
        The bound correlation id.
    """
    resolved = request_id or new_request_id()
    request_id_var.set(resolved)
    return resolved


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract caller-supplied ``extra=`` fields from a record."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_RECORD_KEYS and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }

        extras = _extra_fields(record)
        if extras:
            payload["context"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # default=str keeps non-serialisable context (Paths, enums) from raising
        # inside the logging subsystem, where an exception would be swallowed.
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Render log records for human reading during development."""

    def __init__(self, *, use_colour: bool) -> None:
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        level = record.levelname
        if self._use_colour:
            level = f"{_ANSI.get(level, '')}{level:<8}{_ANSI_RESET}"
        else:
            level = f"{level:<8}"

        request_id = request_id_var.get()
        prefix = f"{timestamp} {level} {record.name}"
        if request_id != "-":
            prefix = f"{prefix} [{request_id}]"

        line = f"{prefix}  {record.getMessage()}"

        extras = _extra_fields(record)
        if extras:
            rendered = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
            line = f"{line}  ({rendered})"

        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _force_utf8_stream(stream: object) -> None:
    """Switch ``stream`` to UTF-8 with replacement, if it supports it.

    Text written to a console that cannot represent a character would otherwise
    raise ``UnicodeEncodeError`` from inside a log call. ``errors="replace"``
    guarantees logging degrades to a placeholder glyph instead of raising.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # Already detached or a stream that cannot be reconfigured; the
        # formatter's own escaping is the remaining safety net.
        pass


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Install the root logging handler. Safe to call more than once.

    Args:
        level: Minimum level name, e.g. ``"INFO"``.
        json_output: Emit JSON when true, human-readable text otherwise.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # Windows consoles default to cp1252, which cannot encode most of the 202
    # languages this service handles. Without this, logging a Hindi or Arabic
    # translation raises UnicodeEncodeError inside the logging subsystem.
    _force_utf8_stream(sys.stdout)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(use_colour=sys.stdout.isatty()))

    root.addHandler(handler)
    root.setLevel(level)

    # These libraries log a request line per HTTP call, which drowns out our own
    # records during model downloads.
    for noisy in ("urllib3", "httpx", "httpcore", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
