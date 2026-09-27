"""Request and response schemas.

Validation lives here rather than inside route handlers so that every endpoint
rejects malformed input identically, with a structured error body instead of a
stack trace. Reading ``request.form`` directly in each handler produces the same
opaque 400 for every kind of mistake.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.errors import RequestValidationError
from app.languages import language_codes

__all__ = [
    "TranslateRequest",
    "SpeakRequest",
    "TranscribeOptions",
    "PipelineOptions",
    "parse_request",
]

#: Shared by every schema: reject unknown keys so client typos surface loudly
#: rather than being silently ignored.
_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _validate_language(value: str, field_name: str) -> str:
    """Check ``value`` against the language registry."""
    if value not in language_codes():
        raise ValueError(
            f"{field_name} must be a supported language code; {value!r} is not recognised"
        )
    return value


class TranslateRequest(BaseModel):
    """Body of ``POST /api/translate``."""

    model_config = _STRICT

    text: Annotated[str, Field(min_length=1, max_length=50_000)]
    source_lang: Annotated[str, Field(min_length=2, max_length=16)]
    target_lang: Annotated[str, Field(min_length=2, max_length=16)]

    @field_validator("source_lang")
    @classmethod
    def _check_source(cls, value: str) -> str:
        return _validate_language(value, "source_lang")

    @field_validator("target_lang")
    @classmethod
    def _check_target(cls, value: str) -> str:
        return _validate_language(value, "target_lang")


class SpeakRequest(BaseModel):
    """Body of ``POST /api/speak``."""

    model_config = _STRICT

    text: Annotated[str, Field(min_length=1, max_length=5_000)]
    lang: Annotated[str, Field(min_length=2, max_length=16)]

    @field_validator("lang")
    @classmethod
    def _check_lang(cls, value: str) -> str:
        return _validate_language(value, "lang")


class TranscribeOptions(BaseModel):
    """Form fields accompanying the audio upload on ``POST /api/transcribe``."""

    model_config = _STRICT

    #: ``None`` asks Whisper to auto-detect the spoken language.
    source_lang: str | None = None

    @field_validator("source_lang")
    @classmethod
    def _check_source(cls, value: str | None) -> str | None:
        if value is None or value == "" or value == "auto":
            return None
        return _validate_language(value, "source_lang")


class PipelineOptions(BaseModel):
    """Form fields for ``POST /api/pipeline`` (transcribe, translate, speak)."""

    model_config = _STRICT

    source_lang: str | None = None
    target_lang: Annotated[str, Field(min_length=2, max_length=16)]
    #: Whether to synthesise the translated text. Disable to save a round trip
    #: when the client only needs text.
    speak: bool = True

    @field_validator("source_lang")
    @classmethod
    def _check_source(cls, value: str | None) -> str | None:
        if value is None or value == "" or value == "auto":
            return None
        return _validate_language(value, "source_lang")

    @field_validator("target_lang")
    @classmethod
    def _check_target(cls, value: str) -> str:
        return _validate_language(value, "target_lang")


def parse_request[ModelT: BaseModel](model: type[ModelT], payload: dict[str, Any]) -> ModelT:
    """Validate ``payload`` against ``model``, translating pydantic errors.

    Args:
        model: The schema to validate against.
        payload: Raw request data.

    Returns:
        The validated model instance.

    Raises:
        RequestValidationError: If validation fails, with per-field details.
    """
    try:
        return model.model_validate(payload)
    except Exception as exc:  # pydantic raises ValidationError, a subclass
        errors = getattr(exc, "errors", None)
        details: dict[str, Any] = {}
        if callable(errors):
            details["fields"] = [
                {
                    "field": ".".join(str(part) for part in item.get("loc", ())) or "body",
                    "problem": item.get("msg", "invalid"),
                }
                for item in errors()
            ]
        raise RequestValidationError(
            "The request payload is invalid.", details=details
        ) from exc
