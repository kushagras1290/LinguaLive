"""Custom exception hierarchy for the translation service.

Every exception carries an HTTP status and a stable machine-readable ``code`` so
the API layer can serialise failures without ever inspecting exception *types*
or leaking internal messages to clients.

No code in this package may raise a bare ``Exception``.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "TranslationAppError",
    "ConfigurationError",
    "EngineError",
    "EngineNotAvailableError",
    "ModelLoadError",
    "InferenceError",
    "InferenceTimeoutError",
    "AudioError",
    "AudioDecodeError",
    "EmptyAudioError",
    "AudioTooLongError",
    "PayloadTooLargeError",
    "LanguageError",
    "UnknownLanguageError",
    "UnsupportedCapabilityError",
    "RequestValidationError",
]


class TranslationAppError(Exception):
    """Base class for every error raised by this application.

    Attributes:
        message: Human-readable description, safe to return to a client.
        code: Stable machine-readable identifier (e.g. ``"unknown_language"``).
        http_status: HTTP status the API layer should respond with.
        details: Optional structured context included in the error payload.
    """

    default_message: Final[str] = "An unexpected application error occurred."
    code: str = "internal_error"
    http_status: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message: str = message or self.default_message
        self.details: dict[str, Any] = details or {}
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        """Serialise into the API's canonical error envelope."""
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details:
            payload["error"]["details"] = self.details
        return payload


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(TranslationAppError):
    """Invalid or missing configuration detected at startup.

    Raised during application construction so the process fails fast rather than
    serving traffic in a half-configured state.
    """

    default_message = "The application is misconfigured."
    code = "configuration_error"
    http_status = 500


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class EngineError(TranslationAppError):
    """Base class for inference-engine failures."""

    default_message = "An inference engine failed."
    code = "engine_error"
    http_status = 502


class EngineNotAvailableError(EngineError):
    """A required engine was not configured or its dependencies are missing."""

    default_message = "The requested inference engine is not available."
    code = "engine_unavailable"
    http_status = 503


class ModelLoadError(EngineError):
    """A model could not be loaded from disk or the remote hub."""

    default_message = "A model failed to load."
    code = "model_load_failed"
    http_status = 503


class InferenceError(EngineError):
    """A model loaded successfully but failed while processing a request."""

    default_message = "Inference failed."
    code = "inference_failed"
    http_status = 502


class InferenceTimeoutError(InferenceError):
    """An inference call exceeded its configured deadline."""

    default_message = "Inference timed out."
    code = "inference_timeout"
    http_status = 504


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
class AudioError(TranslationAppError):
    """Base class for audio-handling failures."""

    default_message = "The audio could not be processed."
    code = "audio_error"
    http_status = 400


class AudioDecodeError(AudioError):
    """The uploaded bytes are not decodable audio in any supported container."""

    default_message = (
        "The audio could not be decoded. Supported containers include WAV, WebM, "
        "Ogg, MP3, MP4/M4A and FLAC."
    )
    code = "audio_decode_failed"
    http_status = 415


class EmptyAudioError(AudioError):
    """The audio decoded successfully but contains no usable signal.

    Covers both zero-length input and pure digital silence, which would otherwise
    produce a division by zero during peak normalisation.
    """

    default_message = "The audio contains no audible speech."
    code = "audio_empty"
    http_status = 422


class AudioTooLongError(AudioError):
    """Decoded audio exceeds the configured maximum duration."""

    default_message = "The audio is longer than the configured limit."
    code = "audio_too_long"
    http_status = 413


class PayloadTooLargeError(TranslationAppError):
    """An upload exceeded the configured byte limit."""

    default_message = "The uploaded file is too large."
    code = "payload_too_large"
    http_status = 413


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
class LanguageError(TranslationAppError):
    """Base class for language-selection failures."""

    default_message = "Invalid language selection."
    code = "language_error"
    http_status = 400


class UnknownLanguageError(LanguageError):
    """A language code is not present in the canonical language registry."""

    default_message = "Unknown language code."
    code = "unknown_language"
    http_status = 400


class UnsupportedCapabilityError(LanguageError):
    """A known language does not support the requested capability.

    NLLB covers 202 languages, Whisper 100 and gTTS 68. Asking for speech
    synthesis in a translation-only language is a client error, not a crash.
    """

    default_message = "This language does not support the requested operation."
    code = "capability_unsupported"
    http_status = 422


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
class RequestValidationError(TranslationAppError):
    """The request body or form failed schema validation."""

    default_message = "The request payload is invalid."
    code = "validation_error"
    http_status = 400
