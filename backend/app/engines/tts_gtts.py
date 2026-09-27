"""Speech synthesis via gTTS.

gTTS needs no API key and no local model, so it stays viable on any free hosting
tier. It covers 68 of NLLB's 202 languages; the language registry declares which,
and :func:`app.languages.gtts_code` rejects the rest with a 422 before any
network call is made.

Audio is returned as bytes rather than written to a file. A fixed temp path
would race under concurrent requests and leak a file per call.
"""

from __future__ import annotations

import io
from typing import Any, Final

from app.config import Settings
from app.engines.base import SpeechResult, TTSEngine
from app.errors import InferenceError
from app.languages import gtts_code
from app.logging_conf import get_logger

__all__ = ["GTTSEngine"]

_LOG = get_logger(__name__)

_MIME_TYPE: Final[str] = "audio/mpeg"

#: gTTS rejects very long input; the public endpoint also rate-limits it.
_MAX_SYNTHESIS_CHARS: Final[int] = 5_000


class GTTSEngine(TTSEngine):
    """Google Translate text-to-speech.

    Stateless: there is no model to load, so :meth:`_load` is a no-op and the
    engine reports itself as always ready.
    """

    name = "gtts"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def _load(self) -> None:
        """Verify gTTS is importable. There are no weights to load.

        Raises:
            InferenceError: If the package is missing.
        """
        try:
            import gtts  # noqa: PLC0415, F401
        except ImportError as exc:
            raise InferenceError(
                "gTTS is not installed. Run: pip install gTTS",
            ) from exc

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        """Render ``text`` as MP3 speech audio.

        Args:
            text: Text to speak.
            language: Application language code.

        Returns:
            MP3 bytes and their MIME type.

        Raises:
            InferenceError: If synthesis or the network call fails.
            UnknownLanguageError: If ``language`` is not in the registry.
            UnsupportedCapabilityError: If gTTS cannot speak ``language``.
        """
        stripped = text.strip()
        if not stripped:
            raise InferenceError("Cannot synthesise speech from empty text.")

        # Raises UnsupportedCapabilityError (422) rather than letting gTTS throw
        # an unhandled exception for unsupported languages.
        lang_code = gtts_code(language)

        if len(stripped) > _MAX_SYNTHESIS_CHARS:
            stripped = stripped[:_MAX_SYNTHESIS_CHARS]
            _LOG.warning(
                "Truncated text for synthesis",
                extra={"limit": _MAX_SYNTHESIS_CHARS, "language": language},
            )

        self.ensure_loaded()

        from gtts import gTTS  # noqa: PLC0415
        from gtts.tts import gTTSError  # noqa: PLC0415

        buffer = io.BytesIO()
        try:
            gTTS(text=stripped, lang=lang_code, slow=False).write_to_fp(buffer)
        except gTTSError as exc:
            raise InferenceError(
                f"Speech synthesis failed for {language!r}: {exc}",
                details={"language": language, "gtts_code": lang_code},
            ) from exc
        except (OSError, ValueError, AssertionError) as exc:
            raise InferenceError(
                f"Speech synthesis failed unexpectedly for {language!r}: {exc}",
                details={"language": language, "gtts_code": lang_code},
            ) from exc

        audio = buffer.getvalue()
        if not audio:
            raise InferenceError(
                "Speech synthesis produced no audio.",
                details={"language": language},
            )

        _LOG.debug(
            "Synthesised speech",
            extra={"language": language, "gtts_code": lang_code, "bytes": len(audio)},
        )
        return SpeechResult(
            audio=audio, mime_type=_MIME_TYPE, language=language, engine=self.name
        )

    def describe(self) -> dict[str, Any]:
        """Return engine metadata. gTTS needs no warm-up, so it is always ready."""
        return {**super().describe(), "provider": "google-translate", "requires_key": False}
