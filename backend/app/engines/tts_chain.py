"""A TTS engine that tries several backends in order.

Speech synthesis has more failure modes than the other capabilities: a language
may have no published voice, a download may fail, or a hosted service may rate
limit. Rather than surfacing each of those to the user, the chain degrades:

    MMS-TTS (neural, ~200 languages)
        -> gTTS (hosted, 68 languages)
            -> our formant synthesiser (offline, any language, always works)

The last link needs no weights and no network, so the chain never fails outright
for a language the registry knows about.

Errors that indicate a *client* mistake — an unknown language code — are not
retried down the chain, since every backend would reject them identically.
"""

from __future__ import annotations

from typing import Any, Final

from app.config import Settings
from app.engines.base import SpeechResult, TTSEngine
from app.errors import InferenceError, LanguageError, TranslationAppError
from app.logging_conf import get_logger

__all__ = ["ChainTTSEngine"]

_LOG = get_logger(__name__)


class ChainTTSEngine(TTSEngine):
    """Delegates synthesis to the first backend that succeeds."""

    name = "tts_chain"

    def __init__(self, settings: Settings, backends: tuple[TTSEngine, ...]) -> None:
        """Initialise the chain.

        Args:
            settings: Validated application settings.
            backends: Ordered backends, most preferred first. The last should be
                one that cannot fail, so the chain always yields audio.

        Raises:
            ValueError: If ``backends`` is empty.
        """
        super().__init__()
        if not backends:
            raise ValueError("ChainTTSEngine requires at least one backend.")
        self._settings = settings
        self._backends: Final[tuple[TTSEngine, ...]] = backends

    def _load(self) -> None:
        """Load only the primary backend.

        Fallbacks stay unloaded until they are actually needed, so a broken
        optional dependency never blocks startup.
        """
        self._backends[0].ensure_loaded()

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        """Synthesise via the first backend that succeeds.

        Args:
            text: Text to speak.
            language: Application language code.

        Returns:
            The first successful :class:`~app.engines.base.SpeechResult`.

        Raises:
            UnknownLanguageError: If ``language`` is not in the registry. Raised
                immediately without trying further backends.
            InferenceError: If every backend failed.
        """
        failures: list[str] = []

        for index, backend in enumerate(self._backends):
            is_last = index == len(self._backends) - 1
            try:
                result = backend.synthesise(text, language=language)
            except LanguageError as exc:
                # An unknown language code is a client error and a missing voice
                # is expected; only the former should abort the chain.
                if exc.code == "unknown_language":
                    raise
                failures.append(f"{backend.name}: {exc.message}")
                _LOG.info(
                    "TTS backend cannot handle language; trying next",
                    extra={"backend": backend.name, "language": language},
                )
                continue
            except TranslationAppError as exc:
                failures.append(f"{backend.name}: {exc.message}")
                log = _LOG.error if is_last else _LOG.warning
                log(
                    "TTS backend failed",
                    extra={
                        "backend": backend.name,
                        "language": language,
                        "error": exc.message,
                        "is_last": is_last,
                    },
                )
                continue

            if index > 0:
                _LOG.info(
                    "TTS served by fallback backend",
                    extra={
                        "backend": backend.name,
                        "language": language,
                        "skipped": index,
                    },
                )
            return result

        raise InferenceError(
            "Every speech-synthesis backend failed.",
            details={"language": language, "failures": failures},
        )

    def describe(self) -> dict[str, Any]:
        """Return metadata for the chain and each backend in order."""
        return {
            **super().describe(),
            "chain": [backend.name for backend in self._backends],
            "backends": [backend.describe() for backend in self._backends],
        }
