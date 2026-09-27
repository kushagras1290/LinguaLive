"""Inference engine contracts.

Every model-backed capability sits behind one of these abstract bases. Nothing
outside ``app.engines`` may import ``torch``, ``transformers``, ``faster_whisper``
or ``gtts`` directly — routes and streaming code depend only on these interfaces.

That indirection is what lets the same HTTP API run models locally during
development and call a remote host in production, without clients or the
route handlers changing at all.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.logging_conf import get_logger

__all__ = [
    "TranscriptionSegment",
    "TranscriptionResult",
    "TranslationResult",
    "SpeechResult",
    "Engine",
    "ASREngine",
    "MTEngine",
    "TTSEngine",
    "EngineSet",
]

_LOG = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One contiguous span of recognised speech."""

    start: float
    end: float
    text: str
    #: Average token log-probability; ``None`` when the engine does not report it.
    avg_logprob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": self.avg_logprob,
        }


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """The outcome of a speech-recognition call."""

    text: str
    language: str
    #: Detected-language confidence in [0, 1]; ``None`` when not reported.
    language_probability: float | None = None
    duration: float = 0.0
    segments: tuple[TranscriptionSegment, ...] = ()
    engine: str = "unknown"

    @property
    def is_empty(self) -> bool:
        """True when no speech was recognised."""
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": round(self.duration, 3),
            "segments": [segment.to_dict() for segment in self.segments],
            "engine": self.engine,
        }


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """The outcome of a machine-translation call."""

    text: str
    source_lang: str
    target_lang: str
    engine: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {
            "text": self.text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "engine": self.engine,
        }


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """Synthesised audio and the metadata needed to serve it."""

    audio: bytes
    mime_type: str
    language: str
    engine: str = "unknown"

    @property
    def size_bytes(self) -> int:
        """Length of the encoded audio payload."""
        return len(self.audio)


class Engine(ABC):
    """Base class providing thread-safe lazy model loading.

    Model construction is deferred until first use and guarded by a lock, so
    concurrent requests arriving during a cold start cannot load the same weights
    twice. :meth:`_load` runs exactly once per instance.
    """

    #: Stable identifier reported in API responses and health checks.
    name: str = "engine"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._load_lock: threading.Lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model is resident in memory."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load the model if it is not already loaded.

        Raises:
            ModelLoadError: If loading fails.
        """
        if self._loaded:
            return
        with self._load_lock:
            # Re-check: another thread may have loaded while we waited.
            if self._loaded:
                return
            _LOG.info("Loading engine", extra={"engine": self.name})
            self._load()
            self._loaded = True
            _LOG.info("Engine ready", extra={"engine": self.name})

    @abstractmethod
    def _load(self) -> None:
        """Construct the underlying model. Called at most once per instance."""

    def describe(self) -> dict[str, Any]:
        """Return engine metadata for the health endpoint."""
        return {"name": self.name, "loaded": self._loaded}


class ASREngine(Engine):
    """Speech recognition."""

    name = "asr"

    @abstractmethod
    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe mono float32 audio in [-1, 1].

        Args:
            audio: 1-D float32 samples.
            sample_rate: Sample rate of ``audio`` in Hz.
            language: Whisper language code, or ``None`` to auto-detect.

        Returns:
            The transcription, possibly with empty text if no speech was found.

        Raises:
            InferenceError: If transcription fails.
        """


class MTEngine(Engine):
    """Machine translation."""

    name = "mt"

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        num_beams: int | None = None,
    ) -> TranslationResult:
        """Translate ``text`` between two application language codes.

        Implementations must return the input unchanged when the source and
        target languages are identical.

        Args:
            text: Source text.
            source_lang: Application language code of ``text``.
            target_lang: Application language code to translate into.
            num_beams: Override the configured beam width. The streaming path
                lowers this to trade a little quality for latency. Engines that
                do not decode locally may ignore it.

        Returns:
            The translation.

        Raises:
            InferenceError: If translation fails.
            UnknownLanguageError: If either code is unknown.
        """


class TTSEngine(Engine):
    """Speech synthesis."""

    name = "tts"

    @abstractmethod
    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        """Render ``text`` as speech audio.

        Args:
            text: Text to speak.
            language: Application language code.

        Returns:
            Encoded audio and its MIME type.

        Raises:
            InferenceError: If synthesis fails.
            UnsupportedCapabilityError: If the language cannot be spoken.
        """


@dataclass(slots=True)
class EngineSet:
    """The engines bound to a running application."""

    asr: ASREngine
    mt: MTEngine
    tts: TTSEngine
    warnings: list[str] = field(default_factory=list)

    def preload(self) -> None:
        """Eagerly load every engine so the first request is not slowed.

        Failures are logged and re-raised, because a configuration that cannot
        load its models should not start serving traffic.
        """
        for engine in (self.asr, self.mt, self.tts):
            engine.ensure_loaded()

    def describe(self) -> dict[str, Any]:
        """Return metadata for every engine, for the health endpoint."""
        return {
            "asr": self.asr.describe(),
            "mt": self.mt.describe(),
            "tts": self.tts.describe(),
        }
