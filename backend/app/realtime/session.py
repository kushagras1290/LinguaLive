"""Per-connection state for a live translation session.

Holds the rolling audio buffer, drives the voice activity detector, and decides
when to run a partial or final transcription. Deliberately transport-agnostic —
it never touches a socket — so the logic can be unit tested without a browser.

The lifecycle of one utterance:

    speech onset -> accumulate -> every N seconds emit a PARTIAL
                 -> silence for the configured gap -> emit a FINAL, translate,
                    optionally synthesise, then clear the buffer

Partials are transcribe-only. Translating every partial would triple the compute
for text that is about to be replaced, and NLLB output churns badly on
half-finished sentences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.engines.base import EngineSet
from app.errors import TranslationAppError
from app.languages import Capability, require_capability, require_language
from app.logging_conf import get_logger
from app.realtime.vad import SpeechState, VoiceActivityDetector

__all__ = ["EventType", "StreamEvent", "SessionConfig", "StreamSession"]

_LOG = get_logger(__name__)

#: Guard against a client that never stops sending during one utterance.
_MAX_UTTERANCE_SECONDS: Final[float] = 30.0


class EventType(StrEnum):
    """Event kinds sent to the client."""

    READY = "ready"
    PARTIAL = "partial"
    FINAL = "final"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    LEVEL = "level"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One message bound for the client."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON transmission."""
        return {"type": str(self.type), **self.data}


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Client-supplied configuration for a streaming session.

    Attributes:
        source_lang: Application code of the spoken language, or ``None`` to
            auto-detect.
        target_lang: Application code to translate into.
        speak: Whether to synthesise each final translation.
    """

    source_lang: str | None
    target_lang: str
    speak: bool = False


class StreamSession:
    """Accumulates audio and produces transcription and translation events."""

    def __init__(
        self,
        *,
        settings: Settings,
        engines: EngineSet,
        config: SessionConfig,
    ) -> None:
        self._settings = settings
        self._engines = engines
        self._config = config

        self._sample_rate = settings.stream_sample_rate
        self._buffer: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._vad = VoiceActivityDetector(
            sample_rate=self._sample_rate,
            release_threshold=settings.stream_vad_threshold,
            silence_seconds=settings.stream_silence_seconds,
        )

        self._max_buffer_samples = int(settings.stream_window_seconds * self._sample_rate)
        self._min_chunk_samples = int(settings.stream_min_chunk_seconds * self._sample_rate)
        self._max_utterance_samples = int(_MAX_UTTERANCE_SECONDS * self._sample_rate)

        self._samples_since_partial = 0
        self._last_partial_text = ""
        self._utterance_index = 0
        self._started_at = time.monotonic()
        self._total_samples = 0

        # Resolved once so an unsupported language fails at session start rather
        # than midway through a stream.
        self._whisper_lang: str | None = None
        if config.source_lang is not None:
            self._whisper_lang = require_capability(
                config.source_lang, Capability.TRANSCRIBE
            ).whisper
        require_language(config.target_lang)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock duration of the session."""
        return time.monotonic() - self._started_at

    @property
    def is_expired(self) -> bool:
        """Whether the session has exceeded its configured lifetime."""
        return self.elapsed_seconds > self._settings.stream_max_session_seconds

    def ready_event(self) -> StreamEvent:
        """Build the handshake event confirming the negotiated parameters."""
        return StreamEvent(
            EventType.READY,
            {
                "sample_rate": self._sample_rate,
                "source_lang": self._config.source_lang,
                "target_lang": self._config.target_lang,
                "speak": self._config.speak,
                "frame_samples": self._vad.frame_length,
            },
        )

    def push(self, audio: NDArray[np.float32]) -> list[StreamEvent]:
        """Feed newly received audio and return any resulting events.

        Args:
            audio: Float32 samples in [-1, 1] at the session sample rate.

        Returns:
            Events to forward to the client, in order.
        """
        if audio.size == 0:
            return []

        events: list[StreamEvent] = []
        self._total_samples += audio.size

        frames = self._vad.process(audio)
        self._buffer = np.concatenate([self._buffer, audio])
        self._samples_since_partial += audio.size

        # Bound the buffer while no speech is in progress, so an idle connection
        # cannot grow it without limit.
        if self._vad.state is SpeechState.SILENCE and self._buffer.size > self._max_buffer_samples:
            self._buffer = self._buffer[-self._max_buffer_samples :]

        speech_ended = False
        for frame in frames:
            if frame.is_onset:
                events.append(
                    StreamEvent(EventType.SPEECH_START, {"energy": round(frame.energy, 5)})
                )
            if frame.is_offset:
                speech_ended = True
                events.append(
                    StreamEvent(EventType.SPEECH_END, {"energy": round(frame.energy, 5)})
                )

        if frames:
            # A cheap level meter for the UI's audio-reactive visuals.
            peak = max(frame.energy for frame in frames)
            events.append(StreamEvent(EventType.LEVEL, {"rms": round(peak, 5)}))

        if speech_ended:
            events.extend(self._finalise())
        elif self._should_emit_partial():
            events.extend(self._emit_partial())
        elif self._buffer.size > self._max_utterance_samples:
            # An utterance this long will not fit Whisper's window well; close it
            # off so the user still gets a result.
            _LOG.info("Utterance exceeded the maximum length; finalising early")
            events.extend(self._finalise())

        return events

    def flush(self) -> list[StreamEvent]:
        """Finalise any buffered speech, for use when the client disconnects."""
        if self._buffer.size < self._min_chunk_samples:
            return []
        return self._finalise()

    def _should_emit_partial(self) -> bool:
        """Whether enough new audio has arrived to justify a partial pass."""
        return (
            self._vad.state is SpeechState.SPEECH
            and self._buffer.size >= self._min_chunk_samples
            and self._samples_since_partial >= self._min_chunk_samples
        )

    def _transcribe_buffer(self) -> str:
        """Transcribe the current buffer, returning empty text on failure.

        Streaming must survive a single bad chunk, so engine errors are logged
        and swallowed here rather than tearing down the connection.
        """
        try:
            result = self._engines.asr.transcribe(
                np.ascontiguousarray(self._buffer),
                sample_rate=self._sample_rate,
                language=self._whisper_lang,
            )
        except TranslationAppError as exc:
            _LOG.warning("Streaming transcription failed", extra={"error": exc.message})
            return ""
        return result.text.strip()

    def _emit_partial(self) -> list[StreamEvent]:
        """Run a partial transcription and emit it if the text changed."""
        self._samples_since_partial = 0
        text = self._transcribe_buffer()
        if not text or text == self._last_partial_text:
            return []

        self._last_partial_text = text
        return [
            StreamEvent(
                EventType.PARTIAL,
                {
                    "text": text,
                    "utterance": self._utterance_index,
                    "seconds": round(self._buffer.size / self._sample_rate, 2),
                },
            )
        ]

    def _finalise(self) -> list[StreamEvent]:
        """Transcribe, translate and optionally speak the buffered utterance."""
        if self._buffer.size < self._min_chunk_samples:
            self._reset_utterance()
            return []

        text = self._transcribe_buffer()
        if not text:
            self._reset_utterance()
            return []

        events: list[StreamEvent] = []
        payload: dict[str, Any] = {
            "text": text,
            "utterance": self._utterance_index,
            "seconds": round(self._buffer.size / self._sample_rate, 2),
        }

        source_lang = self._config.source_lang or "en"
        try:
            translation = self._engines.mt.translate(
                text,
                source_lang=source_lang,
                target_lang=self._config.target_lang,
                # Latency matters more than the last few quality points here.
                num_beams=self._settings.nllb_stream_num_beams,
            )
            payload["translation"] = translation.text
            payload["source_lang"] = translation.source_lang
            payload["target_lang"] = translation.target_lang
        except TranslationAppError as exc:
            _LOG.warning("Streaming translation failed", extra={"error": exc.message})
            payload["translation"] = None
            payload["error"] = exc.message

        if self._config.speak and payload.get("translation"):
            import base64  # noqa: PLC0415 - only needed when speech is requested

            try:
                speech = self._engines.tts.synthesise(
                    payload["translation"], language=self._config.target_lang
                )
                payload["speech"] = {
                    "audio_base64": base64.b64encode(speech.audio).decode("ascii"),
                    "mime_type": speech.mime_type,
                    "engine": speech.engine,
                }
            except TranslationAppError as exc:
                _LOG.warning("Streaming synthesis failed", extra={"error": exc.message})
                payload["speech"] = None

        events.append(StreamEvent(EventType.FINAL, payload))
        self._reset_utterance()
        return events

    def _reset_utterance(self) -> None:
        """Clear the buffer and advance to the next utterance."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_partial = 0
        self._last_partial_text = ""
        self._utterance_index += 1
        self._vad.reset()

    def stats(self) -> dict[str, Any]:
        """Return session counters, logged when the connection closes."""
        return {
            "utterances": self._utterance_index,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "audio_seconds": round(self._total_samples / self._sample_rate, 1),
        }
