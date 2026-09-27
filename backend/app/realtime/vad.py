"""Voice activity detection for the streaming path.

Decides where one utterance ends and the next begins, which is what lets the
server promote a rolling partial transcript to a final one at a natural boundary
instead of an arbitrary timer.

Uses short-term energy with hysteresis and a hangover counter rather than a
neural VAD: it costs microseconds, adds no dependency, and is accurate enough
when the decision only needs to find pauses. A single energy threshold without
hysteresis would flicker on every inter-word gap, so two thresholds and a
hangover are what make it usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.logging_conf import get_logger

__all__ = ["SpeechState", "VadFrame", "VoiceActivityDetector"]

_LOG = get_logger(__name__)

#: Analysis frame length. 30 ms is the standard for speech activity decisions.
FRAME_MS: Final[float] = 30.0

#: The onset threshold is this multiple of the release threshold, giving the
#: hysteresis that stops the detector chattering at the boundary.
_ONSET_MULTIPLIER: Final[float] = 2.5

#: Adaptation rate for the noise floor estimate, per silent frame.
_NOISE_ADAPT_RATE: Final[float] = 0.05

#: Noise floor is never allowed below this, so a perfectly clean signal does not
#: drive the threshold to zero and treat dither as speech.
_MIN_NOISE_FLOOR: Final[float] = 1e-4


class SpeechState(StrEnum):
    """Whether the detector currently believes speech is present."""

    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True, slots=True)
class VadFrame:
    """The detector's verdict for one analysis frame.

    Attributes:
        state: Speech or silence.
        energy: RMS energy of the frame.
        is_onset: True on the frame where speech began.
        is_offset: True on the frame where an utterance ended.
        silence_seconds: How long silence has persisted, once in silence.
    """

    state: SpeechState
    energy: float
    is_onset: bool
    is_offset: bool
    silence_seconds: float


class VoiceActivityDetector:
    """Energy-based speech detector with hysteresis and an adaptive noise floor.

    The detector is stateful and processes frames in order; create one per
    streaming session.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        release_threshold: float,
        silence_seconds: float,
        frame_ms: float = FRAME_MS,
    ) -> None:
        """Initialise the detector.

        Args:
            sample_rate: Input sample rate in Hz.
            release_threshold: RMS below which a frame counts as silence, before
                noise-floor adaptation is applied.
            silence_seconds: Silence required to declare an utterance finished.
            frame_ms: Analysis frame length in milliseconds.
        """
        self._sample_rate = sample_rate
        self._frame_length = max(1, int(sample_rate * frame_ms / 1000.0))
        self._frame_seconds = self._frame_length / sample_rate
        self._release_threshold = release_threshold
        self._onset_threshold = release_threshold * _ONSET_MULTIPLIER
        self._required_silence_frames = max(1, int(silence_seconds / self._frame_seconds))

        self._state = SpeechState.SILENCE
        self._silent_frames = 0
        self._noise_floor = release_threshold
        self._pending: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    @property
    def frame_length(self) -> int:
        """Number of samples per analysis frame."""
        return self._frame_length

    @property
    def state(self) -> SpeechState:
        """The detector's current verdict."""
        return self._state

    def reset(self) -> None:
        """Clear all state, as if the detector were newly constructed."""
        self._state = SpeechState.SILENCE
        self._silent_frames = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, audio: NDArray[np.float32]) -> list[VadFrame]:
        """Analyse a chunk of audio.

        Input need not align to frame boundaries; a partial trailing frame is
        buffered and combined with the next call.

        Args:
            audio: Float32 samples in [-1, 1].

        Returns:
            One :class:`VadFrame` per complete analysis frame, in order.
        """
        if audio.size:
            self._pending = np.concatenate([self._pending, audio])

        frames: list[VadFrame] = []
        while self._pending.size >= self._frame_length:
            frame = self._pending[: self._frame_length]
            self._pending = self._pending[self._frame_length :]
            frames.append(self._classify(frame))
        return frames

    def _classify(self, frame: NDArray[np.float32]) -> VadFrame:
        """Classify one complete frame and advance the state machine."""
        energy = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

        # Thresholds ride on the estimated noise floor so the detector adapts to
        # a noisy room instead of treating its hiss as continuous speech.
        onset = max(self._onset_threshold, self._noise_floor * _ONSET_MULTIPLIER)
        release = max(self._release_threshold, self._noise_floor * 1.4)

        is_onset = False
        is_offset = False

        if self._state is SpeechState.SILENCE:
            # Track the noise floor only while silent, so speech never inflates it.
            self._noise_floor = max(
                _MIN_NOISE_FLOOR,
                (1.0 - _NOISE_ADAPT_RATE) * self._noise_floor + _NOISE_ADAPT_RATE * energy,
            )
            if energy >= onset:
                self._state = SpeechState.SPEECH
                self._silent_frames = 0
                is_onset = True
        else:
            if energy < release:
                self._silent_frames += 1
                if self._silent_frames >= self._required_silence_frames:
                    self._state = SpeechState.SILENCE
                    is_offset = True
            else:
                # Hangover reset: brief dips inside a word must not end the
                # utterance, which is why the counter only clears on real energy.
                self._silent_frames = 0

        return VadFrame(
            state=self._state,
            energy=energy,
            is_onset=is_onset,
            is_offset=is_offset,
            silence_seconds=self._silent_frames * self._frame_seconds,
        )
