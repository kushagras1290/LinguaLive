"""Speech recognition via faster-whisper (CTranslate2).

Chosen over ``transformers``' ``WhisperForConditionalGeneration`` because the
CTranslate2 runtime is roughly 4x faster on CPU at int8 and uses a fraction of
the memory, which makes ``large-v3`` viable locally and keeps the default
``small`` model responsive enough to iterate against.

The engine accepts a decoded float32 array rather than a file path, so container
handling lives entirely in :mod:`app.audio` and never leaks into inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.engines.base import ASREngine, TranscriptionResult, TranscriptionSegment
from app.errors import InferenceError, ModelLoadError
from app.logging_conf import get_logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

__all__ = ["FasterWhisperEngine"]

_LOG = get_logger(__name__)

#: Whisper is trained on 16 kHz audio; anything else must be resampled first.
WHISPER_SAMPLE_RATE: Final[int] = 16_000

#: Discard segments whose no-speech probability exceeds this, guarding against
#: the hallucinated filler ("Thank you.", "Bye.") Whisper emits on near-silence.
_NO_SPEECH_THRESHOLD: Final[float] = 0.6


class FasterWhisperEngine(ASREngine):
    """Local Whisper inference backed by CTranslate2."""

    name = "faster_whisper"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._model: WhisperModel | None = None
        self._device: str = settings.resolved_whisper_device()
        self._compute_type: str = settings.resolved_whisper_compute_type(self._device)

    def _load(self) -> None:
        """Instantiate the CTranslate2 Whisper model.

        Raises:
            ModelLoadError: If the runtime or weights are unavailable.
        """
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed. Run: pip install faster-whisper",
            ) from exc

        model_id = self._settings.whisper_model
        _LOG.info(
            "Loading Whisper model",
            extra={
                "model": model_id,
                "device": self._device,
                "compute_type": self._compute_type,
                "cache_dir": str(self._settings.model_cache_dir),
            },
        )
        try:
            self._model = WhisperModel(
                model_id,
                device=self._device,
                compute_type=self._compute_type,
                download_root=str(self._settings.model_cache_dir / "whisper"),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelLoadError(
                f"Could not load Whisper model {model_id!r}: {exc}",
                details={"model": model_id, "device": self._device},
            ) from exc

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe conditioned float32 audio.

        Args:
            audio: 1-D float32 samples in [-1, 1].
            sample_rate: Must equal :data:`WHISPER_SAMPLE_RATE`.
            language: Whisper language code, or ``None`` to auto-detect.

        Returns:
            The transcription. ``text`` is empty when no speech was recognised,
            which is a normal outcome rather than an error.

        Raises:
            InferenceError: If the sample rate is wrong or decoding fails.
        """
        self.ensure_loaded()
        if self._model is None:  # pragma: no cover - ensure_loaded guarantees this
            raise InferenceError("Whisper model is not loaded.")

        if sample_rate != WHISPER_SAMPLE_RATE:
            raise InferenceError(
                f"Whisper requires {WHISPER_SAMPLE_RATE} Hz audio, got {sample_rate} Hz. "
                "Resample before calling transcribe().",
                details={"expected": WHISPER_SAMPLE_RATE, "received": sample_rate},
            )

        if audio.size == 0:
            return TranscriptionResult(text="", language=language or "en", engine=self.name)

        # CTranslate2 requires a contiguous float32 buffer.
        samples = np.ascontiguousarray(audio, dtype=np.float32)

        try:
            segments_iter, info = self._model.transcribe(
                samples,
                language=language,
                task="transcribe",
                beam_size=self._settings.whisper_beam_size,
                log_prob_threshold=self._settings.whisper_logprob_threshold,
                no_speech_threshold=_NO_SPEECH_THRESHOLD,
                condition_on_previous_text=False,  # prevents runaway repetition
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            # faster-whisper decodes lazily; materialising here keeps all
            # inference failures inside this try block.
            segments = tuple(
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    avg_logprob=float(segment.avg_logprob),
                )
                for segment in segments_iter
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise InferenceError(
                f"Whisper transcription failed: {exc}",
                details={"model": self._settings.whisper_model},
            ) from exc

        text = " ".join(segment.text for segment in segments if segment.text).strip()

        _LOG.debug(
            "Transcribed audio",
            extra={
                "segments": len(segments),
                "characters": len(text),
                "detected_language": info.language,
                "duration": round(float(info.duration), 2),
            },
        )

        return TranscriptionResult(
            text=text,
            language=info.language or (language or "en"),
            language_probability=float(info.language_probability)
            if info.language_probability is not None
            else None,
            duration=float(info.duration),
            segments=segments,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        """Return engine metadata including the resolved device and precision."""
        return {
            **super().describe(),
            "model": self._settings.whisper_model,
            "device": self._device,
            "compute_type": self._compute_type,
        }
