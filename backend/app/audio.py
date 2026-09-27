"""Audio decoding and conditioning.

A naive ``wave.open`` + normalise pipeline breaks the microphone path in three
ways, each of which this module guards against:

1. **Sample format.** Whisper's feature extractor requires **float32 normalised
   to [-1, 1]**. Feeding it raw int16 (±32767) produces a saturated log-mel
   spectrogram and badly degraded transcription.

2. **Silence.** Peak normalisation that divides by ``np.max(np.abs(audio))``
   without a zero guard turns a silent clip into ``NaN``/``inf`` and poisons the
   model input.

3. **Containers.** ``wave.open`` only reads RIFF/WAVE. Browsers record
   **WebM/Opus** via ``MediaRecorder``; tagging the Blob ``audio/wav`` on the
   client relabels it without transcoding, so a WAV-only server always raises
   ``wave.Error: file does not start with RIFF id``.

Decoding now goes through PyAV (FFmpeg), which handles every container a browser
or user might supply, and every sample leaving this module is float32 in [-1, 1].
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.errors import AudioDecodeError, AudioTooLongError, EmptyAudioError
from app.logging_conf import get_logger

__all__ = [
    "PCM_DTYPE",
    "decode_audio",
    "pcm16_bytes_to_float32",
    "float32_to_pcm16_bytes",
    "condition_audio",
    "peak_normalise",
    "highpass_filter",
    "rms_level",
    "peak_level",
    "is_silent",
    "duration_seconds",
]

_LOG = get_logger(__name__)

#: Every audio array crossing a module boundary uses this dtype.
PCM_DTYPE: Final[np.dtype] = np.dtype(np.float32)

#: Amplitude below which a clip is treated as containing no signal.
_SILENCE_EPSILON: Final[float] = 1e-5

#: Corner frequency for the rumble/DC-offset filter, in Hz.
_HIGHPASS_HZ: Final[float] = 80.0

#: Filter order. A 10th-order filter rings audibly on short clips; 4th order is
#: ample for removing DC and handling noise.
_HIGHPASS_ORDER: Final[int] = 4

#: Leave headroom after normalising so downstream processing cannot clip.
_NORMALISE_TARGET_PEAK: Final[float] = 0.95


def duration_seconds(audio: NDArray[np.float32], sample_rate: int) -> float:
    """Return the duration of ``audio`` in seconds."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return float(audio.shape[0]) / float(sample_rate)


def peak_level(audio: NDArray[np.float32]) -> float:
    """Return the absolute peak amplitude, or ``0.0`` for empty input."""
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def rms_level(audio: NDArray[np.float32]) -> float:
    """Return the root-mean-square amplitude, or ``0.0`` for empty input."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def is_silent(audio: NDArray[np.float32], threshold: float = _SILENCE_EPSILON) -> bool:
    """Return whether ``audio`` carries no usable signal."""
    return peak_level(audio) < threshold


def decode_audio(
    data: bytes,
    *,
    target_sample_rate: int,
    max_seconds: float | None = None,
) -> NDArray[np.float32]:
    """Decode arbitrary compressed or uncompressed audio to mono float32 PCM.

    Handles every container FFmpeg supports, notably the WebM/Opus produced by
    ``MediaRecorder`` and the Ogg/MP4 variants emitted by Firefox and Safari.
    Downmixing to mono and resampling are delegated to FFmpeg's resampler.

    Args:
        data: Raw bytes of an encoded audio file.
        target_sample_rate: Sample rate to resample to, in Hz.
        max_seconds: Reject audio longer than this. ``None`` disables the check.

    Returns:
        A 1-D float32 array in [-1, 1] at ``target_sample_rate``.

    Raises:
        AudioDecodeError: If the bytes are empty, contain no audio stream, or
            cannot be decoded.
        AudioTooLongError: If the decoded audio exceeds ``max_seconds``.
        EmptyAudioError: If the decoded audio is empty or pure silence.
    """
    if not data:
        raise AudioDecodeError("The uploaded audio is empty.")

    # Imported lazily so importing this module stays cheap and so the Hugging
    # Face cache environment is already configured before any heavy import.
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AudioDecodeError(
            "The 'av' package is required to decode audio but is not installed."
        ) from exc

    chunks: list[NDArray[np.float32]] = []
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise AudioDecodeError("The file contains no audio stream.")

            stream = container.streams.audio[0]
            # 'flt' = planar-free 32-bit float, which maps directly onto the
            # dtype Whisper expects, so no manual scaling is ever needed.
            resampler = av.AudioResampler(
                format="flt", layout="mono", rate=target_sample_rate
            )

            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))

            # Flush the resampler's internal buffer, otherwise the final few
            # milliseconds are silently dropped.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))

    except AudioDecodeError:
        raise
    except (av.AVError, ValueError, RuntimeError) as exc:
        raise AudioDecodeError(
            f"Could not decode the audio: {exc}",
            details={"bytes": len(data)},
        ) from exc

    if not chunks:
        raise EmptyAudioError("The audio decoded to zero samples.")

    audio = np.concatenate(chunks).astype(PCM_DTYPE, copy=False)

    seconds = duration_seconds(audio, target_sample_rate)
    if max_seconds is not None and seconds > max_seconds:
        raise AudioTooLongError(
            f"The audio is {seconds:.1f}s long, exceeding the {max_seconds:.0f}s limit.",
            details={"duration_seconds": round(seconds, 2), "limit_seconds": max_seconds},
        )

    if is_silent(audio):
        raise EmptyAudioError(
            "The audio contains only silence.",
            details={"duration_seconds": round(seconds, 2)},
        )

    _LOG.debug(
        "Decoded audio",
        extra={
            "bytes": len(data),
            "samples": int(audio.size),
            "duration_seconds": round(seconds, 2),
            "sample_rate": target_sample_rate,
        },
    )
    return audio


def pcm16_bytes_to_float32(data: bytes) -> NDArray[np.float32]:
    """Convert little-endian signed 16-bit PCM bytes to float32 in [-1, 1].

    Used by the WebSocket streaming path, where the browser sends raw PCM frames
    rather than an encoded container.

    Args:
        data: Raw interleaved mono PCM-16 bytes.

    Returns:
        A 1-D float32 array. Returns an empty array for empty input.

    Raises:
        AudioDecodeError: If the buffer length is not a whole number of samples.
    """
    if not data:
        return np.zeros(0, dtype=PCM_DTYPE)
    if len(data) % 2 != 0:
        raise AudioDecodeError(
            "PCM-16 payload has an odd byte length, so it is not sample-aligned.",
            details={"bytes": len(data)},
        )
    # 32768.0 (not 32767) matches the asymmetric int16 range and keeps the
    # result strictly within [-1, 1].
    samples = np.frombuffer(data, dtype="<i2").astype(PCM_DTYPE)
    return samples / 32768.0


def float32_to_pcm16_bytes(audio: NDArray[np.float32]) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian PCM-16 bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def peak_normalise(
    audio: NDArray[np.float32],
    *,
    target_peak: float = _NORMALISE_TARGET_PEAK,
) -> NDArray[np.float32]:
    """Scale ``audio`` so its loudest sample sits at ``target_peak``.

    Silent input is returned unchanged rather than producing ``NaN``.

    Args:
        audio: Float32 samples.
        target_peak: Desired peak amplitude in (0, 1].

    Returns:
        A float32 array. The input is returned unmodified when it is silent.
    """
    peak = peak_level(audio)
    if peak < _SILENCE_EPSILON:
        return audio
    return (audio * (target_peak / peak)).astype(PCM_DTYPE, copy=False)


def highpass_filter(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    cutoff_hz: float = _HIGHPASS_HZ,
    order: int = _HIGHPASS_ORDER,
) -> NDArray[np.float32]:
    """Remove DC offset and low-frequency rumble.

    Uses ``sosfiltfilt`` for zero phase distortion, and returns the input
    unchanged when it is too short for the filter's padding requirement.

    Args:
        audio: Float32 samples.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Corner frequency in Hz.
        order: Filter order.

    Returns:
        The filtered float32 array, always the same length as the input.
    """
    if audio.size == 0:
        return audio

    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        _LOG.warning(
            "Highpass cutoff is at or above Nyquist; skipping filter",
            extra={"cutoff_hz": cutoff_hz, "sample_rate": sample_rate},
        )
        return audio

    from scipy import signal  # noqa: PLC0415 - heavy import kept local

    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")

    # sosfiltfilt pads by 3 * (order * 2) samples; shorter input raises.
    min_length = 3 * (order * 2) + 1
    if audio.size <= min_length:
        return audio

    filtered = signal.sosfiltfilt(sos, audio)
    return np.asarray(filtered, dtype=PCM_DTYPE)


def condition_audio(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    apply_highpass: bool = True,
    normalise: bool = True,
) -> NDArray[np.float32]:
    """Apply the standard conditioning chain before inference.

    The chain is high-pass then peak-normalise, and the output is guaranteed to
    be float32 within [-1, 1] — exactly the contract Whisper's feature extractor
    expects.

    Args:
        audio: Float32 samples.
        sample_rate: Sample rate in Hz.
        apply_highpass: Whether to run the high-pass filter.
        normalise: Whether to peak-normalise.

    Returns:
        The conditioned float32 array.
    """
    conditioned = audio.astype(PCM_DTYPE, copy=False)
    if apply_highpass:
        conditioned = highpass_filter(conditioned, sample_rate)
    if normalise:
        conditioned = peak_normalise(conditioned)
    # Filtering can overshoot slightly; clip so the [-1, 1] contract always holds.
    return np.clip(conditioned, -1.0, 1.0).astype(PCM_DTYPE, copy=False)
