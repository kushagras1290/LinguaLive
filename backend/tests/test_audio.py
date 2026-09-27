"""Tests for audio decoding and conditioning.

These cover the three failure modes of a naive microphone path: int16 leaking
into the model, division by zero on silence, and the inability to decode
anything but RIFF/WAVE.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.audio import (
    PCM_DTYPE,
    condition_audio,
    decode_audio,
    duration_seconds,
    float32_to_pcm16_bytes,
    highpass_filter,
    is_silent,
    pcm16_bytes_to_float32,
    peak_level,
    peak_normalise,
    rms_level,
)
from app.errors import AudioDecodeError, AudioTooLongError, EmptyAudioError

from .conftest import SAMPLE_RATE, make_tone, make_wav_bytes


class TestDecodeAudio:
    """Container decoding."""

    def test_decodes_wav_to_float32_in_range(self, wav_bytes: bytes) -> None:
        """The core contract: float32 mono in [-1, 1].

        Handing Whisper int16 values up to +/-32767 saturates the log-mel
        spectrogram.
        """
        audio = decode_audio(wav_bytes, target_sample_rate=SAMPLE_RATE)

        assert audio.dtype == PCM_DTYPE
        assert audio.ndim == 1
        assert np.max(np.abs(audio)) <= 1.0
        assert audio.size == pytest.approx(SAMPLE_RATE, rel=0.02)

    def test_resamples_to_target_rate(self) -> None:
        """Input at a different rate is resampled, not passed through."""
        source = make_wav_bytes(make_tone(seconds=1.0, sample_rate=44_100), 44_100)
        audio = decode_audio(source, target_sample_rate=SAMPLE_RATE)
        assert audio.size == pytest.approx(SAMPLE_RATE, rel=0.05)

    def test_downmixes_stereo_to_mono(self) -> None:
        """Two-channel input becomes a single channel."""
        buffer = io.BytesIO()
        mono = (make_tone() * 32767).astype("<i2")
        interleaved = np.repeat(mono, 2)  # duplicate each sample across channels
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(interleaved.tobytes())

        audio = decode_audio(buffer.getvalue(), target_sample_rate=SAMPLE_RATE)
        assert audio.ndim == 1
        assert audio.size == pytest.approx(SAMPLE_RATE, rel=0.02)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(AudioDecodeError):
            decode_audio(b"", target_sample_rate=SAMPLE_RATE)

    def test_rejects_non_audio_bytes(self) -> None:
        """Random bytes are a decode failure, not a crash."""
        with pytest.raises(AudioDecodeError):
            decode_audio(b"this is not audio at all" * 50, target_sample_rate=SAMPLE_RATE)

    def test_rejects_pure_silence(self, silence_wav_bytes: bytes) -> None:
        """Silence raises a typed error instead of producing NaN downstream."""
        with pytest.raises(EmptyAudioError):
            decode_audio(silence_wav_bytes, target_sample_rate=SAMPLE_RATE)

    def test_enforces_duration_limit(self) -> None:
        long_audio = make_wav_bytes(make_tone(seconds=3.0))
        with pytest.raises(AudioTooLongError) as excinfo:
            decode_audio(long_audio, target_sample_rate=SAMPLE_RATE, max_seconds=1.0)
        assert excinfo.value.details["limit_seconds"] == 1.0


class TestPeakNormalise:
    """Peak normalisation, including the silence guard."""

    def test_scales_to_target_peak(self) -> None:
        quiet = make_tone(amplitude=0.05)
        normalised = peak_normalise(quiet, target_peak=0.95)
        assert peak_level(normalised) == pytest.approx(0.95, abs=1e-3)

    def test_silence_returns_unchanged_without_nan(self) -> None:
        """The bug: `audio / np.max(np.abs(audio))` divided by zero here."""
        silence = np.zeros(1000, dtype=PCM_DTYPE)
        result = peak_normalise(silence)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.array_equal(result, silence)

    def test_empty_array_is_safe(self) -> None:
        assert peak_normalise(np.zeros(0, dtype=PCM_DTYPE)).size == 0


class TestConditionAudio:
    """The full conditioning chain."""

    def test_output_always_within_unit_range(self) -> None:
        loud = (make_tone(amplitude=0.99) * 1.5).astype(PCM_DTYPE)
        conditioned = condition_audio(loud, SAMPLE_RATE)

        assert conditioned.dtype == PCM_DTYPE
        assert np.max(np.abs(conditioned)) <= 1.0

    def test_removes_dc_offset(self) -> None:
        """A constant offset is exactly what the high-pass filter exists for."""
        offset = make_tone(amplitude=0.3) + 0.4
        conditioned = condition_audio(offset.astype(PCM_DTYPE), SAMPLE_RATE)
        assert abs(float(np.mean(conditioned))) < 0.05

    def test_short_input_survives_filtering(self) -> None:
        """sosfiltfilt raises on input shorter than its padding requirement."""
        tiny = np.array([0.1, -0.2, 0.3], dtype=PCM_DTYPE)
        assert condition_audio(tiny, SAMPLE_RATE).size == 3


class TestHighpassFilter:
    """High-pass filtering edge cases."""

    def test_preserves_length(self, tone: np.ndarray) -> None:
        assert highpass_filter(tone, SAMPLE_RATE).size == tone.size

    def test_skips_when_cutoff_exceeds_nyquist(self, tone: np.ndarray) -> None:
        result = highpass_filter(tone, SAMPLE_RATE, cutoff_hz=SAMPLE_RATE)
        assert np.array_equal(result, tone)

    def test_empty_input_is_safe(self) -> None:
        assert highpass_filter(np.zeros(0, dtype=PCM_DTYPE), SAMPLE_RATE).size == 0


class TestPCM16Conversion:
    """PCM-16 conversion used by the streaming path."""

    def test_round_trip_preserves_signal(self, tone: np.ndarray) -> None:
        restored = pcm16_bytes_to_float32(float32_to_pcm16_bytes(tone))
        # 16-bit quantisation error is bounded by one LSB.
        assert np.max(np.abs(restored - tone)) < 1e-3

    def test_output_stays_within_unit_range(self) -> None:
        loud = np.array([1.0, -1.0, 0.999], dtype=PCM_DTYPE)
        restored = pcm16_bytes_to_float32(float32_to_pcm16_bytes(loud))
        assert np.max(np.abs(restored)) <= 1.0

    def test_empty_input_yields_empty_array(self) -> None:
        assert pcm16_bytes_to_float32(b"").size == 0

    def test_odd_length_buffer_is_rejected(self) -> None:
        """An odd byte count means the frame is not sample-aligned."""
        with pytest.raises(AudioDecodeError):
            pcm16_bytes_to_float32(b"\x01\x02\x03")


class TestLevelHelpers:
    """Level measurement helpers."""

    def test_peak_and_rms_of_empty_array(self) -> None:
        empty = np.zeros(0, dtype=PCM_DTYPE)
        assert peak_level(empty) == 0.0
        assert rms_level(empty) == 0.0

    def test_sine_rms_matches_theory(self) -> None:
        """RMS of a sine is amplitude / sqrt(2)."""
        assert rms_level(make_tone(amplitude=1.0)) == pytest.approx(
            1 / np.sqrt(2), abs=0.01
        )

    def test_silence_detection(self) -> None:
        assert is_silent(np.zeros(100, dtype=PCM_DTYPE))
        assert not is_silent(make_tone(amplitude=0.5))

    def test_duration_calculation(self, tone: np.ndarray) -> None:
        assert duration_seconds(tone, SAMPLE_RATE) == pytest.approx(1.0, abs=0.01)

    def test_duration_rejects_invalid_rate(self, tone: np.ndarray) -> None:
        with pytest.raises(ValueError):
            duration_seconds(tone, 0)
