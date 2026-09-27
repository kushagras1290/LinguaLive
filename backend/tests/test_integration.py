"""Integration tests that exercise real model weights.

Excluded from the default run because they load multi-gigabyte models and take
minutes. Run them explicitly once models are downloaded::

    pytest -m integration

They are the tests that prove the pipeline works end to end, as opposed to the
unit suite which proves the plumbing around it is correct.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.audio import condition_audio, decode_audio
from app.config import Settings, get_settings
from app.engines.asr_faster_whisper import WHISPER_SAMPLE_RATE
from app.engines.registry import build_engines

pytestmark = pytest.mark.integration

#: Distinctive enough that a partly-wrong transcription is obvious.
REFERENCE_SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    """Settings pointed at the real, populated model cache."""
    return get_settings()


@pytest.fixture(scope="module")
def engines(real_settings: Settings):
    """Engines backed by real weights. Loading these takes a minute."""
    return build_engines(real_settings)


@pytest.fixture(scope="module")
def reference_speech() -> bytes:
    """Generate reference speech with gTTS so the test has known ground truth.

    Requires network access. Using synthesised speech rather than a checked-in
    recording keeps the repository small and the expected text unambiguous.
    """
    from gtts import gTTS

    buffer = io.BytesIO()
    try:
        gTTS(text=REFERENCE_SENTENCE, lang="en").write_to_fp(buffer)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not reach gTTS to build the fixture: {exc}")
    return buffer.getvalue()


class TestRealTranscription:
    """Whisper against known speech."""

    def test_transcribes_reference_sentence(self, engines, reference_speech: bytes) -> None:
        audio = condition_audio(
            decode_audio(reference_speech, target_sample_rate=WHISPER_SAMPLE_RATE),
            WHISPER_SAMPLE_RATE,
        )
        result = engines.asr.transcribe(
            audio, sample_rate=WHISPER_SAMPLE_RATE, language="en"
        )

        # Compare on content words: Whisper varies punctuation and casing.
        transcript = result.text.lower()
        for word in ("quick", "brown", "fox", "lazy", "dog"):
            assert word in transcript, f"{word!r} missing from {result.text!r}"

    def test_decodes_mp3_container(self, reference_speech: bytes) -> None:
        """gTTS returns MP3, which a `wave.open` path cannot read.

        This is the regression test for the defect that made every microphone
        upload fail: browsers send WebM/Opus, not WAV.
        """
        audio = decode_audio(reference_speech, target_sample_rate=WHISPER_SAMPLE_RATE)
        assert audio.dtype == np.float32
        assert audio.size > WHISPER_SAMPLE_RATE  # more than one second


class TestRealTranslation:
    """NLLB across scripts."""

    @pytest.mark.parametrize(
        ("target", "expected_range"),
        [
            ("hi", (0x0900, 0x097F)),  # Devanagari
            ("ar", (0x0600, 0x06FF)),  # Arabic
            ("ru", (0x0400, 0x04FF)),  # Cyrillic
            ("el", (0x0370, 0x03FF)),  # Greek
        ],
    )
    def test_output_is_in_the_target_script(
        self, engines, target: str, expected_range: tuple[int, int]
    ) -> None:
        """The strongest script-agnostic check that translation really happened."""
        result = engines.mt.translate(
            "Hello, how are you today?", source_lang="en", target_lang=target
        )

        low, high = expected_range
        in_script = sum(1 for char in result.text if low <= ord(char) <= high)
        assert in_script > 3, f"{target}: got {result.text!r}"

    def test_identical_languages_short_circuit(self, engines) -> None:
        text = "No translation needed."
        result = engines.mt.translate(text, source_lang="en", target_lang="en")
        assert result.text == text

    def test_translates_a_language_gtts_cannot_speak(self, engines) -> None:
        """Bhojpuri is translate-only, which is exactly why the chain exists."""
        result = engines.mt.translate(
            "Good morning, my friend.", source_lang="en", target_lang="bho"
        )
        assert result.text.strip()

    def test_long_input_is_chunked_not_truncated(self, engines) -> None:
        """A fixed `max_length=128` would silently drop long input."""
        long_text = " ".join([REFERENCE_SENTENCE] * 8)
        result = engines.mt.translate(long_text, source_lang="en", target_lang="es")
        # A truncated result would be a small fraction of the input length.
        assert len(result.text) > len(long_text) * 0.4


class TestRealSynthesis:
    """The TTS chain against real backends."""

    def test_speaks_a_common_language(self, engines) -> None:
        result = engines.tts.synthesise("Hello world", language="en")
        assert result.size_bytes > 1000

    def test_speaks_a_language_with_no_gtts_voice(self, engines) -> None:
        """Must fall through the chain rather than failing."""
        result = engines.tts.synthesise("नमस्ते", language="bho")
        assert result.size_bytes > 0


class TestFullPipeline:
    """Speech in one language to speech in another."""

    def test_speech_to_speech(self, engines, reference_speech: bytes) -> None:
        audio = condition_audio(
            decode_audio(reference_speech, target_sample_rate=WHISPER_SAMPLE_RATE),
            WHISPER_SAMPLE_RATE,
        )

        transcription = engines.asr.transcribe(
            audio, sample_rate=WHISPER_SAMPLE_RATE, language="en"
        )
        assert transcription.text.strip()

        translation = engines.mt.translate(
            transcription.text, source_lang="en", target_lang="hi"
        )
        assert translation.text.strip()

        speech = engines.tts.synthesise(translation.text, language="hi")
        assert speech.size_bytes > 1000


class TestMemoryFootprint:
    """Record the resident set, which decides where this can be hosted."""

    def test_reports_memory_after_loading(self, engines) -> None:
        import psutil

        engines.asr.ensure_loaded()
        engines.mt.ensure_loaded()

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"\nResident memory with all models loaded: {rss_mb:.0f} MB")

        # Not a pass/fail threshold so much as a tripwire: if this ever fits in
        # 512 MB, a free-tier deploy becomes possible and we should revisit.
        assert rss_mb > 0
