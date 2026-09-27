"""Tests for the text-to-speech pipeline we built.

Covers text normalisation, the from-scratch formant synthesiser, and the
fallback chain that guarantees speech is always producible.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.engines.base import SpeechResult, TTSEngine
from app.engines.tts_chain import ChainTTSEngine
from app.engines.tts_formant import FormantTTSEngine
from app.errors import (
    InferenceError,
    UnknownLanguageError,
    UnsupportedCapabilityError,
)
from app.tts.normalise import expand_number, normalise_text
from app.tts.synth import (
    SAMPLE_RATE,
    FormantSynthesiser,
    SynthesisVoice,
    text_to_phonemes,
)
from app.tts.synth import _grapheme_to_phonemes as grapheme_to_phonemes


class TestNumberExpansion:
    """Numbers must be spoken, not spelled or skipped."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0", "zero"),
            ("7", "seven"),
            ("13", "thirteen"),
            ("21", "twenty-one"),
            ("100", "one hundred"),
            ("342", "three hundred and forty-two"),
            ("1000", "one thousand"),
            ("2026", "two thousand twenty-six"),
        ],
    )
    def test_english_integers(self, value: str, expected: str) -> None:
        assert expand_number(value, "en") == expected

    def test_negative_numbers(self) -> None:
        assert expand_number("-5", "en") == "minus five"

    def test_decimals(self) -> None:
        assert expand_number("3.5", "en") == "three point five"

    @pytest.mark.parametrize(
        ("language", "expected"),
        [("es", "veintiuno"), ("fr", "vingt et un"), ("de", "einundzwanzig")],
    )
    def test_other_languages(self, language: str, expected: str) -> None:
        assert expand_number("21", language) == expected

    def test_unsupported_language_falls_back_to_digits(self) -> None:
        """Digit reading is always intelligible even when not idiomatic."""
        assert expand_number("21", "sw") == "two one"

    def test_very_large_numbers_read_as_digits(self) -> None:
        result = expand_number("9" * 20, "en")
        assert "nine" in result


class TestNormalisation:
    """The full normalisation chain."""

    def test_expands_embedded_numbers(self) -> None:
        assert "twenty-five" in normalise_text("I have 25 apples", "en")

    def test_expands_currency_after_the_amount(self) -> None:
        """'$5' is spoken '5 dollars', not 'dollars 5'."""
        result = normalise_text("It costs $5", "en")
        assert "five" in result
        assert "dollars" in result
        assert result.index("five") < result.index("dollars")

    def test_expands_symbols(self) -> None:
        result = normalise_text("50% off & more", "en")
        assert "percent" in result
        assert "and" in result

    def test_replaces_urls(self) -> None:
        result = normalise_text("Visit https://example.com now", "en")
        assert "https" not in result
        assert "link" in result

    def test_expands_email_addresses(self) -> None:
        result = normalise_text("mail me at a@b.com", "en")
        assert "@" not in result

    def test_expands_abbreviations(self) -> None:
        assert "doctor" in normalise_text("Dr. Smith", "en")

    def test_strips_markup_characters(self) -> None:
        result = normalise_text("**bold** and _italic_", "en")
        assert "*" not in result
        assert "_" not in result

    def test_collapses_repeated_punctuation(self) -> None:
        assert "!!!" not in normalise_text("Wow!!! Really???", "en")

    def test_empty_input_yields_empty_output(self) -> None:
        assert normalise_text("", "en") == ""
        assert normalise_text("   ", "en") == ""

    def test_preserves_non_latin_text(self) -> None:
        """Normalisation must not mangle the scripts most languages use."""
        assert "नमस्ते" in normalise_text("नमस्ते", "hi")
        assert "مرحبا" in normalise_text("مرحبا", "ar")


class TestGraphemeToPhoneme:
    """The letter-to-sound rules."""

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("hello", ["hh", "ah", "l", "ow"]),
            ("world", ["w", "er", "l", "d"]),
            ("my", ["m", "ay"]),
            ("yes", ["y", "eh", "s"]),
            ("city", ["s", "ih", "t", "iy"]),
            ("gem", ["jh", "eh", "m"]),
        ],
    )
    def test_known_words(self, word: str, expected: list[str]) -> None:
        assert grapheme_to_phonemes(word) == expected

    def test_y_is_a_vowel_word_internally(self) -> None:
        """'y' is a consonant only word-initially or before a vowel."""
        assert "y" not in grapheme_to_phonemes("system")
        assert grapheme_to_phonemes("yes")[0] == "y"

    def test_soft_c_and_g_before_front_vowels(self) -> None:
        assert grapheme_to_phonemes("cent")[0] == "s"
        assert grapheme_to_phonemes("cat")[0] == "k"

    def test_degeminates_doubled_consonants(self) -> None:
        """English spells 'happy' with two p's but pronounces one."""
        assert grapheme_to_phonemes("happy").count("p") == 1

    def test_never_returns_empty(self) -> None:
        assert grapheme_to_phonemes("'''") != []

    def test_phoneme_sequence_is_bracketed_by_silence(self) -> None:
        phonemes = text_to_phonemes("hello world")
        assert phonemes[0] == "sil"
        assert phonemes[-1] == "sil"


class TestFormantSynthesiser:
    """Our from-scratch DSP synthesiser."""

    @pytest.fixture
    def synthesiser(self) -> FormantSynthesiser:
        # Function-scoped: construction is cheap (no weights) and a fresh
        # instance keeps the seeded RNG deterministic per test.
        return FormantSynthesiser()

    def test_produces_float32_in_range(self, synthesiser: FormantSynthesiser) -> None:
        audio = synthesiser.synthesise("hello world")
        assert audio.dtype == np.float32
        assert audio.size > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_no_nan_or_inf(self, synthesiser: FormantSynthesiser) -> None:
        audio = synthesiser.synthesise("testing one two three")
        assert not np.any(np.isnan(audio))
        assert not np.any(np.isinf(audio))

    def test_longer_text_yields_longer_audio(
        self, synthesiser: FormantSynthesiser
    ) -> None:
        short = synthesiser.synthesise("hi")
        long = synthesiser.synthesise("hello there this is a much longer sentence")
        assert long.size > short.size

    def test_empty_text_yields_no_audio(self, synthesiser: FormantSynthesiser) -> None:
        assert synthesiser.synthesise("").size == 0

    def test_output_is_deterministic(self, synthesiser: FormantSynthesiser) -> None:
        """A seeded RNG makes the synthesiser reproducible, which is testable."""
        first = FormantSynthesiser().synthesise("hello")
        second = FormantSynthesiser().synthesise("hello")
        assert np.array_equal(first, second)

    def test_fades_prevent_boundary_clicks(
        self, synthesiser: FormantSynthesiser
    ) -> None:
        audio = synthesiser.synthesise("hello world")
        assert abs(float(audio[0])) < 0.05
        assert abs(float(audio[-1])) < 0.05

    def test_runs_faster_than_real_time(self, synthesiser: FormantSynthesiser) -> None:
        """The fallback is useless if it cannot keep up with speech."""
        import time

        text = "this sentence exists purely to measure synthesis throughput"
        started = time.perf_counter()
        audio = synthesiser.synthesise(text)
        elapsed = time.perf_counter() - started

        assert elapsed < audio.size / SAMPLE_RATE

    def test_voice_parameters_change_output(
        self, synthesiser: FormantSynthesiser
    ) -> None:
        default = synthesiser.synthesise("hello")
        deep = synthesiser.synthesise(
            "hello", voice=SynthesisVoice(base_pitch_hz=70.0, formant_shift=0.85)
        )
        assert not np.array_equal(default, deep)


class TestFormantEngine:
    """The formant synthesiser behind the engine interface."""

    @pytest.fixture
    def engine(self, settings: Settings) -> FormantTTSEngine:
        return FormantTTSEngine(settings)

    def test_produces_wav_audio(self, engine: FormantTTSEngine) -> None:
        result = engine.synthesise("hello world", language="en")
        assert result.mime_type == "audio/wav"
        assert result.audio.startswith(b"RIFF")
        assert result.size_bytes > 0

    def test_works_for_languages_with_no_neural_voice(
        self, engine: FormantTTSEngine
    ) -> None:
        """The whole point of the fallback: it never refuses a known language."""
        result = engine.synthesise("test", language="bho")
        assert result.size_bytes > 0

    def test_rejects_empty_text(self, engine: FormantTTSEngine) -> None:
        with pytest.raises(InferenceError):
            engine.synthesise("   ", language="en")

    def test_rejects_unknown_language(self, engine: FormantTTSEngine) -> None:
        with pytest.raises(UnknownLanguageError):
            engine.synthesise("hello", language="not-a-language")

    def test_needs_no_network_or_weights(self, engine: FormantTTSEngine) -> None:
        described = engine.describe()
        assert described["requires_network"] is False
        assert described["requires_weights"] is False


class _AlwaysFails(TTSEngine):
    """A backend that always raises, for exercising the chain."""

    name = "always_fails"

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error
        self.call_count = 0

    def _load(self) -> None:
        return None

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        self.call_count += 1
        raise self._error


class _AlwaysWorks(TTSEngine):
    """A backend that always succeeds."""

    name = "always_works"

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def _load(self) -> None:
        return None

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        self.call_count += 1
        return SpeechResult(
            audio=b"OK", mime_type="audio/wav", language=language, engine=self.name
        )


class TestChain:
    """The MMS -> gTTS -> formant fallback chain."""

    def test_uses_first_backend_when_it_succeeds(self, settings: Settings) -> None:
        primary = _AlwaysWorks()
        secondary = _AlwaysWorks()
        chain = ChainTTSEngine(settings, (primary, secondary))

        chain.synthesise("hello", language="en")

        assert primary.call_count == 1
        assert secondary.call_count == 0

    def test_falls_through_on_missing_voice(self, settings: Settings) -> None:
        primary = _AlwaysFails(UnsupportedCapabilityError("no voice"))
        fallback = _AlwaysWorks()
        chain = ChainTTSEngine(settings, (primary, fallback))

        result = chain.synthesise("hello", language="bho")

        assert result.engine == "always_works"
        assert fallback.call_count == 1

    def test_falls_through_on_inference_failure(self, settings: Settings) -> None:
        primary = _AlwaysFails(InferenceError("network down"))
        fallback = _AlwaysWorks()
        chain = ChainTTSEngine(settings, (primary, fallback))

        assert chain.synthesise("hello", language="en").engine == "always_works"

    def test_unknown_language_aborts_immediately(self, settings: Settings) -> None:
        """Every backend would reject it identically, so do not retry."""
        primary = _AlwaysFails(UnknownLanguageError("nope"))
        fallback = _AlwaysWorks()
        chain = ChainTTSEngine(settings, (primary, fallback))

        with pytest.raises(UnknownLanguageError):
            chain.synthesise("hello", language="xx")
        assert fallback.call_count == 0

    def test_raises_when_every_backend_fails(self, settings: Settings) -> None:
        chain = ChainTTSEngine(
            settings,
            (_AlwaysFails(InferenceError("a")), _AlwaysFails(InferenceError("b"))),
        )
        with pytest.raises(InferenceError) as excinfo:
            chain.synthesise("hello", language="en")
        assert len(excinfo.value.details["failures"]) == 2

    def test_rejects_empty_backend_list(self, settings: Settings) -> None:
        with pytest.raises(ValueError):
            ChainTTSEngine(settings, ())

    def test_real_chain_always_speaks(self, settings: Settings) -> None:
        """End to end: a language with no neural voice still produces audio."""
        chain = ChainTTSEngine(
            settings,
            (
                _AlwaysFails(UnsupportedCapabilityError("no MMS voice")),
                _AlwaysFails(UnsupportedCapabilityError("no gTTS voice")),
                FormantTTSEngine(settings),
            ),
        )
        result = chain.synthesise("hello world", language="bho")
        assert result.engine == "formant"
        assert result.audio.startswith(b"RIFF")
