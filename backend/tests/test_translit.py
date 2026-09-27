"""Tests for script transliteration.

Transliteration is what makes the formant fallback usable for the ~180 of 202
supported languages that are not written in Latin script. Without it the
"always works" backend produced silence for all of them.
"""

from __future__ import annotations

import pytest

from app.tts.synth import FormantSynthesiser, text_to_phonemes
from app.tts.translit import is_transliterable, script_of, transliterate


class TestLatinPassthrough:
    """Latin text must survive untouched."""

    def test_ascii_is_unchanged(self) -> None:
        assert transliterate("Hello world") == "Hello world"

    def test_punctuation_and_digits_survive(self) -> None:
        assert transliterate("Hello, world! 42") == "Hello, world! 42"

    def test_accents_are_stripped_to_base_letters(self) -> None:
        assert transliterate("café naïve") == "cafe naive"

    def test_empty_input(self) -> None:
        assert transliterate("") == ""


class TestScriptCoverage:
    """Each supported script produces plausible Latin output."""

    @pytest.mark.parametrize(
        ("text", "expected_substring"),
        [
            ("नमस्ते", "namaste"),      # Devanagari
            ("दुनिया", "duniyaa"),      # Devanagari with vowel signs
            ("привет", "preevyet"),     # Cyrillic
            ("Доброго дня", "dobrogo"), # Cyrillic, capitalised
            ("κόσμε", "kosme"),         # Greek
            ("שלום", "shlom"),          # Hebrew
            ("こんにちは", "konnee"),      # Kana
            ("ᱡᱚᱦᱟᱨ", "johar"),         # Ol Chiki (the Santali greeting)
            ("ⴰⵣⵓⵍ", "azul"),           # Tifinagh (the Berber greeting)
        ],
    )
    def test_produces_expected_romanisation(
        self, text: str, expected_substring: str
    ) -> None:
        assert expected_substring in transliterate(text)

    @pytest.mark.parametrize(
        "text",
        [
            "नमस्ते दुनिया",      # Devanagari
            "Привет мир",          # Cyrillic
            "Γειά σου Κόσμε",      # Greek
            "مرحبا بالعالم",       # Arabic
            "שלום עולם",           # Hebrew
            "こんにちは",             # Kana
            "สวัสดี",              # Thai
            "བཀྲ་ཤིས་བདེ་ལེགས",    # Tibetan
            "မင်္ဂလာပါ",           # Myanmar
        ],
    )
    def test_every_script_yields_pronounceable_output(self, text: str) -> None:
        result = transliterate(text)
        assert result.strip(), f"{script_of(text)} produced nothing"
        assert any(char.isascii() and char.isalpha() for char in result)


class TestDevanagariRules:
    """Devanagari is an abugida, so the inherent vowel needs handling."""

    def test_virama_suppresses_inherent_vowel(self) -> None:
        """Without this, every conjunct gains a spurious 'a'."""
        # क + ् + ष should not render as "kasha"
        assert transliterate("क्ष").count("a") <= 1

    def test_vowel_sign_replaces_inherent_vowel(self) -> None:
        """'कि' is 'ki', not 'kai'."""
        assert transliterate("कि") == "ki"

    def test_bare_consonant_keeps_inherent_vowel(self) -> None:
        assert transliterate("क") == "ka"


class TestIdeographs:
    """Han characters carry no character-level reading."""

    def test_han_yields_nothing(self) -> None:
        """Honest failure beats inventing a pronunciation."""
        assert transliterate("你好世界") == ""

    def test_reports_not_transliterable(self) -> None:
        assert not is_transliterable("你好世界")

    def test_mixed_script_keeps_the_mappable_part(self) -> None:
        result = transliterate("hello 世界 world")
        assert "hello" in result
        assert "world" in result


class TestIsTransliterable:
    """The capability check."""

    @pytest.mark.parametrize("text", ["hello", "नमस्ते", "привет", "ᱡᱚᱦᱟᱨ"])
    def test_accepts_mappable_text(self, text: str) -> None:
        assert is_transliterable(text)

    @pytest.mark.parametrize("text", ["", "你好", "   ", "123"])
    def test_rejects_unmappable_text(self, text: str) -> None:
        assert not is_transliterable(text)


class TestEndToEndSynthesis:
    """The formant synthesiser must now speak non-Latin scripts."""

    @pytest.fixture
    def synthesiser(self) -> FormantSynthesiser:
        return FormantSynthesiser()

    @pytest.mark.parametrize(
        "text",
        [
            "नमस्ते दुनिया",
            "Привет мир",
            "Γειά σου Κόσμε",
            "مرحبا بالعالم",
            "བཀྲ་ཤིས་བདེ་ལེགས",
            "ᱡᱚᱦᱟᱨ",
        ],
    )
    def test_non_latin_produces_audio(
        self, synthesiser: FormantSynthesiser, text: str
    ) -> None:
        """The regression test for the fallback returning silence."""
        assert synthesiser.synthesise(text).size > 0

    def test_phonemes_are_generated_for_non_latin(self) -> None:
        phonemes = text_to_phonemes("नमस्ते")
        # More than just the two bracketing silences.
        assert len(phonemes) > 2

    def test_han_still_produces_no_audio(
        self, synthesiser: FormantSynthesiser
    ) -> None:
        """Unmappable input must fail honestly rather than emit noise."""
        assert synthesiser.synthesise("你好世界").size == 0
