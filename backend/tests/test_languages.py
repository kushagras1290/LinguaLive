"""Tests for the language registry.

The registry is the fix for the capability-mismatch bugs, so these tests assert
the invariants that made the old dictionary dangerous: duplicate keys, codes the
downstream libraries do not actually accept, and unchecked capability claims.
"""

from __future__ import annotations

import pytest

from app.errors import UnknownLanguageError, UnsupportedCapabilityError
from app.languages import (
    Capability,
    all_languages,
    get_language,
    gtts_code,
    language_codes,
    nllb_code,
    require_capability,
    require_language,
    whisper_code,
)


class TestRegistryIntegrity:
    """Structural invariants of the registry itself."""

    def test_has_full_nllb_coverage(self) -> None:
        """NLLB-200 covers 202 languages; the registry should match."""
        assert len(all_languages()) == 202

    def test_codes_are_unique(self) -> None:
        """The old map defined 'gaz' twice, silently dropping one entry."""
        codes = [language.code for language in all_languages()]
        assert len(codes) == len(set(codes))

    def test_nllb_codes_are_unique(self) -> None:
        """Two app codes mapping to one NLLB code would make swap ambiguous."""
        nllb = [language.nllb for language in all_languages()]
        assert len(nllb) == len(set(nllb))

    def test_every_entry_has_a_scripted_nllb_code(self) -> None:
        for language in all_languages():
            assert "_" in language.nllb, f"{language.code} has a malformed NLLB code"

    def test_names_are_populated(self) -> None:
        for language in all_languages():
            assert language.name.strip()
            assert language.native_name.strip()

    def test_sorted_by_name(self) -> None:
        names = [language.name for language in all_languages()]
        assert names == sorted(names)


class TestExternalCodeValidity:
    """Every declared external code must be one the library actually accepts.

    This is the test that would have caught `sat_Beng` (Santali is Ol Chiki),
    and gTTS's `iw`/`jw` rather than `he`/`jv`.
    """

    def test_whisper_codes_are_real(self) -> None:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        for language in all_languages():
            if language.whisper is not None:
                assert language.whisper in LANGUAGES, (
                    f"{language.code}: Whisper does not know {language.whisper!r}"
                )

    def test_gtts_codes_are_real(self) -> None:
        from gtts.lang import tts_langs

        supported = tts_langs()
        for language in all_languages():
            if language.gtts is not None:
                assert language.gtts in supported, (
                    f"{language.code}: gTTS does not know {language.gtts!r}"
                )

    def test_hebrew_uses_gtts_legacy_code(self) -> None:
        """gTTS spells Hebrew 'iw', not 'he'."""
        assert get_language("he").gtts == "iw"

    def test_javanese_uses_gtts_legacy_code(self) -> None:
        """gTTS and Whisper both spell Javanese 'jw', not 'jv'."""
        javanese = get_language("jv")
        assert javanese.gtts == "jw"
        assert javanese.whisper == "jw"

    def test_santali_uses_ol_chiki_script(self) -> None:
        """Santali is written in Ol Chiki, not Bengali script."""
        assert get_language("sat").nllb == "sat_Olck"


class TestCapabilities:
    """Capability flags and their enforcement."""

    def test_transcribable_count_matches_whisper(self) -> None:
        """Whisper supports 100 languages; several map to more than one entry."""
        transcribable = [lang for lang in all_languages() if lang.can_transcribe]
        assert 95 <= len(transcribable) <= 115

    def test_translate_only_language_reports_no_transcription(self) -> None:
        """Bhojpuri is translatable by NLLB but inaudible to Whisper."""
        bhojpuri = get_language("bho")
        assert bhojpuri.can_translate
        assert not bhojpuri.can_transcribe

    def test_require_capability_rejects_unsupported(self) -> None:
        with pytest.raises(UnsupportedCapabilityError) as excinfo:
            require_capability("bho", Capability.TRANSCRIBE)
        assert excinfo.value.http_status == 422

    def test_require_capability_allows_supported(self) -> None:
        assert require_capability("en", Capability.TRANSCRIBE).code == "en"

    def test_whisper_code_raises_for_unsupported(self) -> None:
        with pytest.raises(UnsupportedCapabilityError):
            whisper_code("bho")

    def test_gtts_code_raises_for_unsupported(self) -> None:
        """Must be a typed 422, never an unhandled exception and a 500."""
        with pytest.raises(UnsupportedCapabilityError):
            gtts_code("bho")

    def test_nllb_code_works_for_every_language(self) -> None:
        for language in all_languages():
            assert nllb_code(language.code) == language.nllb


class TestLookups:
    """Lookup behaviour."""

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(UnknownLanguageError) as excinfo:
            require_language("not-a-language")
        assert excinfo.value.http_status == 400

    def test_get_language_returns_none_for_unknown(self) -> None:
        assert get_language("not-a-language") is None

    def test_language_codes_matches_registry(self) -> None:
        assert language_codes() == {language.code for language in all_languages()}


class TestTextDirection:
    """Right-to-left detection, derived from the script tag."""

    @pytest.mark.parametrize("code", ["ar", "he", "fa", "ur", "ps", "sd", "ug", "yi"])
    def test_rtl_languages(self, code: str) -> None:
        assert get_language(code).rtl, f"{code} should be right-to-left"

    @pytest.mark.parametrize("code", ["en", "hi", "zh", "ja", "ru", "el", "ta"])
    def test_ltr_languages(self, code: str) -> None:
        assert not get_language(code).rtl, f"{code} should be left-to-right"


class TestSerialisation:
    """The shape the API returns."""

    def test_to_dict_contains_expected_keys(self) -> None:
        payload = get_language("hi").to_dict()
        assert set(payload) == {
            "code",
            "name",
            "native_name",
            "script",
            "rtl",
            "can_translate",
            "can_transcribe",
            "can_speak",
            "has_neural_voice",
        }

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        json.dumps([language.to_dict() for language in all_languages()])


class TestMMSMapping:
    """MMS voice-code derivation."""

    def test_derives_from_nllb_prefix(self) -> None:
        assert get_language("hi").mms == "hin"
        assert get_language("sw").mms == "swh"

    def test_applies_overrides(self) -> None:
        """MMS publishes one Arabic voice under the macrolanguage code."""
        assert get_language("ar").mms == "ara"

    def test_returns_none_for_unavailable(self) -> None:
        """CJK languages are absent from the MMS TTS release."""
        assert get_language("zh").mms is None
        assert get_language("ja").mms is None
