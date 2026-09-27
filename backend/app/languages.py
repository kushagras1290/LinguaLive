"""Canonical language registry.

This module is the single source of truth for language support. A flat
code-to-NLLB dictionary is the obvious approach and it causes three classes of
defect, each prevented here:

1. **Capability mismatch.** NLLB-200 translates 202 languages, Whisper hears 100
   and gTTS speaks 68. Offering all 202 for every operation means a request to,
   say, transcribe Bhojpuri raises an unhandled exception. Each entry here
   declares its capabilities explicitly, and unsupported combinations are
   rejected with a clean 422 instead of a 500.

2. **Wrong external codes.** gTTS uses ``iw`` for Hebrew and ``jw`` for Javanese,
   not ``he``/``jv``; Santali is ``sat_Olck`` (Ol Chiki), not ``sat_Beng``.

3. **Duplicate keys.** A dict literal silently discards the first of two
   identical keys; the registry is validated for uniqueness at import.

Text direction is derived from the script suffix of the NLLB code rather than
maintained by hand, which removes an entire category of transcription error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from app.errors import UnknownLanguageError, UnsupportedCapabilityError

__all__ = [
    "Capability",
    "Language",
    "all_languages",
    "get_language",
    "require_language",
    "require_capability",
    "nllb_code",
    "whisper_code",
    "gtts_code",
    "language_codes",
]

#: Scripts written right-to-left. Used to derive :attr:`Language.rtl`.
_RTL_SCRIPTS: Final[frozenset[str]] = frozenset({"Arab", "Hebr", "Thaa", "Syrc"})

#: ISO 639-3 codes where MMS's naming diverges from the FLORES-200 prefix.
_MMS_OVERRIDES: Final[dict[str, str]] = {
    "arb": "ara",  # MMS ships one Arabic voice under the macrolanguage code
    "als": "sqi",  # Tosk Albanian is published as Albanian
    "khk": "khk",  # Halh Mongolian keeps its specific code
    "npi": "npi",
    "pes": "pes",
    "swh": "swh",
    "zsm": "zlm",  # Standard Malay is published under the macrolanguage code
    "plt": "plt",
    "gaz": "gaz",
    "uzn": "uzb",
    "azj": "azj",
    "lvs": "lav",
    "ydd": "ydd",
    "pbt": "pbt",
}

#: Languages absent from the MMS text-to-speech release. Notably the CJK
#: languages, whose non-alphabetic scripts the MMS character models do not cover.
_MMS_UNAVAILABLE: Final[frozenset[str]] = frozenset(
    {
        "zho",  # Mandarin (Simplified and Traditional)
        "yue",  # Cantonese
        "jpn",  # Japanese
        "khm",  # Khmer
        "mya",  # Burmese
        "bod",  # Tibetan
        "dzo",  # Dzongkha
        "shn",  # Shan
        "sat",  # Santali (Ol Chiki)
        "taq",  # Tamasheq
        "tzm",  # Central Atlas Tamazight
        "knc",  # Kanuri
    }
)


class Capability:
    """Capability identifiers used in error messages and the public API."""

    TRANSLATE: Final[str] = "translate"
    TRANSCRIBE: Final[str] = "transcribe"
    SPEAK: Final[str] = "speak"


@dataclass(frozen=True, slots=True)
class Language:
    """A single supported language and the external codes it maps to.

    Attributes:
        code: Application-level identifier, used in every API payload.
        name: English display name.
        native_name: Endonym, shown in the language picker.
        nllb: FLORES-200 code for NLLB translation. Always present.
        whisper: ISO-639-1 code Whisper accepts, or ``None`` if unsupported.
        gtts: Code gTTS accepts, or ``None`` if unsupported.
    """

    code: str
    name: str
    native_name: str
    nllb: str
    whisper: str | None
    gtts: str | None

    @property
    def script(self) -> str:
        """The ISO 15924 script tag from the NLLB code (e.g. ``"Deva"``)."""
        _, _, script = self.nllb.partition("_")
        return script

    @property
    def rtl(self) -> bool:
        """True when the language is written right-to-left."""
        return self.script in _RTL_SCRIPTS

    @property
    def can_translate(self) -> bool:
        """True when NLLB can translate to and from this language."""
        return True  # Membership in this registry implies NLLB support.

    @property
    def can_transcribe(self) -> bool:
        """True when Whisper can transcribe speech in this language."""
        return self.whisper is not None

    @property
    def mms(self) -> str | None:
        """Candidate ``facebook/mms-tts-*`` model suffix for this language.

        MMS is keyed by ISO 639-3, which is exactly the prefix of the FLORES-200
        code, so the mapping is derived rather than maintained by hand. A small
        override table handles the cases where MMS diverges, and
        :data:`_MMS_UNAVAILABLE` lists languages MMS's TTS release omits.

        Returns ``None`` when no MMS voice is expected to exist. Because the full
        1107-language inventory cannot be verified offline, this is a *candidate*
        only; the TTS chain falls back automatically when a download fails.
        """
        iso3, _, _ = self.nllb.partition("_")
        if iso3 in _MMS_UNAVAILABLE:
            return None
        return _MMS_OVERRIDES.get(iso3, iso3)

    @property
    def can_speak(self) -> bool:
        """True when at least one TTS engine can synthesise this language.

        The formant synthesiser in :mod:`app.tts.synth` is language-agnostic and
        always available, so every language is technically speakable. This flag
        reports whether a *neural* voice exists, which is what the UI should key
        its speaker button on.
        """
        return self.gtts is not None or self.mms is not None

    def supports(self, capability: str) -> bool:
        """Return whether this language supports ``capability``."""
        match capability:
            case Capability.TRANSLATE:
                return self.can_translate
            case Capability.TRANSCRIBE:
                return self.can_transcribe
            case Capability.SPEAK:
                return self.can_speak
            case _:
                return False

    def to_dict(self) -> dict[str, str | bool]:
        """Serialise for the ``/api/languages`` response."""
        return {
            "code": self.code,
            "name": self.name,
            "native_name": self.native_name,
            "script": self.script,
            "rtl": self.rtl,
            "can_translate": self.can_translate,
            "can_transcribe": self.can_transcribe,
            "can_speak": self.can_speak,
            # Our own formant synthesiser is language-agnostic, so speech is
            # always possible even without a neural voice — just more robotic.
            "has_neural_voice": self.gtts is not None or self.mms is not None,
        }


# --------------------------------------------------------------------------- #
# The registry.
#
# Columns: code, English name, native name, NLLB code, Whisper code, gTTS code.
# ``None`` means the corresponding model does not support that language.
# --------------------------------------------------------------------------- #
_RAW: Final[tuple[tuple[str, str, str, str, str | None, str | None], ...]] = (
    # --- Widely spoken --------------------------------------------------- #
    ("en", "English", "English", "eng_Latn", "en", "en"),
    ("es", "Spanish", "Español", "spa_Latn", "es", "es"),
    ("fr", "French", "Français", "fra_Latn", "fr", "fr"),
    ("de", "German", "Deutsch", "deu_Latn", "de", "de"),
    ("it", "Italian", "Italiano", "ita_Latn", "it", "it"),
    ("pt", "Portuguese", "Português", "por_Latn", "pt", "pt"),
    ("ru", "Russian", "Русский", "rus_Cyrl", "ru", "ru"),
    ("zh", "Chinese (Simplified)", "简体中文", "zho_Hans", "zh", "zh-CN"),
    ("zh-Hant", "Chinese (Traditional)", "繁體中文", "zho_Hant", "zh", "zh-TW"),
    ("yue", "Cantonese", "粵語", "yue_Hant", "yue", "yue"),
    ("ja", "Japanese", "日本語", "jpn_Jpan", "ja", "ja"),
    ("ko", "Korean", "한국어", "kor_Hang", "ko", "ko"),
    ("ar", "Arabic (Modern Standard)", "العربية", "arb_Arab", "ar", "ar"),
    ("tr", "Turkish", "Türkçe", "tur_Latn", "tr", "tr"),
    ("fa", "Persian", "فارسی", "pes_Arab", "fa", None),
    ("he", "Hebrew", "עברית", "heb_Hebr", "he", "iw"),
    ("pl", "Polish", "Polski", "pol_Latn", "pl", "pl"),
    ("nl", "Dutch", "Nederlands", "nld_Latn", "nl", "nl"),
    ("vi", "Vietnamese", "Tiếng Việt", "vie_Latn", "vi", "vi"),
    ("th", "Thai", "ไทย", "tha_Thai", "th", "th"),
    ("id", "Indonesian", "Bahasa Indonesia", "ind_Latn", "id", "id"),
    ("ms", "Malay", "Bahasa Melayu", "zsm_Latn", "ms", "ms"),
    ("uk", "Ukrainian", "Українська", "ukr_Cyrl", "uk", "uk"),
    ("sw", "Swahili", "Kiswahili", "swh_Latn", "sw", "sw"),
    # --- South Asian ----------------------------------------------------- #
    ("hi", "Hindi", "हिन्दी", "hin_Deva", "hi", "hi"),
    ("bn", "Bengali", "বাংলা", "ben_Beng", "bn", "bn"),
    ("ur", "Urdu", "اردو", "urd_Arab", "ur", "ur"),
    ("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru", "pa", "pa"),
    ("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", "gu", "gu"),
    ("mr", "Marathi", "मराठी", "mar_Deva", "mr", "mr"),
    ("ta", "Tamil", "தமிழ்", "tam_Taml", "ta", "ta"),
    ("te", "Telugu", "తెలుగు", "tel_Telu", "te", "te"),
    ("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", "kn", "kn"),
    ("ml", "Malayalam", "മലയാളം", "mal_Mlym", "ml", "ml"),
    ("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya", None, None),
    ("as", "Assamese", "অসমীয়া", "asm_Beng", "as", None),
    ("ne", "Nepali", "नेपाली", "npi_Deva", "ne", "ne"),
    ("si", "Sinhala", "සිංහල", "sin_Sinh", "si", "si"),
    ("sa", "Sanskrit", "संस्कृतम्", "san_Deva", "sa", None),
    ("mai", "Maithili", "मैथिली", "mai_Deva", None, None),
    ("bho", "Bhojpuri", "भोजपुरी", "bho_Deva", None, None),
    ("awa", "Awadhi", "अवधी", "awa_Deva", None, None),
    ("mag", "Magahi", "मगही", "mag_Deva", None, None),
    ("hne", "Chhattisgarhi", "छत्तीसगढ़ी", "hne_Deva", None, None),
    ("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "sat_Olck", None, None),
    ("mni", "Meitei (Manipuri)", "ꯃꯤꯇꯩꯂꯣꯟ", "mni_Beng", None, None),
    ("lus", "Mizo", "Mizo ṭawng", "lus_Latn", None, None),
    ("ks", "Kashmiri (Arabic)", "کٲشُر", "kas_Arab", None, None),
    ("ks-Deva", "Kashmiri (Devanagari)", "कॉशुर", "kas_Deva", None, None),
    ("sd", "Sindhi", "سنڌي", "snd_Arab", "sd", None),
    # --- East & Southeast Asian ------------------------------------------ #
    ("my", "Burmese", "မြန်မာ", "mya_Mymr", "my", "my"),
    ("km", "Khmer", "ខ្មែរ", "khm_Khmr", "km", "km"),
    ("lo", "Lao", "ລາວ", "lao_Laoo", "lo", None),
    ("tl", "Filipino", "Filipino", "tgl_Latn", "tl", "tl"),
    ("ceb", "Cebuano", "Cebuano", "ceb_Latn", None, None),
    ("ilo", "Ilocano", "Ilokano", "ilo_Latn", None, None),
    ("pag", "Pangasinan", "Pangasinan", "pag_Latn", None, None),
    ("war", "Waray", "Waray", "war_Latn", None, None),
    ("jv", "Javanese", "Basa Jawa", "jav_Latn", "jw", "jw"),
    ("su", "Sundanese", "Basa Sunda", "sun_Latn", "su", "su"),
    ("min", "Minangkabau", "Minangkabau", "min_Latn", None, None),
    ("ban", "Balinese", "Basa Bali", "ban_Latn", None, None),
    ("bug", "Buginese", "Basa Ugi", "bug_Latn", None, None),
    ("ace", "Acehnese (Latin)", "Bahsa Acèh", "ace_Latn", None, None),
    ("ace-Arab", "Acehnese (Arabic)", "بهسا اچيه", "ace_Arab", None, None),
    ("bjn", "Banjar (Latin)", "Bahasa Banjar", "bjn_Latn", None, None),
    ("bjn-Arab", "Banjar (Arabic)", "بهاس بنجر", "bjn_Arab", None, None),
    ("shn", "Shan", "တႆး", "shn_Mymr", None, None),
    ("kac", "Jingpho", "Jinghpaw", "kac_Latn", None, None),
    ("bo", "Tibetan", "བོད་སྐད", "bod_Tibt", "bo", None),
    ("dz", "Dzongkha", "རྫོང་ཁ", "dzo_Tibt", None, None),
    ("mn", "Mongolian", "Монгол", "khk_Cyrl", "mn", None),
    # --- European -------------------------------------------------------- #
    ("sv", "Swedish", "Svenska", "swe_Latn", "sv", "sv"),
    ("da", "Danish", "Dansk", "dan_Latn", "da", "da"),
    ("no", "Norwegian Bokmål", "Norsk bokmål", "nob_Latn", "no", "no"),
    ("nn", "Norwegian Nynorsk", "Nynorsk", "nno_Latn", "nn", None),
    ("fi", "Finnish", "Suomi", "fin_Latn", "fi", "fi"),
    ("et", "Estonian", "Eesti", "est_Latn", "et", "et"),
    ("lv", "Latvian", "Latviešu", "lvs_Latn", "lv", "lv"),
    ("lt", "Lithuanian", "Lietuvių", "lit_Latn", "lt", "lt"),
    ("ltg", "Latgalian", "Latgaļu", "ltg_Latn", None, None),
    ("cs", "Czech", "Čeština", "ces_Latn", "cs", "cs"),
    ("sk", "Slovak", "Slovenčina", "slk_Latn", "sk", "sk"),
    ("sl", "Slovenian", "Slovenščina", "slv_Latn", "sl", None),
    ("hr", "Croatian", "Hrvatski", "hrv_Latn", "hr", "hr"),
    ("bs", "Bosnian", "Bosanski", "bos_Latn", "bs", "bs"),
    ("sr", "Serbian", "Српски", "srp_Cyrl", "sr", "sr"),
    ("mk", "Macedonian", "Македонски", "mkd_Cyrl", "mk", None),
    ("bg", "Bulgarian", "Български", "bul_Cyrl", "bg", "bg"),
    ("ro", "Romanian", "Română", "ron_Latn", "ro", "ro"),
    ("hu", "Hungarian", "Magyar", "hun_Latn", "hu", "hu"),
    ("el", "Greek", "Ελληνικά", "ell_Grek", "el", "el"),
    ("be", "Belarusian", "Беларуская", "bel_Cyrl", "be", None),
    ("ca", "Catalan", "Català", "cat_Latn", "ca", "ca"),
    ("gl", "Galician", "Galego", "glg_Latn", "gl", "gl"),
    ("eu", "Basque", "Euskara", "eus_Latn", "eu", "eu"),
    ("ast", "Asturian", "Asturianu", "ast_Latn", None, None),
    ("oc", "Occitan", "Occitan", "oci_Latn", "oc", None),
    ("is", "Icelandic", "Íslenska", "isl_Latn", "is", "is"),
    ("fo", "Faroese", "Føroyskt", "fao_Latn", "fo", None),
    ("ga", "Irish", "Gaeilge", "gle_Latn", None, None),
    ("gd", "Scottish Gaelic", "Gàidhlig", "gla_Latn", None, None),
    ("cy", "Welsh", "Cymraeg", "cym_Latn", "cy", "cy"),
    ("mt", "Maltese", "Malti", "mlt_Latn", "mt", None),
    ("sq", "Albanian", "Shqip", "als_Latn", "sq", "sq"),
    ("lb", "Luxembourgish", "Lëtzebuergesch", "ltz_Latn", "lb", None),
    ("li", "Limburgish", "Limburgs", "lim_Latn", None, None),
    ("fur", "Friulian", "Furlan", "fur_Latn", None, None),
    ("lij", "Ligurian", "Ligure", "lij_Latn", None, None),
    ("lmo", "Lombard", "Lombard", "lmo_Latn", None, None),
    ("vec", "Venetian", "Vèneto", "vec_Latn", None, None),
    ("scn", "Sicilian", "Sicilianu", "scn_Latn", None, None),
    ("srd", "Sardinian", "Sardu", "srd_Latn", None, None),
    ("szl", "Silesian", "Ślōnski", "szl_Latn", None, None),
    ("crh", "Crimean Tatar", "Qırımtatarca", "crh_Latn", None, None),
    ("eo", "Esperanto", "Esperanto", "epo_Latn", None, None),
    # --- Caucasus, Central & West Asia ----------------------------------- #
    ("hy", "Armenian", "Հայերեն", "hye_Armn", "hy", None),
    ("ka", "Georgian", "ქართული", "kat_Geor", "ka", None),
    ("az", "Azerbaijani (North)", "Azərbaycan", "azj_Latn", "az", None),
    ("azb", "Azerbaijani (South)", "آذربایجان", "azb_Arab", None, None),
    ("kk", "Kazakh", "Қазақ", "kaz_Cyrl", "kk", None),
    ("ky", "Kyrgyz", "Кыргызча", "kir_Cyrl", None, None),
    ("uz", "Uzbek", "Oʻzbek", "uzn_Latn", "uz", None),
    ("tk", "Turkmen", "Türkmen", "tuk_Latn", "tk", None),
    ("tt", "Tatar", "Татар", "tat_Cyrl", "tt", None),
    ("ba", "Bashkir", "Башҡорт", "bak_Cyrl", "ba", None),
    ("tg", "Tajik", "Тоҷикӣ", "tgk_Cyrl", "tg", None),
    ("ug", "Uyghur", "ئۇيغۇرچە", "uig_Arab", None, None),
    ("ps", "Pashto", "پښتو", "pbt_Arab", "ps", None),
    ("prs", "Dari", "دری", "prs_Arab", None, None),
    ("ckb", "Kurdish (Sorani)", "کوردیی ناوەندی", "ckb_Arab", None, None),
    ("kmr", "Kurdish (Kurmanji)", "Kurmancî", "kmr_Latn", None, None),
    ("yi", "Yiddish", "ייִדיש", "ydd_Hebr", "yi", None),
    # --- Arabic varieties ------------------------------------------------ #
    ("arz", "Arabic (Egyptian)", "مصرى", "arz_Arab", "ar", "ar"),
    ("ary", "Arabic (Moroccan)", "الدارجة", "ary_Arab", "ar", "ar"),
    ("apc", "Arabic (North Levantine)", "شامي", "apc_Arab", "ar", "ar"),
    ("ajp", "Arabic (South Levantine)", "عربي", "ajp_Arab", "ar", "ar"),
    ("acm", "Arabic (Mesopotamian)", "عراقي", "acm_Arab", "ar", "ar"),
    ("acq", "Arabic (Ta'izzi-Adeni)", "تعزي", "acq_Arab", "ar", "ar"),
    ("aeb", "Arabic (Tunisian)", "تونسي", "aeb_Arab", "ar", "ar"),
    ("ars", "Arabic (Najdi)", "نجدي", "ars_Arab", "ar", "ar"),
    # --- African ---------------------------------------------------------- #
    ("am", "Amharic", "አማርኛ", "amh_Ethi", "am", "am"),
    ("ti", "Tigrinya", "ትግርኛ", "tir_Ethi", None, None),
    ("ha", "Hausa", "Hausa", "hau_Latn", "ha", "ha"),
    ("yo", "Yoruba", "Yorùbá", "yor_Latn", "yo", None),
    ("ig", "Igbo", "Igbo", "ibo_Latn", None, None),
    ("zu", "Zulu", "isiZulu", "zul_Latn", None, None),
    ("xh", "Xhosa", "isiXhosa", "xho_Latn", None, None),
    ("af", "Afrikaans", "Afrikaans", "afr_Latn", "af", "af"),
    ("st", "Sesotho", "Sesotho", "sot_Latn", None, None),
    ("nso", "Northern Sotho", "Sepedi", "nso_Latn", None, None),
    ("tn", "Setswana", "Setswana", "tsn_Latn", None, None),
    ("ts", "Xitsonga", "Xitsonga", "tso_Latn", None, None),
    ("ss", "Swati", "siSwati", "ssw_Latn", None, None),
    ("sn", "Shona", "chiShona", "sna_Latn", "sn", None),
    ("ny", "Chichewa", "Chichewa", "nya_Latn", None, None),
    ("so", "Somali", "Soomaali", "som_Latn", "so", None),
    ("gaz", "Oromo", "Afaan Oromoo", "gaz_Latn", None, None),
    ("rw", "Kinyarwanda", "Kinyarwanda", "kin_Latn", None, None),
    ("rn", "Rundi", "Ikirundi", "run_Latn", None, None),
    ("lg", "Ganda", "Luganda", "lug_Latn", None, None),
    ("luo", "Luo", "Dholuo", "luo_Latn", None, None),
    ("kam", "Kamba", "Kikamba", "kam_Latn", None, None),
    ("ki", "Kikuyu", "Gĩkũyũ", "kik_Latn", None, None),
    ("bem", "Bemba", "Ichibemba", "bem_Latn", None, None),
    ("tum", "Tumbuka", "Chitumbuka", "tum_Latn", None, None),
    ("ln", "Lingala", "Lingála", "lin_Latn", "ln", None),
    ("kg", "Kikongo", "Kikongo", "kon_Latn", None, None),
    ("lua", "Luba-Kasai", "Ciluba", "lua_Latn", None, None),
    ("cjk", "Chokwe", "Chokwe", "cjk_Latn", None, None),
    ("kmb", "Kimbundu", "Kimbundu", "kmb_Latn", None, None),
    ("umb", "Umbundu", "Umbundu", "umb_Latn", None, None),
    ("wo", "Wolof", "Wolof", "wol_Latn", None, None),
    ("ff", "Fulfulde", "Fulfulde", "fuv_Latn", None, None),
    ("bm", "Bambara", "Bamanankan", "bam_Latn", None, None),
    ("dyu", "Dyula", "Julakan", "dyu_Latn", None, None),
    ("ak", "Akan", "Akan", "aka_Latn", None, None),
    ("tw", "Twi", "Twi", "twi_Latn", None, None),
    ("ee", "Ewe", "Eʋegbe", "ewe_Latn", None, None),
    ("fon", "Fon", "Fon", "fon_Latn", None, None),
    ("kbp", "Kabiyè", "Kabɩyɛ", "kbp_Latn", None, None),
    ("mos", "Mossi", "Mòoré", "mos_Latn", None, None),
    ("sg", "Sango", "Sängö", "sag_Latn", None, None),
    ("dik", "Dinka", "Thuɔŋjäŋ", "dik_Latn", None, None),
    ("nus", "Nuer", "Thok Naath", "nus_Latn", None, None),
    ("knc", "Kanuri (Latin)", "Kanuri", "knc_Latn", None, None),
    ("knc-Arab", "Kanuri (Arabic)", "كنوري", "knc_Arab", None, None),
    ("kab", "Kabyle", "Taqbaylit", "kab_Latn", None, None),
    ("tzm", "Tamazight (Central Atlas)", "ⵜⴰⵎⴰⵣⵉⵖⵜ", "tzm_Tfng", None, None),
    ("taq", "Tamasheq (Latin)", "Tamasheq", "taq_Latn", None, None),
    ("taq-Tfng", "Tamasheq (Tifinagh)", "ⵜⴰⵎⴰⵌⴰⵆ", "taq_Tfng", None, None),
    ("kea", "Kabuverdianu", "Kabuverdianu", "kea_Latn", None, None),
    ("mg", "Malagasy", "Malagasy", "plt_Latn", "mg", None),
    # --- Americas & Pacific ----------------------------------------------- #
    ("ht", "Haitian Creole", "Kreyòl ayisyen", "hat_Latn", "ht", None),
    ("pap", "Papiamento", "Papiamentu", "pap_Latn", None, None),
    ("gn", "Guarani", "Avañe'ẽ", "grn_Latn", None, None),
    ("quy", "Quechua (Ayacucho)", "Runa Simi", "quy_Latn", None, None),
    ("ayr", "Aymara", "Aymar aru", "ayr_Latn", None, None),
    ("mi", "Māori", "Te Reo Māori", "mri_Latn", "mi", None),
    ("sm", "Samoan", "Gagana Samoa", "smo_Latn", None, None),
    ("fj", "Fijian", "Na Vosa Vakaviti", "fij_Latn", None, None),
    ("tpi", "Tok Pisin", "Tok Pisin", "tpi_Latn", None, None),
)


def _build_registry() -> dict[str, Language]:
    """Materialise the registry, rejecting duplicate codes at import time.

    Raises:
        RuntimeError: If a language code appears twice. This is a programming
            error in this module, not a runtime condition, so it is deliberately
            fatal rather than a handled application error.
    """
    registry: dict[str, Language] = {}
    for code, name, native_name, nllb, whisper, gtts in _RAW:
        if code in registry:
            raise RuntimeError(
                f"Duplicate language code {code!r} in the registry. "
                "Each code must appear exactly once."
            )
        registry[code] = Language(
            code=code,
            name=name,
            native_name=native_name,
            nllb=nllb,
            whisper=whisper,
            gtts=gtts,
        )
    return registry


_REGISTRY: Final[dict[str, Language]] = _build_registry()


@lru_cache(maxsize=1)
def all_languages() -> tuple[Language, ...]:
    """Every supported language, sorted by English name."""
    return tuple(sorted(_REGISTRY.values(), key=lambda lang: lang.name))


@lru_cache(maxsize=1)
def language_codes() -> frozenset[str]:
    """The set of valid language codes."""
    return frozenset(_REGISTRY)


def get_language(code: str) -> Language | None:
    """Look up a language, returning ``None`` when it is unknown."""
    return _REGISTRY.get(code)


def require_language(code: str) -> Language:
    """Look up a language or raise.

    Args:
        code: An application-level language code.

    Returns:
        The matching :class:`Language`.

    Raises:
        UnknownLanguageError: If ``code`` is not in the registry.
    """
    language = _REGISTRY.get(code)
    if language is None:
        raise UnknownLanguageError(
            f"Unknown language code {code!r}.",
            details={"code": code, "supported_count": len(_REGISTRY)},
        )
    return language


def require_capability(code: str, capability: str) -> Language:
    """Look up a language and assert it supports ``capability``.

    Args:
        code: An application-level language code.
        capability: One of the :class:`Capability` constants.

    Returns:
        The matching :class:`Language`.

    Raises:
        UnknownLanguageError: If ``code`` is not in the registry.
        UnsupportedCapabilityError: If the language cannot perform ``capability``.
    """
    language = require_language(code)
    if not language.supports(capability):
        reason = {
            Capability.TRANSCRIBE: "Whisper cannot transcribe speech in this language",
            Capability.SPEAK: "gTTS cannot synthesise speech in this language",
            Capability.TRANSLATE: "NLLB cannot translate this language",
        }.get(capability, "This operation is unsupported")
        raise UnsupportedCapabilityError(
            f"{language.name} does not support '{capability}': {reason}.",
            details={"code": code, "capability": capability, "language": language.name},
        )
    return language


def nllb_code(code: str) -> str:
    """Return the FLORES-200 code used by NLLB for ``code``."""
    return require_capability(code, Capability.TRANSLATE).nllb


def whisper_code(code: str) -> str:
    """Return the Whisper language code for ``code``.

    Raises:
        UnsupportedCapabilityError: If Whisper does not support the language.
    """
    language = require_capability(code, Capability.TRANSCRIBE)
    assert language.whisper is not None  # guaranteed by require_capability
    return language.whisper


def gtts_code(code: str) -> str:
    """Return the gTTS language code for ``code``.

    Checks gTTS support specifically rather than the general ``speak``
    capability: a language can be speakable via MMS or the built-in formant
    synthesiser while gTTS still has no voice for it, and this function must
    reject exactly that case so the TTS chain can fall through.

    Raises:
        UnknownLanguageError: If ``code`` is not in the registry.
        UnsupportedCapabilityError: If gTTS has no voice for the language.
    """
    language = require_language(code)
    if language.gtts is None:
        raise UnsupportedCapabilityError(
            f"gTTS has no voice for {language.name}.",
            details={"code": code, "capability": Capability.SPEAK, "backend": "gtts"},
        )
    return language.gtts
