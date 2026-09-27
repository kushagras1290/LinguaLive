"""Script transliteration for the fallback synthesiser.

The formant synthesiser in :mod:`app.tts.synth` reads Latin letters. Most of the
202 supported languages are not written in Latin script, so without this the
"always works" fallback silently produced no audio for them — exactly the
failure the TTS chain exists to prevent.

This maps characters from the major writing systems onto a rough Latin
approximation, which the existing grapheme-to-phoneme rules then pronounce. It
is a *phonetic approximation*, not a scholarly romanisation: the goal is
intelligible sound from a zero-dependency fallback, not transliteration anyone
would cite.

Ideographic scripts (Han, and the Yi and Nushu blocks) are deliberately not
mapped. A Han character carries no pronunciation without a per-language reading
dictionary of tens of thousands of entries, so those return empty and the caller
reports honestly that no voice is available.
"""

from __future__ import annotations

import unicodedata
from typing import Final

from app.logging_conf import get_logger

__all__ = ["transliterate", "is_transliterable", "script_of"]

_LOG = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Devanagari (Hindi, Marathi, Nepali, Awadhi, Bhojpuri, Sanskrit, ...)
#
# An abugida: consonants carry an inherent 'a' unless a vowel sign or the virama
# (U+094D) suppresses it. The inherent vowel is appended here and stripped by
# the virama rule below.
# --------------------------------------------------------------------------- #
_DEVANAGARI: Final[dict[str, str]] = {
    # Independent vowels
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    # Consonants (inherent 'a' included)
    "क": "ka", "ख": "kha", "ग": "ga", "घ": "gha", "ङ": "nga",
    "च": "cha", "छ": "chha", "ज": "ja", "झ": "jha", "ञ": "nya",
    "ट": "ta", "ठ": "tha", "ड": "da", "ढ": "dha", "ण": "na",
    "त": "ta", "थ": "tha", "द": "da", "ध": "dha", "न": "na",
    "प": "pa", "फ": "pha", "ब": "ba", "भ": "bha", "म": "ma",
    "य": "ya", "र": "ra", "ल": "la", "व": "va",
    "श": "sha", "ष": "sha", "स": "sa", "ह": "ha",
    "ळ": "la", "क़": "qa", "ख़": "kha", "ग़": "ga", "ज़": "za",
    "ड़": "ra", "ढ़": "rha", "फ़": "fa",
    # Dependent vowel signs (replace the inherent 'a')
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    # Modifiers
    "ं": "n", "ः": "h", "ँ": "n",
    "्": "",   # virama: suppresses the inherent vowel
    "़": "",   # nukta: already folded into the consonants above
    "।": ".", "॥": ".",
}

# --------------------------------------------------------------------------- #
# Cyrillic (Russian, Ukrainian, Serbian, Bulgarian, Kazakh, ...)
# --------------------------------------------------------------------------- #
_CYRILLIC: Final[dict[str, str]] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "ye", "ё": "yo",
    "ж": "zh", "з": "z", "и": "ee", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "oo",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    # Non-Russian Cyrillic
    "і": "ee", "ї": "yee", "є": "ye", "ґ": "g", "ђ": "j", "ј": "y",
    "љ": "ly", "њ": "ny", "ћ": "ch", "џ": "j", "ѓ": "g", "ќ": "k",
    "ә": "a", "ғ": "g", "қ": "k", "ң": "ng", "ө": "o", "ұ": "u",
    "ү": "u", "һ": "h", "і": "i",
}

# --------------------------------------------------------------------------- #
# Greek
# --------------------------------------------------------------------------- #
_GREEK: Final[dict[str, str]] = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "ee",
    "θ": "th", "ι": "ee", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "ks",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "ee",
    "φ": "f", "χ": "kh", "ψ": "ps", "ω": "o",
    "ά": "a", "έ": "e", "ή": "ee", "ί": "ee", "ό": "o", "ύ": "ee", "ώ": "o",
    "ϊ": "ee", "ϋ": "ee", "ΐ": "ee", "ΰ": "ee",
}

# --------------------------------------------------------------------------- #
# Arabic (Arabic, Persian, Urdu, Pashto, Sindhi, Uyghur, Kashmiri, ...)
#
# Short vowels are usually unwritten. A neutral 'a' is inserted between
# consonants so the output is pronounceable rather than an unsayable cluster.
# --------------------------------------------------------------------------- #
_ARABIC: Final[dict[str, str]] = {
    "ا": "aa", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s",
    "ض": "d", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "oo", "ي": "ee",
    "ى": "aa", "ة": "a", "ء": "a", "أ": "a", "إ": "i", "آ": "aa", "ؤ": "u",
    "ئ": "i", "ٱ": "a",
    # Short vowel diacritics, when present
    "َ": "a", "ِ": "i", "ُ": "u", "ْ": "", "ّ": "", "ً": "an", "ٍ": "in",
    "ٌ": "un",
    # Persian, Urdu and Pashto extensions
    "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g", "ک": "k", "ی": "ee",
    "ٹ": "t", "ڈ": "d", "ڑ": "r", "ں": "n", "ھ": "h", "ے": "e",
    "ٺ": "th", "ٿ": "th", "ڀ": "bh", "ٻ": "b", "ڄ": "j", "ڦ": "ph",
    "ڻ": "n", "ڊ": "d", "ڏ": "d", "ڍ": "dh", "ڳ": "g", "ڱ": "ng",
    "ښ": "sh", "ږ": "zh", "ټ": "t", "ډ": "d", "ړ": "r", "ڼ": "n",
    "۔": ".", "،": ",", "؟": "?",
}

# --------------------------------------------------------------------------- #
# Hebrew (Hebrew, Yiddish)
# --------------------------------------------------------------------------- #
_HEBREW: Final[dict[str, str]] = {
    "א": "a", "ב": "v", "ג": "g", "ד": "d", "ה": "h", "ו": "o", "ז": "z",
    "ח": "kh", "ט": "t", "י": "ee", "כ": "k", "ך": "kh", "ל": "l", "מ": "m",
    "ם": "m", "נ": "n", "ן": "n", "ס": "s", "ע": "a", "פ": "p", "ף": "f",
    "צ": "ts", "ץ": "ts", "ק": "k", "ר": "r", "ש": "sh", "ת": "t",
}

# --------------------------------------------------------------------------- #
# Japanese kana
#
# Fully regular and therefore worth mapping, unlike Han ideographs. Mixed
# Japanese text still loses its kanji, which is why Japanese prefers gTTS.
# --------------------------------------------------------------------------- #
_KANA: Final[dict[str, str]] = {
    "あ": "a", "い": "ee", "う": "oo", "え": "e", "お": "o",
    "か": "ka", "き": "kee", "く": "koo", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gee", "ぐ": "goo", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shee", "す": "soo", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "jee", "ず": "zoo", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chee", "つ": "tsoo", "て": "te", "と": "to",
    "だ": "da", "ぢ": "jee", "づ": "zoo", "で": "de", "ど": "do",
    "な": "na", "に": "nee", "ぬ": "noo", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hee", "ふ": "foo", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bee", "ぶ": "boo", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pee", "ぷ": "poo", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mee", "む": "moo", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yoo", "よ": "yo",
    "ら": "ra", "り": "ree", "る": "roo", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n", "っ": "", "ー": "",
}

# --------------------------------------------------------------------------- #
# Thai
# --------------------------------------------------------------------------- #
_THAI: Final[dict[str, str]] = {
    "ก": "k", "ข": "kh", "ค": "kh", "ง": "ng", "จ": "j", "ฉ": "ch",
    "ช": "ch", "ซ": "s", "ญ": "y", "ด": "d", "ต": "t", "ถ": "th",
    "ท": "th", "ธ": "th", "น": "n", "บ": "b", "ป": "p", "ผ": "ph",
    "ฝ": "f", "พ": "ph", "ฟ": "f", "ภ": "ph", "ม": "m", "ย": "y",
    "ร": "r", "ล": "l", "ว": "w", "ศ": "s", "ษ": "s", "ส": "s",
    "ห": "h", "อ": "o", "ฮ": "h",
    "ะ": "a", "า": "aa", "ิ": "i", "ี": "ee", "ึ": "u", "ื": "u",
    "ุ": "u", "ู": "oo", "เ": "e", "แ": "ae", "โ": "o", "ใ": "ai",
    "ไ": "ai", "ำ": "am", "ๅ": "aa",
    "่": "", "้": "", "๊": "", "๋": "", "็": "", "์": "",
}

# --------------------------------------------------------------------------- #
# Tibetan (Tibetan, Dzongkha)
# --------------------------------------------------------------------------- #
_TIBETAN: Final[dict[str, str]] = {
    "ཀ": "ka", "ཁ": "kha", "ག": "ga", "ང": "nga",
    "ཅ": "cha", "ཆ": "chha", "ཇ": "ja", "ཉ": "nya",
    "ཏ": "ta", "ཐ": "tha", "ད": "da", "ན": "na",
    "པ": "pa", "ཕ": "pha", "བ": "ba", "མ": "ma",
    "ཙ": "tsa", "ཚ": "tsha", "ཛ": "dza", "ཝ": "wa",
    "ཞ": "zha", "ཟ": "za", "འ": "a", "ཡ": "ya",
    "ར": "ra", "ལ": "la", "ཤ": "sha", "ས": "sa",
    "ཧ": "ha", "ཨ": "a",
    "ི": "i", "ུ": "u", "ེ": "e", "ོ": "o",
    "་": " ", "།": ".",
}

# --------------------------------------------------------------------------- #
# Myanmar (Burmese, Shan)
# --------------------------------------------------------------------------- #
_MYANMAR: Final[dict[str, str]] = {
    "က": "ka", "ခ": "kha", "ဂ": "ga", "ဃ": "gha", "င": "nga",
    "စ": "sa", "ဆ": "hsa", "ဇ": "za", "ဈ": "zha", "ည": "nya",
    "ဋ": "ta", "ဌ": "hta", "ဍ": "da", "ဎ": "dha", "ဏ": "na",
    "တ": "ta", "ထ": "hta", "ဒ": "da", "ဓ": "dha", "န": "na",
    "ပ": "pa", "ဖ": "pha", "ဗ": "ba", "ဘ": "bha", "မ": "ma",
    "ယ": "ya", "ရ": "ra", "လ": "la", "ဝ": "wa", "သ": "tha",
    "ဟ": "ha", "ဠ": "la", "အ": "a",
    "ာ": "aa", "ိ": "i", "ီ": "ee", "ု": "u", "ူ": "oo",
    "ေ": "e", "ဲ": "ai", "ံ": "n", "့": "", "း": "", "်": "",
    "္": "", "ျ": "y", "ြ": "r", "ွ": "w", "ှ": "h",
    "။": ".", "၊": ",",
}

# --------------------------------------------------------------------------- #
# Ol Chiki (Santali) and Tifinagh (Tamazight, Tamasheq)
# --------------------------------------------------------------------------- #
_OL_CHIKI: Final[dict[str, str]] = {
    "ᱚ": "o", "ᱛ": "t", "ᱜ": "g", "ᱝ": "ng", "ᱞ": "l", "ᱟ": "a",
    "ᱠ": "k", "ᱡ": "j", "ᱢ": "m", "ᱣ": "w", "ᱤ": "i", "ᱥ": "s",
    "ᱦ": "h", "ᱧ": "ny", "ᱨ": "r", "ᱩ": "u", "ᱪ": "ch", "ᱫ": "d",
    "ᱬ": "n", "ᱭ": "y", "ᱮ": "e", "ᱯ": "p", "ᱰ": "d", "ᱱ": "n",
    "ᱲ": "r", "ᱳ": "o", "ᱴ": "t", "ᱵ": "b", "ᱶ": "v", "ᱷ": "h",
}

_TIFINAGH: Final[dict[str, str]] = {
    "ⴰ": "a", "ⴱ": "b", "ⴳ": "g", "ⴷ": "d", "ⴹ": "d", "ⴻ": "e",
    "ⴼ": "f", "ⴽ": "k", "ⵀ": "h", "ⵃ": "h", "ⵄ": "a", "ⵅ": "kh",
    "ⵇ": "q", "ⵉ": "i", "ⵊ": "j", "ⵍ": "l", "ⵎ": "m", "ⵏ": "n",
    "ⵓ": "u", "ⵔ": "r", "ⵕ": "r", "ⵖ": "gh", "ⵙ": "s", "ⵚ": "s",
    "ⵛ": "sh", "ⵜ": "t", "ⵟ": "t", "ⵡ": "w", "ⵢ": "y", "ⵣ": "z",
    "ⵥ": "z", "ⵌ": "zh",
}

#: Every table, merged. Later tables do not override earlier ones because the
#: scripts occupy disjoint Unicode blocks.
_COMBINED: Final[dict[str, str]] = {
    **_DEVANAGARI,
    **_CYRILLIC,
    **_GREEK,
    **_ARABIC,
    **_HEBREW,
    **_KANA,
    **_THAI,
    **_TIBETAN,
    **_MYANMAR,
    **_OL_CHIKI,
    **_TIFINAGH,
}

#: Unicode blocks with no usable character-to-sound mapping.
_IDEOGRAPHIC_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F), # CJK Extensions B-F
    (0xA000, 0xA4CF),   # Yi syllables
    (0x1B170, 0x1B2FF), # Nushu
)


def _is_ideographic(char: str) -> bool:
    """Whether ``char`` is an ideograph with no character-level reading."""
    codepoint = ord(char)
    return any(low <= codepoint <= high for low, high in _IDEOGRAPHIC_RANGES)


def script_of(text: str) -> str:
    """Return a coarse script name for ``text``, for logging and diagnostics."""
    for char in text:
        if char.isspace() or not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        return name.split()[0].title()
    return "Unknown"


def is_transliterable(text: str) -> bool:
    """Whether enough of ``text`` can be mapped to produce usable speech.

    Args:
        text: Input in any script.

    Returns:
        True when at least one character is Latin or has a mapping.
    """
    for char in text:
        if char.isascii() and char.isalpha():
            return True
        if char in _COMBINED and _COMBINED[char]:
            return True
    return False


def transliterate(text: str) -> str:
    """Map ``text`` onto a Latin approximation the G2P rules can pronounce.

    Latin characters pass through untouched, so mixed-script input degrades
    gracefully rather than being mangled.

    Args:
        text: Input in any script.

    Returns:
        Latin-script text. Returns an empty string when nothing was mappable,
        which the caller should treat as "no voice available".
    """
    if not text:
        return ""

    # NFC keeps Indic and Arabic combining marks attached to their base letter,
    # which the tables above are written against.
    normalised = unicodedata.normalize("NFC", text)

    pieces: list[str] = []
    unmapped = 0
    devanagari_pending_vowel = False

    for char in normalised:
        # Latin, digits, whitespace and punctuation pass straight through.
        if char.isascii():
            pieces.append(char)
            devanagari_pending_vowel = False
            continue

        if _is_ideographic(char):
            unmapped += 1
            continue

        mapped = _COMBINED.get(char)
        if mapped is None:
            # Cased scripts (Cyrillic, Greek) are tabulated in lowercase only,
            # so retry folded. Without this, every sentence-initial capital was
            # silently dropped.
            lowered = char.lower()
            if lowered != char:
                mapped = _COMBINED.get(lowered)

        if mapped is None:
            # Strip accents and retry: 'é' becomes 'e', 'ā' becomes 'a'.
            decomposed = unicodedata.normalize("NFD", char)
            base = "".join(c for c in decomposed if not unicodedata.combining(c))
            if base and base.isascii():
                pieces.append(base)
            else:
                unmapped += 1
            devanagari_pending_vowel = False
            continue

        # Devanagari virama suppresses the preceding inherent 'a'. Without this
        # every conjunct gains a spurious vowel and the output is unreadable.
        if char == "्" and pieces and pieces[-1].endswith("a"):
            pieces[-1] = pieces[-1][:-1]
            devanagari_pending_vowel = False
            continue

        # An explicit vowel sign replaces the inherent 'a' on the previous
        # consonant rather than adding to it.
        if devanagari_pending_vowel and char in _DEVANAGARI and mapped in {
            "aa", "i", "ee", "u", "oo", "ri", "e", "ai", "o", "au"
        }:
            if pieces and pieces[-1].endswith("a"):
                pieces[-1] = pieces[-1][:-1]

        pieces.append(mapped)
        devanagari_pending_vowel = char in _DEVANAGARI and mapped.endswith("a")

    result = "".join(pieces).strip()

    if unmapped:
        _LOG.debug(
            "Transliteration dropped unmappable characters",
            extra={"dropped": unmapped, "script": script_of(text)},
        )

    return result
