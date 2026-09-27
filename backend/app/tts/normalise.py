"""Text normalisation for speech synthesis.

Acoustic models are trained on spoken-form text. Written-form input — digits,
currency, abbreviations, URLs — is either mispronounced or skipped entirely, so
it must be expanded to words before synthesis. This is the single highest-impact
stage of a TTS pipeline and the one most often skipped.

Number expansion is implemented for the languages where it matters most and
falls back to digit-by-digit reading elsewhere, which is always intelligible even
when it is not idiomatic.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Final

from app.logging_conf import get_logger

__all__ = ["normalise_text", "expand_number", "SUPPORTED_NUMBER_LANGUAGES"]

_LOG = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Number expansion
# --------------------------------------------------------------------------- #
_EN_UNITS: Final[tuple[str, ...]] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_EN_TENS: Final[tuple[str, ...]] = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_EN_SCALES: Final[tuple[tuple[int, str], ...]] = (
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
    (100, "hundred"),
)

_ES_UNITS: Final[tuple[str, ...]] = (
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
    "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
    "diecisiete", "dieciocho", "diecinueve",
)
_ES_TENS: Final[tuple[str, ...]] = (
    "", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
    "ochenta", "noventa",
)

_FR_UNITS: Final[tuple[str, ...]] = (
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
)
_FR_TENS: Final[tuple[str, ...]] = (
    "", "", "vingt", "trente", "quarante", "cinquante", "soixante",
    "soixante-dix", "quatre-vingt", "quatre-vingt-dix",
)

_DE_UNITS: Final[tuple[str, ...]] = (
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht",
    "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
    "sechzehn", "siebzehn", "achtzehn", "neunzehn",
)
_DE_TENS: Final[tuple[str, ...]] = (
    "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig",
    "achtzig", "neunzig",
)

_HI_UNITS: Final[tuple[str, ...]] = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
)

#: Digit names for the language-agnostic fallback.
_DIGIT_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "en": _EN_UNITS[:10],
    "es": _ES_UNITS[:10],
    "fr": _FR_UNITS[:10],
    "de": _DE_UNITS[:10],
    "hi": _HI_UNITS[:10],
}


def _expand_en(value: int) -> str:
    """Expand a non-negative integer into English words."""
    if value < 20:
        return _EN_UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _EN_TENS[tens] + (f"-{_EN_UNITS[unit]}" if unit else "")
    for scale, name in _EN_SCALES:
        if value >= scale:
            count, remainder = divmod(value, scale)
            words = f"{_expand_en(count)} {name}"
            if remainder:
                joiner = " and " if remainder < 100 and scale == 100 else " "
                words = f"{words}{joiner}{_expand_en(remainder)}"
            return words
    return str(value)  # pragma: no cover - unreachable for non-negative ints


def _expand_es(value: int) -> str:
    """Expand a non-negative integer into Spanish words (0-999 exact)."""
    if value < 20:
        return _ES_UNITS[value]
    if value < 30:
        return f"veinti{_ES_UNITS[value - 20]}"
    if value < 100:
        tens, unit = divmod(value, 10)
        return _ES_TENS[tens] + (f" y {_ES_UNITS[unit]}" if unit else "")
    if value == 100:
        return "cien"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        head = "ciento" if hundreds == 1 else f"{_ES_UNITS[hundreds]}cientos"
        return f"{head} {_expand_es(remainder)}".strip() if remainder else head
    thousands, remainder = divmod(value, 1000)
    head = "mil" if thousands == 1 else f"{_expand_es(thousands)} mil"
    return f"{head} {_expand_es(remainder)}".strip() if remainder else head


def _expand_fr(value: int) -> str:
    """Expand a non-negative integer into French words (0-999 exact)."""
    if value < 20:
        return _FR_UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        if tens in (7, 9):  # soixante-dix / quatre-vingt-dix families
            base = _FR_TENS[tens - 1]
            return f"{base}-{_FR_UNITS[10 + unit]}"
        if unit == 1 and tens != 8:
            return f"{_FR_TENS[tens]} et un"
        return _FR_TENS[tens] + (f"-{_FR_UNITS[unit]}" if unit else "")
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        head = "cent" if hundreds == 1 else f"{_FR_UNITS[hundreds]} cents"
        return f"{head} {_expand_fr(remainder)}".strip() if remainder else head
    thousands, remainder = divmod(value, 1000)
    head = "mille" if thousands == 1 else f"{_expand_fr(thousands)} mille"
    return f"{head} {_expand_fr(remainder)}".strip() if remainder else head


def _expand_de(value: int) -> str:
    """Expand a non-negative integer into German words (0-999 exact)."""
    if value < 20:
        return _DE_UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        if unit:
            unit_word = "ein" if unit == 1 else _DE_UNITS[unit]
            return f"{unit_word}und{_DE_TENS[tens]}"
        return _DE_TENS[tens]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        head = f"{_DE_UNITS[hundreds]}hundert"
        return f"{head}{_expand_de(remainder)}" if remainder else head
    thousands, remainder = divmod(value, 1000)
    head = f"{_expand_de(thousands)}tausend"
    return f"{head}{_expand_de(remainder)}" if remainder else head


def _expand_hi(value: int) -> str:
    """Expand a small non-negative integer into Hindi words.

    Hindi has irregular forms for every value from 0-99, so only 0-19 are
    expanded exactly; larger values fall back to digit reading, which remains
    intelligible.
    """
    if value < 20:
        return _HI_UNITS[value]
    return " ".join(_HI_UNITS[int(digit)] for digit in str(value))


#: Integer expanders keyed by application language code.
_NUMBER_EXPANDERS: Final[dict[str, Callable[[int], str]]] = {
    "en": _expand_en,
    "es": _expand_es,
    "fr": _expand_fr,
    "de": _expand_de,
    "hi": _expand_hi,
}

SUPPORTED_NUMBER_LANGUAGES: Final[frozenset[str]] = frozenset(_NUMBER_EXPANDERS)

#: Upper bound for word expansion. Beyond this, digit reading is clearer anyway
#: and avoids absurd output like "nine hundred billion ...".
_MAX_EXPANDABLE: Final[int] = 999_999_999_999


def expand_number(token: str, language: str) -> str:
    """Expand a numeric token into spoken words.

    Args:
        token: A bare integer or decimal, optionally signed.
        language: Application language code.

    Returns:
        The spoken form, or digit-by-digit reading when the language has no
        expander or the value is out of range.
    """
    base = language.split("-")[0]
    expander = _NUMBER_EXPANDERS.get(base)
    digits = _DIGIT_NAMES.get(base, _EN_UNITS[:10])

    negative = token.startswith("-")
    cleaned = token.lstrip("+-").replace(",", "")

    integer_part, _, fraction_part = cleaned.partition(".")

    def read_digits(text: str) -> str:
        return " ".join(digits[int(ch)] for ch in text if ch.isdigit())

    if expander is None or not integer_part.isdigit() or int(integer_part) > _MAX_EXPANDABLE:
        spoken = read_digits(integer_part)
    else:
        spoken = expander(int(integer_part))

    if fraction_part:
        point = {"es": "punto", "fr": "virgule", "de": "Komma", "hi": "दशमलव"}.get(
            base, "point"
        )
        spoken = f"{spoken} {point} {read_digits(fraction_part)}"

    if negative:
        minus = {"es": "menos", "fr": "moins", "de": "minus", "hi": "ऋण"}.get(base, "minus")
        spoken = f"{minus} {spoken}"

    return spoken.strip()


# --------------------------------------------------------------------------- #
# Symbol and abbreviation expansion
# --------------------------------------------------------------------------- #
_CURRENCY: Final[dict[str, dict[str, str]]] = {
    "en": {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen", "₹": "rupees"},
    "es": {"$": "dólares", "€": "euros", "£": "libras", "¥": "yenes", "₹": "rupias"},
    "fr": {"$": "dollars", "€": "euros", "£": "livres", "¥": "yens", "₹": "roupies"},
    "de": {"$": "Dollar", "€": "Euro", "£": "Pfund", "¥": "Yen", "₹": "Rupien"},
    "hi": {"$": "डॉलर", "€": "यूरो", "£": "पाउंड", "¥": "येन", "₹": "रुपये"},
}

_SYMBOLS: Final[dict[str, dict[str, str]]] = {
    "en": {"&": " and ", "%": " percent ", "+": " plus ", "=": " equals ", "@": " at "},
    "es": {"&": " y ", "%": " por ciento ", "+": " más ", "=": " igual ", "@": " arroba "},
    "fr": {"&": " et ", "%": " pour cent ", "+": " plus ", "=": " égale ", "@": " arobase "},
    "de": {"&": " und ", "%": " Prozent ", "+": " plus ", "=": " gleich ", "@": " at "},
    "hi": {"&": " और ", "%": " प्रतिशत ", "+": " जोड़ ", "=": " बराबर ", "@": " ऐट "},
}

_EN_ABBREVIATIONS: Final[dict[str, str]] = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss",
    "prof": "professor", "st": "street", "rd": "road", "ave": "avenue",
    "etc": "et cetera", "vs": "versus", "approx": "approximately",
    "e.g": "for example", "i.e": "that is", "no": "number",
}

_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_REPEATED_PUNCT: Final[re.Pattern[str]] = re.compile(r"([!?.,;:])\1+")

#: Characters that carry no pronunciation and only confuse a character-level
#: acoustic model.
_STRIP_CHARS: Final[re.Pattern[str]] = re.compile(r"[*_`~^<>{}\[\]|\\]")


def normalise_text(text: str, language: str = "en") -> str:
    """Convert written-form text into spoken form.

    The stages are ordered so that later rules cannot corrupt earlier output:
    URLs and emails are replaced first (they contain dots and digits that would
    otherwise be expanded), then currency, symbols, abbreviations, and finally
    bare numbers.

    Args:
        text: Raw input text.
        language: Application language code, used to pick localised word forms.

    Returns:
        Normalised text ready for phonemisation or a character-level model.
    """
    if not text or not text.strip():
        return ""

    base = language.split("-")[0]
    # NFKC folds ligatures and full-width forms into their canonical equivalents,
    # which keeps downstream character tables small.
    working = unicodedata.normalize("NFKC", text)

    spoken_url = {"es": " enlace ", "fr": " lien ", "de": " Link ", "hi": " लिंक "}.get(
        base, " link "
    )
    working = _URL_PATTERN.sub(spoken_url, working)

    at_word = _SYMBOLS.get(base, _SYMBOLS["en"])["@"]
    working = _EMAIL_PATTERN.sub(
        lambda match: match.group(0).replace("@", at_word).replace(".", " dot "), working
    )

    working = _STRIP_CHARS.sub(" ", working)

    # Currency symbol before the amount reads after it in speech:
    # "$5" -> "5 dollars".
    for symbol, word in _CURRENCY.get(base, _CURRENCY["en"]).items():
        working = re.sub(
            rf"{re.escape(symbol)}\s*({_NUMBER_PATTERN.pattern})",
            rf"\1 {word}",
            working,
        )
        working = working.replace(symbol, f" {word} ")

    for symbol, word in _SYMBOLS.get(base, _SYMBOLS["en"]).items():
        working = working.replace(symbol, word)

    if base == "en":
        def _expand_abbreviation(match: re.Match[str]) -> str:
            word = match.group(0).rstrip(".").lower()
            return _EN_ABBREVIATIONS.get(word, match.group(0))

        working = re.sub(r"\b[A-Za-z]{1,6}\.", _expand_abbreviation, working)

    working = _NUMBER_PATTERN.sub(
        lambda match: f" {expand_number(match.group(0), base)} ", working
    )

    working = _REPEATED_PUNCT.sub(r"\1", working)
    working = _WHITESPACE.sub(" ", working).strip()

    _LOG.debug(
        "Normalised text",
        extra={"language": language, "before": len(text), "after": len(working)},
    )
    return working
