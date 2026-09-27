"""A formant speech synthesiser written from scratch.

No pretrained weights, no external synthesis library — just digital signal
processing. This is the fallback that guarantees the service can always speak,
offline, for any language we can supply phoneme rules for, in under 5 MB.

Architecture (a cascade/parallel Klatt-style synthesiser):

    text -> phonemes -> targets -> [ source ] -> [ formant filters ] -> PCM
                                        |               |
                              glottal pulses or     3 resonators
                              fricative noise       at F1, F2, F3

The source-filter model of speech treats the vocal folds as a periodic pulse
train (voiced sounds) or turbulent noise (unvoiced), shaped by vocal-tract
resonances called formants. Vowel identity is carried almost entirely by the
first two formants, which is why F1/F2 targets alone produce intelligible vowels.

The result sounds robotic. It is fully intelligible, entirely deterministic, and
costs nothing to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.logging_conf import get_logger
from app.tts.translit import transliterate

__all__ = [
    "Phoneme",
    "SynthesisVoice",
    "FormantSynthesiser",
    "text_to_phonemes",
    "DEFAULT_VOICE",
]

_LOG = get_logger(__name__)

#: Output sample rate. 22.05 kHz gives 11 kHz of bandwidth, ample for speech.
SAMPLE_RATE: Final[int] = 22_050


@dataclass(frozen=True, slots=True)
class Phoneme:
    """Acoustic target for a single speech sound.

    Attributes:
        symbol: Identifier used by the grapheme-to-phoneme rules.
        f1: First formant in Hz — correlates inversely with tongue height.
        f2: Second formant in Hz — correlates with tongue frontness.
        f3: Third formant in Hz — mostly speaker identity and /r/ colouring.
        duration_ms: Nominal duration before prosodic scaling.
        voiced: Whether the glottal source is periodic.
        amplitude: Relative loudness in [0, 1].
        noise: Fraction of the source that is turbulent noise, for fricatives.
        plosive: Whether the sound begins with a silent closure and burst.
    """

    symbol: str
    f1: float
    f2: float
    f3: float
    duration_ms: float
    voiced: bool
    amplitude: float = 1.0
    noise: float = 0.0
    plosive: bool = False


# --------------------------------------------------------------------------- #
# Phoneme inventory.
#
# Formant values follow the classic Peterson & Barney measurements for an adult
# male speaker; consonant targets approximate their locus frequencies.
# --------------------------------------------------------------------------- #
_PHONEMES: Final[dict[str, Phoneme]] = {
    # --- Vowels ---------------------------------------------------------- #
    "iy": Phoneme("iy", 270, 2290, 3010, 155, True, 1.00),   # beat
    "ih": Phoneme("ih", 390, 1990, 2550, 120, True, 0.95),   # bit
    "ey": Phoneme("ey", 460, 2000, 2600, 165, True, 1.00),   # bait
    "eh": Phoneme("eh", 530, 1840, 2480, 130, True, 0.95),   # bet
    "ae": Phoneme("ae", 660, 1720, 2410, 175, True, 1.00),   # bat
    "aa": Phoneme("aa", 730, 1090, 2440, 180, True, 1.00),   # father
    "ao": Phoneme("ao", 570,  840, 2410, 175, True, 1.00),   # bought
    "ow": Phoneme("ow", 490,  910, 2450, 170, True, 1.00),   # boat
    "uh": Phoneme("uh", 440, 1020, 2240, 120, True, 0.90),   # book
    "uw": Phoneme("uw", 300,  870, 2240, 165, True, 0.95),   # boot
    "ah": Phoneme("ah", 640, 1190, 2390, 120, True, 0.95),   # but
    "er": Phoneme("er", 490, 1350, 1690, 165, True, 0.95),   # bird
    "ay": Phoneme("ay", 660, 1700, 2400, 200, True, 1.00),   # bite
    "aw": Phoneme("aw", 660, 1200, 2400, 200, True, 1.00),   # bout
    "oy": Phoneme("oy", 490, 1100, 2400, 200, True, 1.00),   # boy
    # --- Nasals ---------------------------------------------------------- #
    "m":  Phoneme("m",  280,  900, 2200,  85, True, 0.55),
    "n":  Phoneme("n",  280, 1700, 2600,  80, True, 0.55),
    "ng": Phoneme("ng", 280, 2300, 2750,  85, True, 0.50),
    # --- Liquids and glides ---------------------------------------------- #
    "l":  Phoneme("l",  360, 1300, 2900,  75, True, 0.75),
    "r":  Phoneme("r",  350, 1200, 1600,  80, True, 0.75),
    "w":  Phoneme("w",  300,  610, 2200,  70, True, 0.70),
    "y":  Phoneme("y",  270, 2300, 3000,  65, True, 0.70),
    # --- Voiced fricatives ------------------------------------------------ #
    "v":  Phoneme("v",  340, 1100, 2400,  75, True, 0.45, noise=0.55),
    "dh": Phoneme("dh", 300, 1400, 2600,  70, True, 0.40, noise=0.55),
    "z":  Phoneme("z",  320, 1600, 2600,  85, True, 0.50, noise=0.70),
    "zh": Phoneme("zh", 320, 1800, 2500,  85, True, 0.50, noise=0.70),
    # --- Voiceless fricatives --------------------------------------------- #
    "f":  Phoneme("f",  400, 1600, 2800,  95, False, 0.30, noise=1.0),
    "th": Phoneme("th", 400, 1600, 2800,  90, False, 0.28, noise=1.0),
    "s":  Phoneme("s",  400, 1900, 3400, 105, False, 0.42, noise=1.0),
    "sh": Phoneme("sh", 400, 1800, 2500, 105, False, 0.45, noise=1.0),
    "hh": Phoneme("hh", 500, 1500, 2500,  70, False, 0.22, noise=1.0),
    # --- Plosives ---------------------------------------------------------- #
    "b":  Phoneme("b",  300,  900, 2100,  70, True,  0.55, noise=0.30, plosive=True),
    "d":  Phoneme("d",  300, 1700, 2600,  65, True,  0.55, noise=0.35, plosive=True),
    "g":  Phoneme("g",  300, 1900, 2400,  70, True,  0.55, noise=0.35, plosive=True),
    "p":  Phoneme("p",  400, 1100, 2200,  80, False, 0.38, noise=1.00, plosive=True),
    "t":  Phoneme("t",  400, 1700, 2600,  75, False, 0.42, noise=1.00, plosive=True),
    "k":  Phoneme("k",  400, 1900, 2400,  80, False, 0.42, noise=1.00, plosive=True),
    # --- Affricates -------------------------------------------------------- #
    "ch": Phoneme("ch", 400, 1800, 2500, 110, False, 0.45, noise=1.00, plosive=True),
    "jh": Phoneme("jh", 320, 1800, 2500, 100, True,  0.50, noise=0.70, plosive=True),
    # --- Silence ----------------------------------------------------------- #
    "sil": Phoneme("sil", 500, 1500, 2500, 90, False, 0.0),
}

# --------------------------------------------------------------------------- #
# Grapheme-to-phoneme rules for English.
#
# Ordered longest-first so digraphs match before their component letters. Each
# entry is (pattern, phonemes). Patterns are regular expressions anchored at the
# current cursor position; ``_`` in a pattern marks a word boundary.
# --------------------------------------------------------------------------- #
_G2P_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    # Four- and three-letter sequences
    ("ough", ("ao", "f")),
    ("tion", ("sh", "ah", "n")),
    ("sion", ("zh", "ah", "n")),
    ("ight", ("ay", "t")),
    ("augh", ("ae", "f")),
    ("dge",  ("jh",)),
    ("tch",  ("ch",)),
    ("sch",  ("sh",)),
    ("air",  ("eh", "r")),
    ("are",  ("eh", "r")),
    ("ear",  ("ih", "r")),
    ("eer",  ("ih", "r")),
    ("oor",  ("ao", "r")),
    ("our",  ("aw", "r")),
    ("ure",  ("y", "uw", "r")),
    # Two-letter sequences
    ("ch",  ("ch",)),
    ("sh",  ("sh",)),
    ("th",  ("th",)),
    ("ph",  ("f",)),
    ("gh",  ("g",)),
    ("wh",  ("w",)),
    ("ck",  ("k",)),
    ("ng",  ("ng",)),
    ("qu",  ("k", "w")),
    ("ee",  ("iy",)),
    ("ea",  ("iy",)),
    ("ie",  ("iy",)),
    ("ei",  ("iy",)),
    ("oo",  ("uw",)),
    ("ou",  ("aw",)),
    ("ow",  ("ow",)),
    ("oa",  ("ow",)),
    ("oi",  ("oy",)),
    ("oy",  ("oy",)),
    ("au",  ("ao",)),
    ("aw",  ("ao",)),
    ("ai",  ("ey",)),
    ("ay",  ("ey",)),
    ("ar",  ("aa", "r")),
    ("or",  ("ao", "r")),
    ("er",  ("er",)),
    ("ir",  ("er",)),
    ("ur",  ("er",)),
    ("ui",  ("ih",)),
    ("ey",  ("iy",)),
    # Single letters
    ("a", ("ae",)),
    ("b", ("b",)),
    ("c", ("k",)),
    ("d", ("d",)),
    ("e", ("eh",)),
    ("f", ("f",)),
    ("g", ("g",)),
    ("h", ("hh",)),
    ("i", ("ih",)),
    ("j", ("jh",)),
    ("k", ("k",)),
    ("l", ("l",)),
    ("m", ("m",)),
    ("n", ("n",)),
    ("o", ("aa",)),
    ("p", ("p",)),
    ("q", ("k",)),
    ("r", ("r",)),
    ("s", ("s",)),
    ("t", ("t",)),
    ("u", ("ah",)),
    ("v", ("v",)),
    ("w", ("w",)),
    ("x", ("k", "s")),
    ("y", ("y",)),
    ("z", ("z",)),
)

#: Letters that soften ``c`` and ``g`` when they follow.
_SOFTENING_VOWELS: Final[frozenset[str]] = frozenset("eiy")

#: Vowel letters, used to decide whether ``y`` is acting as a consonant.
_VOWEL_LETTERS: Final[frozenset[str]] = frozenset("aeiou")

#: Vowel phonemes, exempt from consonant degemination.
_VOWEL_PHONEMES: Final[frozenset[str]] = frozenset(
    {"iy", "ih", "ey", "eh", "ae", "aa", "ao", "ow", "uh", "uw", "ah", "er",
     "ay", "aw", "oy"}
)

#: Words common enough that rule-based output is noticeably wrong.
_LEXICON: Final[dict[str, tuple[str, ...]]] = {
    "the": ("dh", "ah"),
    "a": ("ah",),
    "of": ("ah", "v"),
    "to": ("t", "uw"),
    "and": ("ae", "n", "d"),
    "is": ("ih", "z"),
    "was": ("w", "ah", "z"),
    "are": ("aa", "r"),
    "you": ("y", "uw"),
    "i": ("ay",),
    "he": ("hh", "iy"),
    "she": ("sh", "iy"),
    "we": ("w", "iy"),
    "they": ("dh", "ey"),
    "have": ("hh", "ae", "v"),
    "has": ("hh", "ae", "z"),
    "one": ("w", "ah", "n"),
    "two": ("t", "uw"),
    "who": ("hh", "uw"),
    "what": ("w", "ah", "t"),
    "where": ("w", "eh", "r"),
    "there": ("dh", "eh", "r"),
    "here": ("hh", "ih", "r"),
    "said": ("s", "eh", "d"),
    "do": ("d", "uw"),
    "does": ("d", "ah", "z"),
    "world": ("w", "er", "l", "d"),
    "hello": ("hh", "ah", "l", "ow"),
}

_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z']+|[.,!?;:]")


@dataclass(frozen=True, slots=True)
class SynthesisVoice:
    """Voice characteristics applied on top of the phoneme targets.

    Attributes:
        base_pitch_hz: Mean fundamental frequency.
        pitch_range: Fractional pitch excursion for intonation.
        speed: Duration multiplier; higher is faster.
        formant_shift: Scales all formants, approximating vocal-tract length.
        breathiness: Aspiration noise mixed into voiced sounds.
    """

    base_pitch_hz: float = 110.0
    pitch_range: float = 0.18
    speed: float = 1.0
    formant_shift: float = 1.0
    breathiness: float = 0.06


DEFAULT_VOICE: Final[SynthesisVoice] = SynthesisVoice()


def _grapheme_to_phonemes(word: str) -> list[str]:
    """Convert one lowercase word into a phoneme sequence via ordered rules."""
    if word in _LEXICON:
        return list(_LEXICON[word])

    phonemes: list[str] = []
    cursor = 0
    length = len(word)

    while cursor < length:
        # Silent terminal 'e' ("make", "time") is dropped, but not in short words
        # where it carries the only vowel ("be", "he").
        if cursor == length - 1 and word[cursor] == "e" and length > 3:
            break

        # 'y' is the only English letter that is a consonant or a vowel purely by
        # position, so it cannot be handled by the position-free rule table.
        if word[cursor] == "y":
            following = word[cursor + 1 : cursor + 2]
            if cursor == 0 or following in _VOWEL_LETTERS:
                phonemes.append("y")            # yes, beyond  -> consonant
            elif cursor == length - 1:
                # Final 'y': /ay/ in monosyllables (my, try), /iy/ otherwise.
                phonemes.append("ay" if length <= 3 else "iy")
            else:
                phonemes.append("ih")           # synthesizer, system
            cursor += 1
            continue

        for pattern, mapped in _G2P_RULES:
            if not word.startswith(pattern, cursor):
                continue

            following = word[cursor + len(pattern) : cursor + len(pattern) + 1]
            # Soft c/g before e, i, y: "city" -> /s/, "gem" -> /jh/.
            if pattern == "c" and following in _SOFTENING_VOWELS:
                phonemes.append("s")
            elif pattern == "g" and following in _SOFTENING_VOWELS:
                phonemes.append("jh")
            else:
                phonemes.extend(mapped)
            cursor += len(pattern)
            break
        else:
            cursor += 1  # unmapped character (apostrophe, stray symbol)

    if not phonemes:
        return ["ah"]

    # Degeminate: English spells doubled consonants but pronounces one
    # ("happy" is /hapi/, not /happi/). Vowels are exempt, since a repeated
    # vowel phoneme can legitimately span a syllable boundary.
    collapsed: list[str] = [phonemes[0]]
    for symbol in phonemes[1:]:
        if symbol == collapsed[-1] and symbol not in _VOWEL_PHONEMES:
            continue
        collapsed.append(symbol)
    return collapsed


def text_to_phonemes(text: str) -> list[str]:
    """Convert normalised text into a phoneme sequence with pauses.

    Punctuation becomes silence, which is what produces audible phrasing.

    Non-Latin input is transliterated first. The letter-to-sound rules only
    understand Latin letters, so without that step every non-Latin language
    produced silence — and since most of the 202 supported languages are not
    Latin-script, the fallback would have been useless for the majority of them.

    Args:
        text: Normalised text in any script.

    Returns:
        A list of phoneme symbols, always beginning and ending with silence.
        Returns just the bracketing silences when nothing was pronounceable.
    """
    # Cheap check first: pure ASCII skips transliteration entirely.
    if not text.isascii():
        text = transliterate(text)

    tokens = _WORD_PATTERN.findall(text.lower())
    phonemes: list[str] = ["sil"]

    for token in tokens:
        if token in ".!?":
            phonemes.extend(["sil", "sil"])  # sentence-final pause
        elif token in ",;:":
            phonemes.append("sil")           # phrase-internal pause
        else:
            phonemes.extend(_grapheme_to_phonemes(token))
            phonemes.append("sil")           # inter-word boundary

    phonemes.append("sil")
    return phonemes


class FormantSynthesiser:
    """Renders phoneme sequences to audio using source-filter synthesis."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._rng = np.random.default_rng(seed=0)  # deterministic output

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate

    def _glottal_pulses(self, num_samples: int, pitch_hz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate a glottal source with the given pitch contour.

        Uses a Rosenberg-style asymmetric pulse (a raised cosine opening phase
        followed by a steeper closing phase), which has a far more speech-like
        spectral tilt than an impulse train.
        """
        source = np.zeros(num_samples, dtype=np.float64)
        # Integrating instantaneous frequency gives phase, so pitch can glide
        # continuously rather than stepping per frame.
        phase = np.cumsum(pitch_hz / self._sample_rate)
        cycle_position = np.mod(phase, 1.0)

        open_quotient = 0.6
        opening = cycle_position < open_quotient * 0.7
        closing = (cycle_position >= open_quotient * 0.7) & (cycle_position < open_quotient)

        open_phase = cycle_position[opening] / (open_quotient * 0.7)
        source[opening] = 0.5 * (1.0 - np.cos(np.pi * open_phase))

        close_phase = (cycle_position[closing] - open_quotient * 0.7) / (open_quotient * 0.3)
        source[closing] = np.cos(0.5 * np.pi * close_phase)

        return source

    def _resonator(
        self,
        signal_in: NDArray[np.float64],
        frequency: NDArray[np.float64],
        bandwidth: float,
    ) -> NDArray[np.float64]:
        """Apply a time-varying two-pole resonator.

        Implements the standard digital resonator difference equation

            y[n] = a*x[n] + b*y[n-1] + c*y[n-2]

        with coefficients recomputed per sample so formants can glide smoothly
        between phoneme targets. Written as an explicit loop because the
        recurrence is inherently sequential.
        """
        num_samples = signal_in.size
        output = np.zeros(num_samples, dtype=np.float64)

        # Pole radius and angle from bandwidth and centre frequency.
        radius = float(np.exp(-np.pi * bandwidth / self._sample_rate))
        theta = 2.0 * np.pi * frequency / self._sample_rate
        coefficient_b = 2.0 * radius * np.cos(theta)
        coefficient_c = -(radius**2)
        # Normalise so the resonator has unity gain at its centre frequency.
        gain = 1.0 - coefficient_b - coefficient_c

        previous_1 = 0.0
        previous_2 = 0.0
        for index in range(num_samples):
            current = (
                gain[index] * signal_in[index]
                + coefficient_b[index] * previous_1
                + coefficient_c * previous_2
            )
            output[index] = current
            previous_2 = previous_1
            previous_1 = current

        return output

    def _build_contours(
        self, phonemes: list[str], voice: SynthesisVoice
    ) -> tuple[NDArray[np.float64], ...]:
        """Build per-sample formant, amplitude, voicing and noise contours.

        Targets are held for the middle of each phoneme and linearly interpolated
        across boundaries, which produces the formant transitions that carry most
        consonant identity.
        """
        durations = [
            max(1, int(_PHONEMES[p].duration_ms * 0.001 * self._sample_rate / voice.speed))
            for p in phonemes
        ]
        total = sum(durations)

        f1 = np.zeros(total)
        f2 = np.zeros(total)
        f3 = np.zeros(total)
        amplitude = np.zeros(total)
        voicing = np.zeros(total)
        noise = np.zeros(total)

        cursor = 0
        for index, symbol in enumerate(phonemes):
            phoneme = _PHONEMES[symbol]
            length = durations[index]
            segment = slice(cursor, cursor + length)

            f1[segment] = phoneme.f1 * voice.formant_shift
            f2[segment] = phoneme.f2 * voice.formant_shift
            f3[segment] = phoneme.f3 * voice.formant_shift
            voicing[segment] = 1.0 if phoneme.voiced else 0.0
            noise[segment] = phoneme.noise

            # Plosives need a silent closure before the burst, or they sound
            # like fricatives.
            if phoneme.plosive and length > 4:
                closure = length // 3
                amplitude[cursor : cursor + closure] = 0.0
                amplitude[cursor + closure : cursor + length] = phoneme.amplitude
            else:
                amplitude[segment] = phoneme.amplitude

            cursor += length

        # Smooth every contour so transitions glide instead of stepping. The
        # window is ~25 ms, close to natural articulator movement time.
        window_length = max(3, int(0.025 * self._sample_rate))
        kernel = np.hanning(window_length)
        kernel /= kernel.sum()

        def smooth(values: NDArray[np.float64]) -> NDArray[np.float64]:
            padded = np.pad(values, window_length, mode="edge")
            return np.convolve(padded, kernel, mode="same")[
                window_length : window_length + total
            ]

        return smooth(f1), smooth(f2), smooth(f3), smooth(amplitude), smooth(voicing), smooth(noise)

    def _pitch_contour(self, num_samples: int, voice: SynthesisVoice) -> NDArray[np.float64]:
        """Build a declining pitch contour with a phrase-final fall.

        Real speech drifts downward across an utterance (declination) and falls
        sharply at the end of a declarative. A flat contour is the single most
        robotic-sounding thing a synthesiser can do, so both are modelled.
        """
        position = np.linspace(0.0, 1.0, num_samples)
        declination = 1.0 - 0.18 * position
        final_fall = np.where(position > 0.85, 1.0 - 2.0 * (position - 0.85), 1.0)
        # A slow sinusoid adds the micro-variation that keeps it from buzzing.
        micro_variation = 1.0 + voice.pitch_range * 0.25 * np.sin(2.0 * np.pi * 1.7 * position)
        return voice.base_pitch_hz * declination * final_fall * micro_variation

    def synthesise(
        self, text: str, *, voice: SynthesisVoice = DEFAULT_VOICE
    ) -> NDArray[np.float32]:
        """Render ``text`` to audio.

        Args:
            text: Normalised text.
            voice: Voice characteristics.

        Returns:
            Mono float32 samples in [-1, 1] at :attr:`sample_rate`. Returns an
            empty array for empty input.
        """
        phonemes = text_to_phonemes(text)
        if len(phonemes) <= 2:  # nothing but the bracketing silences
            return np.zeros(0, dtype=np.float32)

        f1, f2, f3, amplitude, voicing, noise = self._build_contours(phonemes, voice)
        total = f1.size

        pitch = self._pitch_contour(total, voice)
        voiced_source = self._glottal_pulses(total, pitch)
        noise_source = self._rng.normal(0.0, 1.0, total)

        # Mix periodic and turbulent sources per sample. Breathiness adds a
        # little noise even to fully voiced sounds, which reduces the buzz.
        source = (
            voiced_source * voicing * (1.0 - noise) * (1.0 - voice.breathiness)
            + noise_source * noise * 0.35
            + noise_source * voicing * voice.breathiness * 0.15
        )

        # Cascade the three formant resonators. Bandwidths widen with formant
        # number, matching measured vocal-tract damping.
        shaped = self._resonator(source, f1, bandwidth=80.0)
        shaped = self._resonator(shaped, f2, bandwidth=110.0)
        shaped = self._resonator(shaped, f3, bandwidth=170.0)

        shaped *= amplitude

        # Radiation from the lips acts as a first-order differentiator, adding
        # roughly +6 dB/octave. Omitting it sounds muffled.
        shaped = np.diff(shaped, prepend=0.0)

        peak = float(np.max(np.abs(shaped)))
        if peak < 1e-9:
            return np.zeros(0, dtype=np.float32)
        shaped = shaped / peak * 0.85

        # 10 ms fades prevent clicks at the boundaries.
        fade_length = min(int(0.01 * self._sample_rate), total // 2)
        if fade_length > 0:
            fade = np.linspace(0.0, 1.0, fade_length)
            shaped[:fade_length] *= fade
            shaped[-fade_length:] *= fade[::-1]

        _LOG.debug(
            "Synthesised with formant synthesiser",
            extra={
                "phonemes": len(phonemes),
                "samples": total,
                "seconds": round(total / self._sample_rate, 2),
            },
        )
        return shaped.astype(np.float32)
