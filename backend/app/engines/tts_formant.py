"""Speech synthesis using our own formant synthesiser.

Wraps :class:`app.tts.synth.FormantSynthesiser` in the engine interface. Unlike
every other backend this one needs no model weights, no network and no API key,
so it is the guaranteed last link in the fallback chain: the service can always
produce speech, for any language, even fully offline.

Quality is robotic but intelligible. See :mod:`app.tts.synth` for the DSP.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.engines.base import SpeechResult, TTSEngine
from app.errors import InferenceError
from app.languages import require_language
from app.logging_conf import get_logger
from app.tts.normalise import normalise_text
from app.tts.synth import SAMPLE_RATE, FormantSynthesiser, SynthesisVoice

__all__ = ["FormantTTSEngine"]

_LOG = get_logger(__name__)

_MIME_TYPE: Final[str] = "audio/wav"
_MAX_SYNTHESIS_CHARS: Final[int] = 2_000

#: Per-language voice tuning. Formant shift approximates vocal-tract length and
#: base pitch sets the speaker's register; both make the output less uniform
#: across languages without needing separate models.
_VOICE_PRESETS: Final[dict[str, SynthesisVoice]] = {
    "en": SynthesisVoice(base_pitch_hz=112.0, speed=1.00, formant_shift=1.00),
    "es": SynthesisVoice(base_pitch_hz=118.0, speed=1.06, formant_shift=1.02),
    "fr": SynthesisVoice(base_pitch_hz=115.0, speed=1.04, formant_shift=1.01),
    "de": SynthesisVoice(base_pitch_hz=105.0, speed=0.96, formant_shift=0.98),
    "it": SynthesisVoice(base_pitch_hz=120.0, speed=1.05, formant_shift=1.02),
    "hi": SynthesisVoice(base_pitch_hz=122.0, speed=0.98, formant_shift=1.03),
    "ja": SynthesisVoice(base_pitch_hz=128.0, speed=1.02, formant_shift=1.05),
    "ru": SynthesisVoice(base_pitch_hz=102.0, speed=0.95, formant_shift=0.97),
}

_DEFAULT_VOICE: Final[SynthesisVoice] = SynthesisVoice()


class FormantTTSEngine(TTSEngine):
    """Our from-scratch DSP synthesiser, exposed as a TTS engine."""

    name = "formant"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._synthesiser = FormantSynthesiser(sample_rate=SAMPLE_RATE)

    def _load(self) -> None:
        """No weights to load — the synthesiser is pure DSP."""

    @staticmethod
    def _to_wav_bytes(audio: NDArray[np.float32], sample_rate: int) -> bytes:
        """Encode float32 samples as a 16-bit mono WAV container."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(
                (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            )
        return buffer.getvalue()

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        """Render ``text`` as speech.

        Every language is accepted. The grapheme-to-phoneme rules are tuned for
        Latin-script text; other scripts are approximated rather than rejected,
        because a rough rendering is more useful than silence in a fallback.

        Args:
            text: Text to speak.
            language: Application language code, used to select a voice preset.

        Returns:
            WAV bytes and their MIME type.

        Raises:
            InferenceError: If the text yields no pronounceable content.
            UnknownLanguageError: If ``language`` is not in the registry.
        """
        stripped = text.strip()
        if not stripped:
            raise InferenceError("Cannot synthesise speech from empty text.")

        require_language(language)  # validates the code, result unused

        spoken = normalise_text(stripped, language)
        if len(spoken) > _MAX_SYNTHESIS_CHARS:
            spoken = spoken[:_MAX_SYNTHESIS_CHARS]

        voice = _VOICE_PRESETS.get(language.split("-")[0], _DEFAULT_VOICE)
        audio = self._synthesiser.synthesise(spoken, voice=voice)

        if audio.size == 0:
            raise InferenceError(
                "The text contained no pronounceable characters.",
                details={"language": language},
            )

        _LOG.debug(
            "Synthesised speech with the formant synthesiser",
            extra={
                "language": language,
                "seconds": round(audio.size / SAMPLE_RATE, 2),
                "characters": len(spoken),
            },
        )

        return SpeechResult(
            audio=self._to_wav_bytes(audio, SAMPLE_RATE),
            mime_type=_MIME_TYPE,
            language=language,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        """Return engine metadata. Always ready; no downloads, no network."""
        return {
            **super().describe(),
            "provider": "in-house formant synthesiser",
            "requires_network": False,
            "requires_weights": False,
            "sample_rate": SAMPLE_RATE,
        }
