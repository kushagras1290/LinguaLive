"""Speech synthesis with Meta's MMS-TTS (VITS) models.

MMS publishes a separate VITS checkpoint per language under
``facebook/mms-tts-<iso639-3>``, covering roughly 1100 languages. That pairs
almost one-to-one with NLLB's 202 translation languages, closing the coverage gap
that gTTS alone leaves (68 of 202).

Each checkpoint is around 145 MB, so voices are downloaded lazily on first use
and held in a bounded LRU cache. Downloads that fail — because a language has no
published voice — are remembered as negative results so a missing voice costs one
network round trip per process, not one per request.

Text normalisation is applied by :mod:`app.tts.normalise` before synthesis, since
VITS is character-level and will otherwise spell out digits and symbols.
"""

from __future__ import annotations

import io
import threading
import wave
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.engines.base import SpeechResult, TTSEngine
from app.errors import InferenceError, ModelLoadError, UnsupportedCapabilityError
from app.languages import require_language
from app.logging_conf import get_logger
from app.tts.normalise import normalise_text

if TYPE_CHECKING:
    from transformers import VitsModel

__all__ = ["MMSTTSEngine"]

_LOG = get_logger(__name__)

_MIME_TYPE: Final[str] = "audio/wav"

#: Voices held in memory at once. Each is ~145 MB of float32 weights, so this
#: bounds the resident set at roughly 600 MB in the worst case.
_MAX_CACHED_VOICES: Final[int] = 4

_MAX_SYNTHESIS_CHARS: Final[int] = 2_000


class _VoiceCache:
    """Bounded LRU cache of loaded VITS voices, safe across threads."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        #: Languages known to have no published MMS voice.
        self._missing: set[str] = set()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[Any, Any] | None:
        """Return a cached (model, tokenizer) pair, marking it recently used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: str, value: tuple[Any, Any]) -> None:
        """Insert a voice, evicting the least recently used if full."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                evicted, _ = self._entries.popitem(last=False)
                _LOG.info("Evicted MMS voice from cache", extra={"voice": evicted})

    def mark_missing(self, key: str) -> None:
        """Remember that ``key`` has no published voice."""
        with self._lock:
            self._missing.add(key)

    def is_missing(self, key: str) -> bool:
        """Whether ``key`` was previously found to have no voice."""
        with self._lock:
            return key in self._missing

    def loaded_voices(self) -> list[str]:
        """Names of the currently resident voices."""
        with self._lock:
            return list(self._entries)


class MMSTTSEngine(TTSEngine):
    """Neural speech synthesis via per-language MMS VITS checkpoints."""

    name = "mms_tts"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._cache = _VoiceCache(_MAX_CACHED_VOICES)
        self._torch: Any = None
        self._vits_model_cls: Any = None
        self._tokenizer_cls: Any = None

    def _load(self) -> None:
        """Import the transformers classes. Voices load lazily per language.

        Raises:
            ModelLoadError: If transformers or torch are unavailable.
        """
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoTokenizer, VitsModel  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "transformers and torch are required for MMS-TTS. "
                "Run: pip install transformers torch",
            ) from exc

        self._torch = torch
        self._vits_model_cls = VitsModel
        self._tokenizer_cls = AutoTokenizer

    def _voice_id(self, mms_code: str) -> str:
        """Build the Hub model id for an MMS language code."""
        return f"facebook/mms-tts-{mms_code}"

    def _get_voice(self, mms_code: str) -> tuple[Any, Any]:
        """Load or fetch a cached voice for ``mms_code``.

        Raises:
            UnsupportedCapabilityError: If no voice is published for the language.
            ModelLoadError: If the download or load fails for another reason.
        """
        cached = self._cache.get(mms_code)
        if cached is not None:
            return cached

        if self._cache.is_missing(mms_code):
            raise UnsupportedCapabilityError(
                f"MMS has no published voice for {mms_code!r}.",
                details={"mms_code": mms_code},
            )

        model_id = self._voice_id(mms_code)
        _LOG.info("Loading MMS voice", extra={"voice": mms_code, "model": model_id})

        try:
            # VITS checkpoints store weight-norm parameters (`weight_g`/`weight_v`)
            # that transformers re-parametrises on load. That is expected and
            # harmless, but it prints a several-hundred-line warning per voice.
            from transformers.utils import logging as hf_logging  # noqa: PLC0415

            previous_verbosity = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                tokenizer = self._tokenizer_cls.from_pretrained(model_id)
                model = self._vits_model_cls.from_pretrained(model_id)
            finally:
                hf_logging.set_verbosity(previous_verbosity)
            model.eval()
            model.requires_grad_(False)
        except OSError as exc:
            # A 404 from the Hub means the voice does not exist; remember it so
            # the next request for this language skips straight to the fallback.
            self._cache.mark_missing(mms_code)
            raise UnsupportedCapabilityError(
                f"No MMS voice is available for {mms_code!r}.",
                details={"mms_code": mms_code, "model": model_id},
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise ModelLoadError(
                f"Could not load MMS voice {model_id!r}: {exc}",
                details={"mms_code": mms_code},
            ) from exc

        entry = (model, tokenizer)
        self._cache.put(mms_code, entry)
        return entry

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
        """Render ``text`` as speech using the language's MMS voice.

        Args:
            text: Text to speak.
            language: Application language code.

        Returns:
            WAV bytes and their MIME type.

        Raises:
            InferenceError: If generation fails or produces no audio.
            UnknownLanguageError: If ``language`` is not in the registry.
            UnsupportedCapabilityError: If no MMS voice exists for the language.
        """
        stripped = text.strip()
        if not stripped:
            raise InferenceError("Cannot synthesise speech from empty text.")

        entry = require_language(language)
        mms_code = entry.mms
        if mms_code is None:
            raise UnsupportedCapabilityError(
                f"MMS does not publish a voice for {entry.name}.",
                details={"language": language},
            )

        # VITS is character-level: unnormalised digits and symbols are either
        # spelled out wrongly or dropped.
        spoken = normalise_text(stripped, language)
        if len(spoken) > _MAX_SYNTHESIS_CHARS:
            spoken = spoken[:_MAX_SYNTHESIS_CHARS]
            _LOG.warning(
                "Truncated text for MMS synthesis",
                extra={"limit": _MAX_SYNTHESIS_CHARS, "language": language},
            )

        self.ensure_loaded()
        model, tokenizer = self._get_voice(mms_code)

        try:
            inputs = tokenizer(spoken, return_tensors="pt")

            # Each MMS voice has a script-specific character vocabulary, so text
            # in the wrong script tokenises to nothing. That yields an empty
            # float tensor, and the embedding lookup then fails with an obscure
            # dtype error. Detect it here and report it as a missing capability
            # so the chain falls through to a backend that can cope.
            if inputs["input_ids"].numel() == 0:
                raise UnsupportedCapabilityError(
                    f"The MMS voice for {mms_code!r} cannot represent this text; "
                    "it is probably written in a different script.",
                    details={
                        "language": language,
                        "mms_code": mms_code,
                        "characters": len(spoken),
                    },
                )

            with self._torch.inference_mode():
                output = model(**inputs).waveform
            audio = output.squeeze().cpu().numpy().astype(np.float32)
        except UnsupportedCapabilityError:
            raise
        except (RuntimeError, ValueError, IndexError) as exc:
            raise InferenceError(
                f"MMS synthesis failed for {language!r}: {exc}",
                details={"language": language, "mms_code": mms_code},
            ) from exc

        if audio.size == 0:
            raise InferenceError(
                "MMS synthesis produced no audio.",
                details={"language": language},
            )

        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak * 0.95

        sample_rate = int(model.config.sampling_rate)
        _LOG.debug(
            "Synthesised speech with MMS",
            extra={
                "language": language,
                "mms_code": mms_code,
                "seconds": round(audio.size / sample_rate, 2),
            },
        )

        return SpeechResult(
            audio=self._to_wav_bytes(audio, sample_rate),
            mime_type=_MIME_TYPE,
            language=language,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        """Return engine metadata including which voices are resident."""
        return {
            **super().describe(),
            "provider": "facebook/mms-tts",
            "cached_voices": self._cache.loaded_voices(),
            "cache_capacity": _MAX_CACHED_VOICES,
        }
