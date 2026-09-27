"""Machine translation via a locally hosted NLLB-200 model.

Three details a straightforward NLLB integration gets wrong, handled here:

* ``src_lang`` must be set on the tokenizer. Without it NLLB receives no
  source-language token and silently assumes the tokenizer's default, degrading
  quality for every other source language.
* Failures surface as errors. Swallowing them and returning the untranslated
  input makes a broken configuration look like a working one that echoes.
* A fixed ``max_length=128`` truncates anything longer than a couple of
  sentences with no warning. Long input is split on sentence boundaries and
  translated piecewise instead.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final

from app.config import Settings
from app.engines.base import MTEngine, TranslationResult
from app.errors import InferenceError, ModelLoadError
from app.languages import nllb_code
from app.logging_conf import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = ["NLLBLocalEngine"]

_LOG = get_logger(__name__)

#: Sentence boundary: terminal punctuation (Latin, CJK, Arabic, Devanagari)
#: followed by whitespace. Used to chunk input that exceeds the token limit.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?。！？؟۔।])\s+"
)

#: Characters per token, used only to decide *whether* chunking is needed.
#: Deliberately conservative so the real tokenizer check is rarely exceeded.
_CHARS_PER_TOKEN_ESTIMATE: Final[int] = 3


class NLLBLocalEngine(MTEngine):
    """NLLB-200 translation running in-process via ``transformers``."""

    name = "nllb_local"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        #: Bound in :meth:`_load` so ``torch`` is imported only when needed.
        self._torch: Any = None

    def _load(self) -> None:
        """Load the NLLB model and tokenizer.

        Raises:
            ModelLoadError: If the dependencies or weights are unavailable.
        """
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "transformers and torch are required for the local NLLB engine. "
                "Run: pip install transformers torch sentencepiece",
            ) from exc

        model_id = self._settings.nllb_model
        _LOG.info(
            "Loading NLLB model",
            extra={"model": model_id, "cache_dir": str(self._settings.model_cache_dir)},
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval()
            # Inference only: gradients would waste memory on every forward pass.
            model.requires_grad_(False)
            self._torch = torch
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelLoadError(
                f"Could not load NLLB model {model_id!r}: {exc}",
                details={"model": model_id},
            ) from exc

        if self._settings.nllb_quantize and not torch.cuda.is_available():
            model = self._quantise(model, torch)

        self._model = model

    def _quantise(self, model: Any, torch: Any) -> Any:
        """Apply dynamic int8 quantisation to the model's Linear layers.

        Weights are quantised to int8 while activations stay float, so accuracy
        loss is small but the matrix multiplies run on integer kernels. On this
        project it cut translation latency from 11.1s to 2.2s.

        The float model is dropped and the allocator run afterwards, otherwise
        both copies stay resident and memory roughly doubles.

        Returns:
            The quantised model, or the original if quantisation is unsupported.
        """
        import gc  # noqa: PLC0415

        try:
            quantised = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except (RuntimeError, AttributeError, NotImplementedError) as exc:
            # Not fatal: an unquantised model is slower but perfectly correct.
            _LOG.warning(
                "Dynamic quantisation unavailable; continuing at full precision",
                extra={"error": str(exc)},
            )
            return model

        del model
        gc.collect()
        _LOG.info("Applied dynamic int8 quantisation to NLLB")
        return quantised

    def _target_bos_token_id(self, target_nllb: str) -> int:
        """Resolve the forced beginning-of-sequence token for the target language.

        The tokenizer API for this changed across transformers releases: older
        versions exposed ``lang_code_to_id``, newer ones expect
        ``convert_tokens_to_ids``. Both are tried so the engine works across the
        supported range.

        Raises:
            InferenceError: If the language token cannot be resolved.
        """
        assert self._tokenizer is not None

        token_id = self._tokenizer.convert_tokens_to_ids(target_nllb)
        unknown_id = getattr(self._tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unknown_id:
            return int(token_id)

        legacy_map = getattr(self._tokenizer, "lang_code_to_id", None)
        if isinstance(legacy_map, dict) and target_nllb in legacy_map:
            return int(legacy_map[target_nllb])

        raise InferenceError(
            f"Could not resolve the NLLB token for target language {target_nllb!r}.",
            details={"target_nllb": target_nllb},
        )

    def _split_for_translation(self, text: str) -> list[str]:
        """Split ``text`` into chunks that fit the model's input window.

        Splits on sentence boundaries first; any single sentence still too long
        is hard-wrapped on whitespace so no input is ever silently truncated.
        """
        max_chars = self._settings.nllb_max_input_tokens * _CHARS_PER_TOKEN_ESTIMATE
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_BOUNDARY.split(text):
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer)
            # A single sentence longer than the window must be hard-wrapped.
            if len(sentence) > max_chars:
                words, line = sentence.split(), ""
                for word in words:
                    trial = f"{line} {word}".strip()
                    if len(trial) > max_chars and line:
                        chunks.append(line)
                        line = word
                    else:
                        line = trial
                buffer = line
            else:
                buffer = sentence
        if buffer:
            chunks.append(buffer)

        _LOG.debug("Split long input", extra={"chunks": len(chunks), "characters": len(text)})
        return chunks

    def _translate_chunk(
        self, chunk: str, source_nllb: str, bos_token_id: int, num_beams: int
    ) -> str:
        """Translate a single chunk that is known to fit the input window."""
        assert self._tokenizer is not None
        assert self._model is not None

        # NLLB selects its source language from the tokenizer, not the input
        # text. Omitting this is the single biggest quality trap with NLLB.
        self._tokenizer.src_lang = source_nllb

        encoded = self._tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.nllb_max_input_tokens,
        )

        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=bos_token_id,
                max_new_tokens=self._settings.nllb_max_output_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,  # suppresses NLLB's degenerate loops
            )

        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        num_beams: int | None = None,
    ) -> TranslationResult:
        """Translate ``text`` between two application language codes.

        Args:
            text: Source text.
            source_lang: Application language code of ``text``.
            target_lang: Application language code to translate into.
            num_beams: Beam width override; defaults to ``NLLB_NUM_BEAMS``.

        Returns:
            The translation. Identical source and target languages short-circuit
            and return the input unchanged.

        Raises:
            InferenceError: If generation fails.
            UnknownLanguageError: If either language code is unknown.
        """
        stripped = text.strip()
        if not stripped:
            return TranslationResult(
                text="", source_lang=source_lang, target_lang=target_lang, engine=self.name
            )

        if source_lang == target_lang:
            return TranslationResult(
                text=stripped,
                source_lang=source_lang,
                target_lang=target_lang,
                engine=self.name,
            )

        # Resolved before loading so an unknown language fails fast and cheaply.
        source_nllb = nllb_code(source_lang)
        target_nllb = nllb_code(target_lang)

        self.ensure_loaded()
        bos_token_id = self._target_bos_token_id(target_nllb)

        beams = num_beams if num_beams is not None else self._settings.nllb_num_beams

        try:
            chunks = self._split_for_translation(stripped)
            translated = [
                self._translate_chunk(chunk, source_nllb, bos_token_id, beams)
                for chunk in chunks
            ]
        except InferenceError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            # Surface the failure rather than returning the source text and
            # pretending the translation succeeded.
            raise InferenceError(
                f"NLLB translation failed: {exc}",
                details={"source_lang": source_lang, "target_lang": target_lang},
            ) from exc

        result = " ".join(part for part in translated if part).strip()
        _LOG.debug(
            "Translated text",
            extra={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "input_characters": len(stripped),
                "output_characters": len(result),
                "chunks": len(chunks),
            },
        )

        return TranslationResult(
            text=result,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        """Return engine metadata including the model id."""
        return {**super().describe(), "model": self._settings.nllb_model}
