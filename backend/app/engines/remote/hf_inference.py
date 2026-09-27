"""Remote engines backed by the Hugging Face Inference API.

These exist so the service can run on a host that cannot fit the model weights in
memory. They call the *same* models used locally — ``openai/whisper-large-v3``
and ``facebook/nllb-200-distilled-600M`` — so switching engines changes latency
and cost, never the set of supported languages.

Selecting these requires ``HF_TOKEN``; :meth:`Settings.validate_runtime_requirements`
enforces that at startup rather than failing on the first request.
"""

from __future__ import annotations

import io
import time
import wave
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.engines.base import (
    ASREngine,
    MTEngine,
    TranscriptionResult,
    TranslationResult,
)
from app.errors import InferenceError, InferenceTimeoutError, ModelLoadError
from app.languages import nllb_code
from app.logging_conf import get_logger

__all__ = ["HFInferenceASREngine", "HFInferenceMTEngine"]

_LOG = get_logger(__name__)

#: HTTP 503 from the Inference API means the model is warming up, not failing.
_MODEL_LOADING_STATUS: Final[int] = 503
_RATE_LIMITED_STATUS: Final[int] = 429

#: Cold starts on the Inference API routinely take 20-30s.
_MAX_COLD_START_RETRIES: Final[int] = 3
_RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (5.0, 10.0, 20.0)


def _require_requests() -> Any:
    """Import ``requests`` or raise a typed error."""
    try:
        import requests  # noqa: PLC0415

        return requests
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelLoadError(
            "The 'requests' package is required for remote engines."
        ) from exc


class _HFInferenceBase:
    """Shared HTTP plumbing for Inference API calls."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._requests = None  # bound on first load

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.hf_token}",
            "User-Agent": "real-time-translation/1.0",
        }

    def _endpoint(self, model_id: str) -> str:
        return f"{self._settings.hf_inference_base_url.rstrip('/')}/{model_id}"

    def _post(
        self,
        model_id: str,
        *,
        data: bytes | None = None,
        json_body: dict[str, Any] | None = None,
        content_type: str | None = None,
    ) -> Any:
        """POST to the Inference API, retrying while the model warms up.

        Returns:
            The parsed JSON response body.

        Raises:
            InferenceTimeoutError: If the request exceeds its deadline.
            InferenceError: For every other failure, including exhausted retries.
        """
        requests = self._requests
        assert requests is not None, "engine was not loaded"

        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type

        url = self._endpoint(model_id)
        last_error = "unknown error"

        for attempt in range(_MAX_COLD_START_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    json=json_body,
                    timeout=self._settings.hf_timeout_seconds,
                )
            except requests.Timeout as exc:
                raise InferenceTimeoutError(
                    f"The Hugging Face Inference API did not respond within "
                    f"{self._settings.hf_timeout_seconds:.0f}s.",
                    details={"model": model_id},
                ) from exc
            except requests.RequestException as exc:
                raise InferenceError(
                    f"Could not reach the Hugging Face Inference API: {exc}",
                    details={"model": model_id},
                ) from exc

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise InferenceError(
                        "The Inference API returned a malformed response.",
                        details={"model": model_id},
                    ) from exc

            if response.status_code in (_MODEL_LOADING_STATUS, _RATE_LIMITED_STATUS):
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < _MAX_COLD_START_RETRIES:
                    delay = _RETRY_BACKOFF_SECONDS[
                        min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    _LOG.info(
                        "Inference API not ready; retrying",
                        extra={
                            "model": model_id,
                            "status": response.status_code,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                        },
                    )
                    time.sleep(delay)
                    continue

            raise InferenceError(
                f"The Inference API returned HTTP {response.status_code}.",
                details={"model": model_id, "body": response.text[:200]},
            )

        raise InferenceError(
            f"The Inference API remained unavailable after "
            f"{_MAX_COLD_START_RETRIES} retries: {last_error}",
            details={"model": model_id},
        )


class HFInferenceASREngine(_HFInferenceBase, ASREngine):
    """Whisper transcription via the Hugging Face Inference API."""

    name = "hf_inference_asr"

    def __init__(self, settings: Settings) -> None:
        ASREngine.__init__(self)
        _HFInferenceBase.__init__(self, settings)

    def _load(self) -> None:
        self._requests = _require_requests()

    @staticmethod
    def _to_wav_bytes(audio: NDArray[np.float32], sample_rate: int) -> bytes:
        """Encode float32 audio as a 16-bit WAV container for upload."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
        return buffer.getvalue()

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio remotely.

        The Inference API's ASR endpoint auto-detects language and does not accept
        a language hint, so ``language`` is used only to label the result.
        """
        self.ensure_loaded()
        if audio.size == 0:
            return TranscriptionResult(text="", language=language or "en", engine=self.name)

        payload = self._post(
            self._settings.hf_asr_model,
            data=self._to_wav_bytes(audio, sample_rate),
            content_type="audio/wav",
        )

        if isinstance(payload, dict) and "text" in payload:
            text = str(payload["text"]).strip()
        elif isinstance(payload, dict) and "error" in payload:
            raise InferenceError(
                f"Remote transcription failed: {payload['error']}",
                details={"model": self._settings.hf_asr_model},
            )
        else:
            raise InferenceError(
                "The Inference API returned an unexpected transcription payload.",
                details={"model": self._settings.hf_asr_model},
            )

        duration = float(audio.size) / float(sample_rate)
        return TranscriptionResult(
            text=text,
            language=language or "en",
            duration=duration,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        return {**ASREngine.describe(self), "model": self._settings.hf_asr_model, "remote": True}


class HFInferenceMTEngine(_HFInferenceBase, MTEngine):
    """NLLB translation via the Hugging Face Inference API."""

    name = "hf_inference_mt"

    def __init__(self, settings: Settings) -> None:
        MTEngine.__init__(self)
        _HFInferenceBase.__init__(self, settings)

    def _load(self) -> None:
        self._requests = _require_requests()

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        num_beams: int | None = None,
    ) -> TranslationResult:
        """Translate ``text`` remotely using the same NLLB model as the local engine.

        ``num_beams`` is accepted for interface compatibility but ignored: the
        hosted endpoint does not expose decoder settings.
        """
        del num_beams
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

        source_nllb = nllb_code(source_lang)
        target_nllb = nllb_code(target_lang)
        self.ensure_loaded()

        payload = self._post(
            self._settings.nllb_model,
            json_body={
                "inputs": stripped,
                "parameters": {"src_lang": source_nllb, "tgt_lang": target_nllb},
                "options": {"wait_for_model": True},
            },
        )

        if isinstance(payload, list) and payload and "translation_text" in payload[0]:
            translated = str(payload[0]["translation_text"]).strip()
        elif isinstance(payload, dict) and "error" in payload:
            raise InferenceError(
                f"Remote translation failed: {payload['error']}",
                details={"source_lang": source_lang, "target_lang": target_lang},
            )
        else:
            raise InferenceError(
                "The Inference API returned an unexpected translation payload.",
                details={"model": self._settings.nllb_model},
            )

        return TranslationResult(
            text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=self.name,
        )

    def describe(self) -> dict[str, Any]:
        return {**MTEngine.describe(self), "model": self._settings.nllb_model, "remote": True}
