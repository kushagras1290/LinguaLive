"""Application configuration, validated at import time.

Settings are read from environment variables and an optional ``.env`` file.
Invalid values raise :class:`~app.errors.ConfigurationError` immediately so the
process dies at startup instead of failing on the first request.

.. important::
   Importing this module has the side effect of pointing the Hugging Face cache
   environment variables at :attr:`Settings.model_cache_dir`. That must happen
   *before* ``transformers``/``huggingface_hub`` are imported anywhere, which is
   why every engine imports its heavy dependencies lazily. See
   :func:`configure_model_cache`.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigurationError

__all__ = [
    "ASREngineName",
    "MTEngineName",
    "TTSEngineName",
    "Environment",
    "LogFormat",
    "Settings",
    "get_settings",
    "configure_model_cache",
    "PROJECT_ROOT",
]

# backend/app/config.py -> backend/app -> backend -> <project root>
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Whisper hard-codes a 30 second receptive field; anything longer is chunked by
# faster-whisper internally. This ceiling exists to bound memory, not accuracy.
_MAX_SUPPORTED_AUDIO_SECONDS: Final[float] = 600.0


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class LogFormat(StrEnum):
    """Log rendering style."""

    JSON = "json"
    CONSOLE = "console"


class ASREngineName(StrEnum):
    """Available speech-recognition backends."""

    FASTER_WHISPER = "faster_whisper"
    HF_INFERENCE = "hf_inference"


class MTEngineName(StrEnum):
    """Available machine-translation backends."""

    NLLB_LOCAL = "nllb_local"
    HF_INFERENCE = "hf_inference"


class TTSEngineName(StrEnum):
    """Available speech-synthesis backends.

    ``CHAIN`` is the default and degrades MMS -> gTTS -> formant, so synthesis
    succeeds for any language in the registry. The individual names select a
    single backend with no fallback, which is what the tests use.
    """

    CHAIN = "chain"
    MMS = "mms"
    GTTS = "gtts"
    FORMANT = "formant"


class Settings(BaseSettings):
    """Validated application settings.

    All fields are populated from the environment. Names are case-insensitive and
    may be prefixed in a ``.env`` file at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- runtime
    environment: Environment = Environment.DEVELOPMENT
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 5000
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    # ------------------------------------------------------------------ paths
    model_cache_dir: Path = PROJECT_ROOT / "models"

    # ---------------------------------------------------------------- engines
    engine_asr: ASREngineName = ASREngineName.FASTER_WHISPER
    engine_mt: MTEngineName = MTEngineName.NLLB_LOCAL
    engine_tts: TTSEngineName = TTSEngineName.CHAIN

    #: Backends the ``chain`` TTS engine tries, in order. The final entry should
    #: be one that cannot fail so synthesis always yields audio.
    tts_chain: str = "mms,gtts,formant"

    #: Load models during application construction rather than on first use.
    #: Off by default so the dev server starts instantly; enable in production
    #: so the first real request does not pay the load cost.
    eager_load_models: bool = False

    # ---------------------------------------------------------------- whisper
    #: A faster-whisper size alias (``tiny``/``base``/``small``/``medium``/
    #: ``large-v3``) or any CTranslate2 model id on the Hub.
    whisper_model: str = "small"
    #: ``auto`` resolves to CUDA when available, otherwise CPU.
    whisper_device: str = "auto"
    #: ``auto`` resolves to ``int8`` on CPU and ``float16`` on CUDA.
    whisper_compute_type: str = "auto"
    whisper_beam_size: Annotated[int, Field(ge=1, le=10)] = 5
    #: Drop segments whose average token log-probability falls below this.
    whisper_logprob_threshold: float = -1.0

    # ------------------------------------------------------------------- nllb
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_max_input_tokens: Annotated[int, Field(ge=16, le=1024)] = 512
    nllb_max_output_tokens: Annotated[int, Field(ge=16, le=1024)] = 512
    nllb_num_beams: Annotated[int, Field(ge=1, le=8)] = 4
    #: Apply dynamic int8 quantisation to the Linear layers on CPU. Measured on
    #: this project: 11.1s -> 2.2s per translation, a 5x speedup, with output
    #: quality preserved. Ignored on CUDA, where fp16 is already faster.
    nllb_quantize: bool = True
    #: Beam count used by the streaming path.
    #:
    #: Deliberately the same as the batch setting. Lowering it to 1 or 2 saves
    #: only ~0.7s once quantisation is enabled, but measurably degrades output:
    #: at beams<=2 this model rendered "brown fox" into Hindi as an obscenity,
    #: which beam search at width 4 avoids. Latency is not worth that risk.
    nllb_stream_num_beams: Annotated[int, Field(ge=1, le=8)] = 4

    # ----------------------------------------------------------------- remote
    #: Required only when an engine is set to ``hf_inference``.
    hf_token: str | None = None
    hf_inference_base_url: str = "https://api-inference.huggingface.co/models"
    hf_asr_model: str = "openai/whisper-large-v3"
    hf_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 60.0

    # ------------------------------------------------------------------ limits
    max_upload_bytes: Annotated[int, Field(ge=1024, le=100 * 1024 * 1024)] = 25 * 1024 * 1024
    max_audio_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_text_chars: Annotated[int, Field(ge=1, le=50_000)] = 5_000
    inference_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 120.0

    # -------------------------------------------------------------- streaming
    #: Sample rate the browser AudioWorklet must resample to before sending.
    stream_sample_rate: Annotated[int, Field(ge=8000, le=48000)] = 16_000
    #: Rolling window of audio kept for re-transcription, in seconds.
    stream_window_seconds: Annotated[float, Field(gt=0, le=30)] = 8.0
    #: Minimum silence that closes an utterance and promotes it to ``final``.
    stream_silence_seconds: Annotated[float, Field(gt=0, le=5)] = 0.8
    #: Minimum audio accumulated before the first partial transcription runs.
    stream_min_chunk_seconds: Annotated[float, Field(gt=0, le=10)] = 1.0
    #: Amplitude below which a frame counts as silence (0..1, peak normalised).
    stream_vad_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.015
    #: Reject streaming sessions that exceed this wall-clock duration.
    stream_max_session_seconds: Annotated[float, Field(gt=0)] = 900.0

    # ------------------------------------------------------------------- cors
    #: Comma-separated list of allowed browser origins.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # -------------------------------------------------------------- validators
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        normalised = value.strip().upper()
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return normalised

    @field_validator("whisper_device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        normalised = value.strip().lower()
        if normalised not in allowed:
            raise ValueError(f"whisper_device must be one of {sorted(allowed)}, got {value!r}")
        return normalised

    @field_validator("whisper_compute_type")
    @classmethod
    def _validate_compute_type(cls, value: str) -> str:
        allowed = {"auto", "int8", "int8_float16", "int8_float32", "float16", "float32"}
        normalised = value.strip().lower()
        if normalised not in allowed:
            raise ValueError(
                f"whisper_compute_type must be one of {sorted(allowed)}, got {value!r}"
            )
        return normalised

    @field_validator("max_audio_seconds")
    @classmethod
    def _validate_audio_ceiling(cls, value: float) -> float:
        if value > _MAX_SUPPORTED_AUDIO_SECONDS:
            raise ValueError(
                f"max_audio_seconds must not exceed {_MAX_SUPPORTED_AUDIO_SECONDS}"
            )
        return value

    @field_validator("model_cache_dir")
    @classmethod
    def _expand_cache_dir(cls, value: Path) -> Path:
        return Path(os.path.expandvars(str(value))).expanduser().resolve()

    # ------------------------------------------------------------- properties
    @property
    def is_production(self) -> bool:
        """True when running with production hardening enabled."""
        return self.environment is Environment.PRODUCTION

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a cleaned list, or ``["*"]`` for wildcard."""
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins or ["*"]

    @property
    def tts_chain_list(self) -> list[str]:
        """The TTS fallback chain as a cleaned list of backend names."""
        names = [name.strip().lower() for name in self.tts_chain.split(",") if name.strip()]
        return names or ["formant"]

    @property
    def uses_remote_engine(self) -> bool:
        """True when any configured engine calls the Hugging Face Inference API."""
        return (
            self.engine_asr is ASREngineName.HF_INFERENCE
            or self.engine_mt is MTEngineName.HF_INFERENCE
        )

    def resolved_whisper_device(self) -> str:
        """Resolve ``auto`` to a concrete device by probing for CUDA.

        Falls back to CPU when torch is absent, since faster-whisper itself does
        not depend on torch.
        """
        if self.whisper_device != "auto":
            return self.whisper_device
        try:
            import torch  # noqa: PLC0415 - optional, probed lazily

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolved_whisper_compute_type(self, device: str) -> str:
        """Resolve ``auto`` to the fastest safe precision for ``device``."""
        if self.whisper_compute_type != "auto":
            return self.whisper_compute_type
        return "float16" if device == "cuda" else "int8"

    def validate_runtime_requirements(self) -> None:
        """Check cross-field constraints that pydantic cannot express alone.

        Raises:
            ConfigurationError: If a selected engine is missing its credentials
                or the model cache directory cannot be created.
        """
        if self.uses_remote_engine and not self.hf_token:
            raise ConfigurationError(
                "HF_TOKEN is required when ENGINE_ASR or ENGINE_MT is set to "
                "'hf_inference'. Set it in your .env file or switch the engine "
                "back to a local implementation.",
                details={"engine_asr": self.engine_asr, "engine_mt": self.engine_mt},
            )

        try:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Model cache directory {self.model_cache_dir} is not writable: {exc}",
                details={"model_cache_dir": str(self.model_cache_dir)},
            ) from exc

        if self.stream_min_chunk_seconds > self.stream_window_seconds:
            raise ConfigurationError(
                "STREAM_MIN_CHUNK_SECONDS cannot exceed STREAM_WINDOW_SECONDS.",
                details={
                    "stream_min_chunk_seconds": self.stream_min_chunk_seconds,
                    "stream_window_seconds": self.stream_window_seconds,
                },
            )


def configure_model_cache(settings: Settings) -> Path:
    """Point every Hugging Face cache variable at the configured directory.

    This keeps multi-gigabyte weights off the system drive. It must run before
    ``transformers``, ``huggingface_hub`` or ``faster_whisper`` are imported,
    because those libraries snapshot the cache path at import time.

    Args:
        settings: Validated settings supplying ``model_cache_dir``.

    Returns:
        The resolved cache directory, guaranteed to exist.

    Raises:
        ConfigurationError: If the directory cannot be created.
    """
    cache_dir = settings.model_cache_dir
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"Could not create model cache directory {cache_dir}: {exc}",
            details={"model_cache_dir": str(cache_dir)},
        ) from exc

    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    # HF_HOME governs huggingface_hub; the others are read by transformers and
    # older library versions that have not migrated to HF_HOME yet.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    # Avoid a startup stall when the machine is offline but weights are cached.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Windows without Developer Mode cannot create symlinks, so the hub falls
    # back to copying. That works fine here and the warning is pure noise.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    return cache_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build, validate and cache the application settings.

    Returns:
        The process-wide :class:`Settings` singleton.

    Raises:
        ConfigurationError: If any environment value fails validation.
    """
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid configuration: {exc.error_count()} setting(s) failed validation.",
            details={
                "errors": [
                    {"field": ".".join(str(p) for p in err["loc"]), "problem": err["msg"]}
                    for err in exc.errors()
                ]
            },
        ) from exc

    settings.validate_runtime_requirements()
    configure_model_cache(settings)
    return settings
