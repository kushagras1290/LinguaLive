"""Engine factory.

Maps configuration values onto concrete engine implementations. This is the only
module that knows every engine class, so adding a backend means adding one entry
here rather than touching route or streaming code.
"""

from __future__ import annotations

from app.config import ASREngineName, MTEngineName, Settings, TTSEngineName
from app.engines.base import ASREngine, EngineSet, MTEngine, TTSEngine
from app.errors import ConfigurationError
from app.logging_conf import get_logger

__all__ = ["build_engines"]

_LOG = get_logger(__name__)


def _build_asr(settings: Settings) -> ASREngine:
    """Construct the configured speech-recognition engine."""
    match settings.engine_asr:
        case ASREngineName.FASTER_WHISPER:
            from app.engines.asr_faster_whisper import FasterWhisperEngine  # noqa: PLC0415

            return FasterWhisperEngine(settings)
        case ASREngineName.HF_INFERENCE:
            from app.engines.remote.hf_inference import HFInferenceASREngine  # noqa: PLC0415

            return HFInferenceASREngine(settings)
        case _:  # pragma: no cover - StrEnum makes this unreachable
            raise ConfigurationError(
                f"Unsupported ASR engine {settings.engine_asr!r}.",
                details={"engine_asr": str(settings.engine_asr)},
            )


def _build_mt(settings: Settings) -> MTEngine:
    """Construct the configured translation engine."""
    match settings.engine_mt:
        case MTEngineName.NLLB_LOCAL:
            from app.engines.mt_nllb_local import NLLBLocalEngine  # noqa: PLC0415

            return NLLBLocalEngine(settings)
        case MTEngineName.HF_INFERENCE:
            from app.engines.remote.hf_inference import HFInferenceMTEngine  # noqa: PLC0415

            return HFInferenceMTEngine(settings)
        case _:  # pragma: no cover
            raise ConfigurationError(
                f"Unsupported MT engine {settings.engine_mt!r}.",
                details={"engine_mt": str(settings.engine_mt)},
            )


def _build_single_tts(name: str, settings: Settings) -> TTSEngine:
    """Construct one concrete TTS backend by name.

    Raises:
        ConfigurationError: If ``name`` has no implementation.
    """
    match name:
        case TTSEngineName.MMS:
            from app.engines.tts_mms import MMSTTSEngine  # noqa: PLC0415

            return MMSTTSEngine(settings)
        case TTSEngineName.GTTS:
            from app.engines.tts_gtts import GTTSEngine  # noqa: PLC0415

            return GTTSEngine(settings)
        case TTSEngineName.FORMANT:
            from app.engines.tts_formant import FormantTTSEngine  # noqa: PLC0415

            return FormantTTSEngine(settings)
        case _:
            raise ConfigurationError(
                f"Unsupported TTS backend {name!r}. Valid backends: "
                f"{', '.join(sorted(n for n in TTSEngineName if n != TTSEngineName.CHAIN))}.",
                details={"backend": name},
            )


def _build_tts(settings: Settings) -> TTSEngine:
    """Construct the configured synthesis engine, or the fallback chain."""
    if settings.engine_tts is not TTSEngineName.CHAIN:
        return _build_single_tts(settings.engine_tts, settings)

    from app.engines.tts_chain import ChainTTSEngine  # noqa: PLC0415

    backends = tuple(
        _build_single_tts(name, settings) for name in settings.tts_chain_list
    )
    return ChainTTSEngine(settings, backends)


def build_engines(settings: Settings) -> EngineSet:
    """Build the engine set described by ``settings``.

    Engines are constructed but not loaded; weights are pulled in on first use
    unless :attr:`Settings.eager_load_models` is set.

    Args:
        settings: Validated application settings.

    Returns:
        The assembled :class:`~app.engines.base.EngineSet`.

    Raises:
        ConfigurationError: If an engine name has no implementation.
    """
    engines = EngineSet(
        asr=_build_asr(settings),
        mt=_build_mt(settings),
        tts=_build_tts(settings),
    )

    # Surfaced through /api/health so degraded capability is visible rather than
    # discovered when a user clicks the speaker icon.
    if settings.engine_tts is TTSEngineName.GTTS:
        engines.warnings.append(
            "gTTS alone covers 68 of 202 languages. Set ENGINE_TTS=chain to fall "
            "back to MMS and the built-in formant synthesiser for the rest."
        )
    elif settings.engine_tts is TTSEngineName.CHAIN:
        if "formant" not in settings.tts_chain_list:
            engines.warnings.append(
                "The TTS chain does not end with 'formant', so synthesis can "
                "fail outright for languages with no neural voice."
            )

    _LOG.info(
        "Engines configured",
        extra={
            "asr": engines.asr.name,
            "mt": engines.mt.name,
            "tts": engines.tts.name,
            "eager_load": settings.eager_load_models,
        },
    )
    return engines
