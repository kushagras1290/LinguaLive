"""Shared pytest fixtures.

Fixtures that would load real models are deliberately avoided: the unit suite
must run in seconds without weights on disk, so engine behaviour is exercised
through fakes and only the integration tests touch real inference.
"""

from __future__ import annotations

import io
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# Make `app` importable when pytest is run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.engines.base import (  # noqa: E402
    ASREngine,
    EngineSet,
    MTEngine,
    SpeechResult,
    TranscriptionResult,
    TranslationResult,
    TTSEngine,
)

SAMPLE_RATE = 16_000


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointed at a throwaway cache directory."""
    cache = tmp_path_factory.mktemp("model-cache")
    return Settings(
        environment="test",
        model_cache_dir=cache,
        log_level="WARNING",
        max_audio_seconds=60.0,
        max_text_chars=1_000,
    )


def make_tone(
    seconds: float = 1.0,
    frequency: float = 220.0,
    amplitude: float = 0.4,
    sample_rate: int = SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Generate a sine tone as float32 samples in [-1, 1]."""
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (amplitude * np.sin(2 * math.pi * frequency * t)).astype(np.float32)


def make_wav_bytes(
    audio: NDArray[np.float32], sample_rate: int = SAMPLE_RATE
) -> bytes:
    """Encode float32 audio as a 16-bit mono WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture
def tone() -> NDArray[np.float32]:
    """One second of a 220 Hz sine wave."""
    return make_tone()


@pytest.fixture
def wav_bytes(tone: NDArray[np.float32]) -> bytes:
    """A valid WAV file containing a tone."""
    return make_wav_bytes(tone)


@pytest.fixture
def silence_wav_bytes() -> bytes:
    """A valid WAV file containing digital silence."""
    return make_wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeASR(ASREngine):
    """Returns a fixed transcript and records how it was called."""

    name = "fake_asr"

    def __init__(self, text: str = "hello world") -> None:
        super().__init__()
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def _load(self) -> None:
        return None

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        # Recorded so tests can assert the audio contract the real engine relies
        # on: float32, mono, within [-1, 1].
        self.calls.append(
            {
                "dtype": audio.dtype,
                "ndim": audio.ndim,
                "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
                "sample_rate": sample_rate,
                "language": language,
            }
        )
        return TranscriptionResult(
            text=self.text,
            language=language or "en",
            duration=audio.size / sample_rate if sample_rate else 0.0,
            engine=self.name,
        )


class FakeMT(MTEngine):
    """Echoes the input with a marker so translation is observable."""

    name = "fake_mt"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def _load(self) -> None:
        return None

    def translate(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        num_beams: int | None = None,
    ) -> TranslationResult:
        self.calls.append(
            {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "num_beams": num_beams,
            }
        )
        translated = text if source_lang == target_lang else f"[{target_lang}] {text}"
        return TranslationResult(
            text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=self.name,
        )


class FakeTTS(TTSEngine):
    """Returns fixed bytes, or raises when configured to fail."""

    name = "fake_tts"

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        super().__init__()
        self.fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    def _load(self) -> None:
        return None

    def synthesise(self, text: str, *, language: str) -> SpeechResult:
        self.calls.append({"text": text, "language": language})
        if self.fail_with is not None:
            raise self.fail_with
        return SpeechResult(
            audio=b"FAKE-AUDIO-BYTES",
            mime_type="audio/wav",
            language=language,
            engine=self.name,
        )


@pytest.fixture
def fake_engines() -> EngineSet:
    """An engine set backed entirely by fakes."""
    return EngineSet(asr=FakeASR(), mt=FakeMT(), tts=FakeTTS())


@pytest.fixture
def app(settings: Settings, fake_engines: EngineSet, monkeypatch: pytest.MonkeyPatch):
    """A Flask app wired to fake engines, so no weights are loaded."""
    import app as app_package

    monkeypatch.setattr(app_package, "build_engines", lambda _: fake_engines)
    flask_app = app_package.create_app(settings)
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture
def client(app):  # noqa: ANN001 - Flask's app type is awkward to spell here
    """A test client for the fake-engine app."""
    return app.test_client()
