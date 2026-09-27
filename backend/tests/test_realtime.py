"""Tests for the live streaming path.

The session and VAD are transport-agnostic by design, so they are tested here
without a browser or a socket.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.engines.base import EngineSet
from app.realtime.session import EventType, SessionConfig, StreamSession
from app.realtime.vad import SpeechState, VoiceActivityDetector

from .conftest import make_tone

STREAM_RATE = 16_000


def silence(seconds: float) -> np.ndarray:
    """Digital silence of a given duration."""
    return np.zeros(int(seconds * STREAM_RATE), dtype=np.float32)


class TestVoiceActivityDetector:
    """Energy-based speech detection."""

    @pytest.fixture
    def detector(self) -> VoiceActivityDetector:
        return VoiceActivityDetector(
            sample_rate=STREAM_RATE, release_threshold=0.015, silence_seconds=0.3
        )

    def test_starts_in_silence(self, detector: VoiceActivityDetector) -> None:
        assert detector.state is SpeechState.SILENCE

    def test_detects_speech_onset(self, detector: VoiceActivityDetector) -> None:
        frames = detector.process(make_tone(seconds=0.5, amplitude=0.4))
        assert any(frame.is_onset for frame in frames)
        assert detector.state is SpeechState.SPEECH

    def test_ignores_silence(self, detector: VoiceActivityDetector) -> None:
        frames = detector.process(silence(0.5))
        assert not any(frame.is_onset for frame in frames)
        assert detector.state is SpeechState.SILENCE

    def test_detects_offset_after_silence(
        self, detector: VoiceActivityDetector
    ) -> None:
        detector.process(make_tone(seconds=0.5, amplitude=0.4))
        frames = detector.process(silence(1.0))
        assert any(frame.is_offset for frame in frames)
        assert detector.state is SpeechState.SILENCE

    def test_brief_gap_does_not_end_utterance(
        self, detector: VoiceActivityDetector
    ) -> None:
        """Inter-word gaps must not split one utterance into many."""
        detector.process(make_tone(seconds=0.4, amplitude=0.4))
        frames = detector.process(silence(0.1))  # shorter than silence_seconds
        assert not any(frame.is_offset for frame in frames)
        assert detector.state is SpeechState.SPEECH

    def test_buffers_partial_frames(self, detector: VoiceActivityDetector) -> None:
        """Input need not align to frame boundaries."""
        tiny = make_tone(seconds=0.001, amplitude=0.4)
        assert detector.process(tiny) == []

    def test_frames_are_emitted_in_order(
        self, detector: VoiceActivityDetector
    ) -> None:
        frames = detector.process(make_tone(seconds=1.0, amplitude=0.4))
        expected = STREAM_RATE // detector.frame_length
        assert len(frames) == pytest.approx(expected, abs=2)

    def test_reset_clears_state(self, detector: VoiceActivityDetector) -> None:
        detector.process(make_tone(seconds=0.5, amplitude=0.4))
        detector.reset()
        assert detector.state is SpeechState.SILENCE

    def test_adapts_to_a_noisy_room(self) -> None:
        """Constant low-level noise must not read as continuous speech."""
        detector = VoiceActivityDetector(
            sample_rate=STREAM_RATE, release_threshold=0.01, silence_seconds=0.3
        )
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 0.02, STREAM_RATE * 2)).astype(np.float32)
        detector.process(noise)
        # The noise floor should have risen enough that the hiss alone no longer
        # trips the onset threshold.
        frames = detector.process((rng.normal(0, 0.02, STREAM_RATE)).astype(np.float32))
        assert not all(frame.state is SpeechState.SPEECH for frame in frames)


class TestStreamSession:
    """Session state machine."""

    @pytest.fixture
    def config(self) -> SessionConfig:
        return SessionConfig(source_lang="en", target_lang="hi", speak=False)

    @pytest.fixture
    def session(
        self, settings: Settings, fake_engines: EngineSet, config: SessionConfig
    ) -> StreamSession:
        return StreamSession(settings=settings, engines=fake_engines, config=config)

    def test_ready_event_reports_parameters(self, session: StreamSession) -> None:
        event = session.ready_event()
        assert event.type is EventType.READY
        assert event.data["sample_rate"] == 16_000
        assert event.data["target_lang"] == "hi"

    def test_empty_audio_produces_no_events(self, session: StreamSession) -> None:
        assert session.push(np.zeros(0, dtype=np.float32)) == []

    def test_emits_speech_start(self, session: StreamSession) -> None:
        events = session.push(make_tone(seconds=0.5, amplitude=0.4))
        assert any(event.type is EventType.SPEECH_START for event in events)

    def test_emits_level_events_for_visuals(self, session: StreamSession) -> None:
        events = session.push(make_tone(seconds=0.2, amplitude=0.4))
        assert any(event.type is EventType.LEVEL for event in events)

    def test_finalises_after_silence(self, session: StreamSession) -> None:
        session.push(make_tone(seconds=1.5, amplitude=0.4))
        events = session.push(silence(1.5))

        finals = [event for event in events if event.type is EventType.FINAL]
        assert len(finals) == 1
        assert finals[0].data["text"] == "hello world"
        assert finals[0].data["translation"] == "[hi] hello world"

    def test_final_uses_configured_stream_beam_width(
        self, session: StreamSession, fake_engines: EngineSet
    ) -> None:
        session.push(make_tone(seconds=1.5, amplitude=0.4))
        session.push(silence(1.5))
        assert fake_engines.mt.calls[-1]["num_beams"] == 4

    def test_flush_emits_buffered_speech(self, session: StreamSession) -> None:
        """A mid-sentence disconnect must not discard the last utterance."""
        session.push(make_tone(seconds=1.5, amplitude=0.4))
        events = session.flush()
        assert any(event.type is EventType.FINAL for event in events)

    def test_flush_with_no_audio_is_empty(self, session: StreamSession) -> None:
        assert session.flush() == []

    def test_utterance_index_advances(self, session: StreamSession) -> None:
        session.push(make_tone(seconds=1.5, amplitude=0.4))
        first = session.push(silence(1.5))
        session.push(make_tone(seconds=1.5, amplitude=0.4))
        second = session.push(silence(1.5))

        first_index = next(e for e in first if e.type is EventType.FINAL).data["utterance"]
        second_index = next(e for e in second if e.type is EventType.FINAL).data["utterance"]
        assert second_index > first_index

    def test_stats_track_usage(self, session: StreamSession) -> None:
        session.push(make_tone(seconds=1.0, amplitude=0.4))
        stats = session.stats()
        assert stats["audio_seconds"] == pytest.approx(1.0, abs=0.1)

    def test_rejects_untranscribable_source_language(
        self, settings: Settings, fake_engines: EngineSet
    ) -> None:
        """Fail at session start, not midway through a stream."""
        from app.errors import UnsupportedCapabilityError

        with pytest.raises(UnsupportedCapabilityError):
            StreamSession(
                settings=settings,
                engines=fake_engines,
                config=SessionConfig(source_lang="bho", target_lang="en"),
            )

    def test_rejects_unknown_target_language(
        self, settings: Settings, fake_engines: EngineSet
    ) -> None:
        from app.errors import UnknownLanguageError

        with pytest.raises(UnknownLanguageError):
            StreamSession(
                settings=settings,
                engines=fake_engines,
                config=SessionConfig(source_lang="en", target_lang="xx"),
            )

    def test_auto_detect_source_is_allowed(
        self, settings: Settings, fake_engines: EngineSet
    ) -> None:
        session = StreamSession(
            settings=settings,
            engines=fake_engines,
            config=SessionConfig(source_lang=None, target_lang="hi"),
        )
        assert session.ready_event().data["source_lang"] is None

    def test_idle_buffer_stays_bounded(self, session: StreamSession) -> None:
        """A quiet connection must not grow the buffer without limit."""
        for _ in range(20):
            session.push(silence(1.0))
        assert session.stats()["audio_seconds"] == pytest.approx(20.0, abs=0.5)


class TestSessionConfigParsing:
    """Handshake frame validation."""

    def test_rejects_non_json(self) -> None:
        from app.errors import RequestValidationError
        from app.realtime.ws import _parse_config

        with pytest.raises(RequestValidationError):
            _parse_config("not json at all")

    def test_rejects_wrong_frame_type(self) -> None:
        from app.errors import RequestValidationError
        from app.realtime.ws import _parse_config

        with pytest.raises(RequestValidationError):
            _parse_config('{"type": "audio"}')

    def test_requires_target_language(self) -> None:
        from app.errors import RequestValidationError
        from app.realtime.ws import _parse_config

        with pytest.raises(RequestValidationError):
            _parse_config('{"type": "config", "source_lang": "en"}')

    def test_accepts_valid_frame(self) -> None:
        from app.realtime.ws import _parse_config

        config = _parse_config(
            '{"type": "config", "source_lang": "en", "target_lang": "hi", "speak": true}'
        )
        assert config.source_lang == "en"
        assert config.target_lang == "hi"
        assert config.speak is True

    @pytest.mark.parametrize("value", ['"auto"', '""', "null"])
    def test_auto_detect_variants_become_none(self, value: str) -> None:
        from app.realtime.ws import _parse_config

        config = _parse_config(
            f'{{"type": "config", "source_lang": {value}, "target_lang": "hi"}}'
        )
        assert config.source_lang is None
