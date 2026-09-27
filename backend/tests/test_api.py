"""Tests for the HTTP API.

These run against fake engines, so they exercise routing, validation, error
mapping and the request/response contract without loading any weights.

The contract itself matters: a client calling `/api/translate` against a server
serving `/translate` 404s on every request. These tests pin the paths clients
actually use.
"""

from __future__ import annotations

import base64
import io

import pytest

from app.errors import InferenceError, UnsupportedCapabilityError

from .conftest import make_tone, make_wav_bytes


def audio_upload(data: bytes, name: str = "clip.wav") -> dict:
    """Build a multipart payload for the audio endpoints."""
    return {"audio": (io.BytesIO(data), name)}


class TestHealth:
    """`/api/health`."""

    def test_reports_ok(self, client) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200

        body = response.get_json()
        assert body["status"] == "ok"
        assert "engines" in body
        assert "memory" in body

    def test_includes_resident_memory(self, client) -> None:
        """Memory is reported so hosting can be sized from a real number."""
        body = client.get("/api/health").get_json()
        assert body["memory"]["rss_mb"] is None or body["memory"]["rss_mb"] > 0


class TestLanguages:
    """`/api/languages`."""

    def test_lists_every_language(self, client) -> None:
        body = client.get("/api/languages").get_json()
        assert body["counts"]["total"] == 202
        assert len(body["languages"]) == 202

    def test_entries_carry_capability_flags(self, client) -> None:
        """The client disables the mic and speaker from these flags."""
        body = client.get("/api/languages").get_json()
        entry = next(item for item in body["languages"] if item["code"] == "en")
        assert entry["can_transcribe"] is True
        assert entry["can_speak"] is True
        assert entry["rtl"] is False

    def test_detail_endpoint_exposes_external_codes(self, client) -> None:
        body = client.get("/api/languages/hi").get_json()
        assert body["nllb"] == "hin_Deva"
        assert body["whisper"] == "hi"

    def test_unknown_language_returns_400(self, client) -> None:
        response = client.get("/api/languages/klingon")
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "unknown_language"


class TestTranslate:
    """`/api/translate`."""

    def test_translates_json_body(self, client) -> None:
        response = client.post(
            "/api/translate",
            json={"text": "hello", "source_lang": "en", "target_lang": "hi"},
        )
        assert response.status_code == 200
        assert response.get_json()["text"] == "[hi] hello"

    def test_accepts_form_encoding(self, client) -> None:
        """Both encodings work, so the client is free to pick either."""
        response = client.post(
            "/api/translate",
            data={"text": "hello", "source_lang": "en", "target_lang": "hi"},
        )
        assert response.status_code == 200

    def test_rejects_missing_fields(self, client) -> None:
        response = client.post("/api/translate", json={"text": "hello"})
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "validation_error"

    def test_rejects_unknown_language(self, client) -> None:
        response = client.post(
            "/api/translate",
            json={"text": "hello", "source_lang": "en", "target_lang": "xx"},
        )
        assert response.status_code == 400

    def test_rejects_empty_text(self, client) -> None:
        response = client.post(
            "/api/translate",
            json={"text": "", "source_lang": "en", "target_lang": "hi"},
        )
        assert response.status_code == 400

    def test_rejects_unknown_field(self, client) -> None:
        """Unknown keys are rejected so client typos surface immediately."""
        response = client.post(
            "/api/translate",
            json={
                "text": "hello",
                "source_lang": "en",
                "target_lang": "hi",
                "temperature": 0.9,
            },
        )
        assert response.status_code == 400

    def test_rejects_oversized_text(self, client, app) -> None:
        limit = app.extensions["settings"].max_text_chars
        response = client.post(
            "/api/translate",
            json={"text": "a" * (limit + 1), "source_lang": "en", "target_lang": "hi"},
        )
        assert response.status_code == 400

    def test_malformed_json_returns_400(self, client) -> None:
        response = client.post(
            "/api/translate",
            data="{not json",
            content_type="application/json",
        )
        assert response.status_code == 400


class TestTranscribe:
    """`/api/transcribe`."""

    def test_transcribes_upload(self, client, wav_bytes: bytes) -> None:
        response = client.post(
            "/api/transcribe",
            data={**audio_upload(wav_bytes), "source_lang": "en"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert response.get_json()["text"] == "hello world"

    def test_passes_float32_audio_to_engine(
        self, client, app, wav_bytes: bytes
    ) -> None:
        """Engines must receive float32, never raw int16."""
        client.post(
            "/api/transcribe",
            data=audio_upload(wav_bytes),
            content_type="multipart/form-data",
        )

        call = app.extensions["engines"].asr.calls[-1]
        assert call["dtype"].name == "float32"
        assert call["ndim"] == 1
        assert call["peak"] <= 1.0
        assert call["sample_rate"] == 16_000

    def test_auto_detects_when_language_omitted(
        self, client, app, wav_bytes: bytes
    ) -> None:
        client.post(
            "/api/transcribe",
            data=audio_upload(wav_bytes),
            content_type="multipart/form-data",
        )
        assert app.extensions["engines"].asr.calls[-1]["language"] is None

    def test_rejects_missing_audio(self, client) -> None:
        response = client.post("/api/transcribe", data={"source_lang": "en"})
        assert response.status_code == 400

    def test_rejects_silence_with_422(self, client, silence_wav_bytes: bytes) -> None:
        """Silence is a client-side condition, not a server error."""
        response = client.post(
            "/api/transcribe",
            data=audio_upload(silence_wav_bytes),
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        assert response.get_json()["error"]["code"] == "audio_empty"

    def test_rejects_undecodable_upload_with_415(self, client) -> None:
        response = client.post(
            "/api/transcribe",
            data=audio_upload(b"definitely not audio" * 40, "junk.bin"),
            content_type="multipart/form-data",
        )
        assert response.status_code == 415

    def test_rejects_untranscribable_language_with_422(
        self, client, wav_bytes: bytes
    ) -> None:
        """Bhojpuri is translatable but Whisper cannot hear it."""
        response = client.post(
            "/api/transcribe",
            data={**audio_upload(wav_bytes), "source_lang": "bho"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 422
        assert response.get_json()["error"]["code"] == "capability_unsupported"

    def test_rejects_audio_over_duration_limit(self, client, app) -> None:
        limit = app.extensions["settings"].max_audio_seconds
        long_audio = make_wav_bytes(make_tone(seconds=limit + 5))
        response = client.post(
            "/api/transcribe",
            data=audio_upload(long_audio),
            content_type="multipart/form-data",
        )
        assert response.status_code == 413


class TestSpeak:
    """`/api/speak`."""

    def test_returns_audio_inline(self, client) -> None:
        response = client.post("/api/speak", json={"text": "hello", "lang": "en"})
        assert response.status_code == 200
        assert response.mimetype == "audio/wav"
        assert response.data == b"FAKE-AUDIO-BYTES"

    def test_reports_engine_in_header(self, client) -> None:
        response = client.post("/api/speak", json={"text": "hello", "lang": "en"})
        assert response.headers["X-TTS-Engine"] == "fake_tts"

    def test_rejects_unknown_language(self, client) -> None:
        response = client.post("/api/speak", json={"text": "hello", "lang": "xx"})
        assert response.status_code == 400

    def test_engine_failure_maps_to_502(self, app, client) -> None:
        app.extensions["engines"].tts.fail_with = InferenceError("synthesis exploded")
        response = client.post("/api/speak", json={"text": "hello", "lang": "en"})
        assert response.status_code == 502

    def test_unsupported_language_maps_to_422(self, app, client) -> None:
        """Must be a typed 422, never an unhandled 500."""
        app.extensions["engines"].tts.fail_with = UnsupportedCapabilityError(
            "no voice available"
        )
        response = client.post("/api/speak", json={"text": "hello", "lang": "bho"})
        assert response.status_code == 422


class TestPipeline:
    """`/api/pipeline`, the combined transcribe/translate/speak route."""

    def test_runs_all_three_stages(self, client, wav_bytes: bytes) -> None:
        response = client.post(
            "/api/pipeline",
            data={
                **audio_upload(wav_bytes),
                "source_lang": "en",
                "target_lang": "hi",
                "speak": "true",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

        body = response.get_json()
        assert body["transcription"]["text"] == "hello world"
        assert body["translation"]["text"] == "[hi] hello world"
        assert base64.b64decode(body["speech"]["audio_base64"]) == b"FAKE-AUDIO-BYTES"

    def test_skips_speech_when_not_requested(self, client, wav_bytes: bytes) -> None:
        response = client.post(
            "/api/pipeline",
            data={
                **audio_upload(wav_bytes),
                "target_lang": "hi",
                "speak": "false",
            },
            content_type="multipart/form-data",
        )
        assert response.get_json()["speech"] is None

    def test_survives_synthesis_failure(self, app, client, wav_bytes: bytes) -> None:
        """A TTS failure must not discard a good transcription and translation."""
        app.extensions["engines"].tts.fail_with = InferenceError("no voice")

        response = client.post(
            "/api/pipeline",
            data={
                **audio_upload(wav_bytes),
                "target_lang": "hi",
                "speak": "true",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

        body = response.get_json()
        assert body["translation"]["text"] == "[hi] hello world"
        assert body["speech"] is None
        assert "speech_error" in body

    def test_empty_transcription_short_circuits(
        self, app, client, wav_bytes: bytes
    ) -> None:
        app.extensions["engines"].asr.text = "   "

        response = client.post(
            "/api/pipeline",
            data={**audio_upload(wav_bytes), "target_lang": "hi"},
            content_type="multipart/form-data",
        )
        body = response.get_json()
        assert body["translation"] is None
        assert "note" in body


class TestErrorHandling:
    """Cross-cutting error behaviour."""

    def test_unknown_route_returns_json(self, client) -> None:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.get_json()["error"]["code"] == "not_found"

    def test_wrong_method_returns_json(self, client) -> None:
        response = client.get("/api/translate")
        assert response.status_code == 405
        assert response.get_json()["error"]["code"] == "method_not_allowed"

    def test_responses_carry_request_id(self, client) -> None:
        response = client.get("/api/health")
        assert response.headers.get("X-Request-ID")

    def test_client_request_id_is_echoed(self, client) -> None:
        """Lets a user quote an id from their console in a bug report."""
        response = client.get("/api/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"


class TestUnicode:
    """Non-Latin scripts must survive the round trip.

    Most of the 202 supported languages are not Latin-script, so escaping or
    mangling them would break the majority of the product.
    """

    @pytest.mark.parametrize(
        "text",
        ["नमस्ते दुनिया", "مرحبا بالعالم", "你好世界", "Γειά σου Κόσμε", "🌍 emoji"],
    )
    def test_round_trips_unescaped(self, client, text: str) -> None:
        response = client.post(
            "/api/translate",
            json={"text": text, "source_lang": "en", "target_lang": "en"},
        )
        assert response.status_code == 200
        assert response.get_json()["text"] == text
