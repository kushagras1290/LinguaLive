"""HTTP API routes.

All endpoints live under ``/api`` and speak JSON. The contract is defined in one
place so that clients and server cannot drift apart on paths or shapes.

Design notes:

* Uploads stream to a per-request buffer instead of a fixed ``temp_audio.wav``,
  so concurrent requests cannot overwrite each other's audio.
* Synthesised audio is returned from memory, so nothing is left on disk.
* Bodies accept JSON *or* form encoding, so the client is free to use either.
"""

from __future__ import annotations

import time
from typing import Any, Final

from flask import Blueprint, Response, current_app, jsonify, request

from app.api.schemas import (
    PipelineOptions,
    SpeakRequest,
    TranscribeOptions,
    TranslateRequest,
    parse_request,
)
from app.audio import condition_audio, decode_audio
from app.engines.asr_faster_whisper import WHISPER_SAMPLE_RATE
from app.engines.base import EngineSet
from app.errors import (
    PayloadTooLargeError,
    RequestValidationError,
    TranslationAppError,
)
from app.languages import Capability, all_languages, require_capability, require_language
from app.logging_conf import get_logger

__all__ = ["api_blueprint"]

_LOG = get_logger(__name__)

api_blueprint = Blueprint("api", __name__, url_prefix="/api")

#: Field name the client must use for the uploaded audio.
_AUDIO_FIELD: Final[str] = "audio"


def _engines() -> EngineSet:
    """Return the engine set bound to the running application."""
    return current_app.extensions["engines"]


def _settings() -> Any:
    """Return the settings bound to the running application."""
    return current_app.extensions["settings"]


def _request_payload() -> dict[str, Any]:
    """Read the request body as a dictionary, accepting JSON or form encoding.

    Returns:
        The decoded payload, empty when the body is absent.

    Raises:
        RequestValidationError: If the body claims to be JSON but is malformed.
    """
    if request.is_json:
        try:
            body = request.get_json(silent=False)
        except Exception as exc:  # werkzeug raises BadRequest on malformed JSON
            raise RequestValidationError("The request body is not valid JSON.") from exc
        if body is None:
            return {}
        if not isinstance(body, dict):
            raise RequestValidationError("The request body must be a JSON object.")
        return body
    return dict(request.form)


def _read_audio_upload() -> bytes:
    """Read the uploaded audio file into memory.

    Returns:
        The raw uploaded bytes.

    Raises:
        RequestValidationError: If no audio part is present.
        PayloadTooLargeError: If the upload exceeds the configured limit.
    """
    if _AUDIO_FIELD not in request.files:
        raise RequestValidationError(
            f"No audio file was provided. Send it as multipart field {_AUDIO_FIELD!r}."
        )

    file_storage = request.files[_AUDIO_FIELD]
    if not file_storage.filename and not file_storage.content_length:
        raise RequestValidationError("The uploaded audio file is empty.")

    data = file_storage.read()
    limit = _settings().max_upload_bytes
    if len(data) > limit:
        raise PayloadTooLargeError(
            f"The audio is {len(data) / 1024 / 1024:.1f} MB, over the "
            f"{limit / 1024 / 1024:.0f} MB limit.",
            details={"bytes": len(data), "limit_bytes": limit},
        )
    if not data:
        raise RequestValidationError("The uploaded audio file is empty.")
    return data


def _prepare_audio(data: bytes) -> Any:
    """Decode and condition uploaded audio for the ASR engine.

    Returns:
        Conditioned float32 samples at :data:`WHISPER_SAMPLE_RATE`.
    """
    settings = _settings()
    decoded = decode_audio(
        data,
        target_sample_rate=WHISPER_SAMPLE_RATE,
        max_seconds=settings.max_audio_seconds,
    )
    return condition_audio(decoded, WHISPER_SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@api_blueprint.get("/health")
def health() -> Response:
    """Report service health, engine state and resident memory.

    Memory is included so the hosting decision can be made from a measured
    number rather than an estimate.
    """
    engines = _engines()
    settings = _settings()

    payload: dict[str, Any] = {
        "status": "ok",
        "environment": str(settings.environment),
        "engines": engines.describe(),
        "warnings": engines.warnings,
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_audio_seconds": settings.max_audio_seconds,
            "max_text_chars": settings.max_text_chars,
        },
    }

    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process()
        payload["memory"] = {
            "rss_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        }
    except ImportError:  # pragma: no cover - psutil is declared
        payload["memory"] = {"rss_mb": None}

    return jsonify(payload)


@api_blueprint.get("/languages")
def languages() -> Response:
    """List every supported language with its per-capability flags.

    The client uses ``can_transcribe`` and ``can_speak`` to disable the microphone
    and speaker controls per language, which is what prevents a user from
    requesting an operation the models cannot perform.
    """
    entries = [language.to_dict() for language in all_languages()]
    return jsonify(
        {
            "languages": entries,
            "counts": {
                "total": len(entries),
                "transcribable": sum(1 for e in entries if e["can_transcribe"]),
                "speakable": sum(1 for e in entries if e["can_speak"]),
            },
        }
    )


# --------------------------------------------------------------------------- #
# Core capabilities
# --------------------------------------------------------------------------- #
@api_blueprint.post("/transcribe")
def transcribe() -> Response:
    """Transcribe an uploaded audio file.

    Accepts any container FFmpeg can decode, including the WebM/Opus that
    ``MediaRecorder`` produces, which a ``wave.open`` path cannot handle.
    """
    started = time.perf_counter()
    data = _read_audio_upload()
    options = parse_request(TranscribeOptions, dict(request.form))

    whisper_lang: str | None = None
    if options.source_lang is not None:
        whisper_lang = require_capability(
            options.source_lang, Capability.TRANSCRIBE
        ).whisper

    audio = _prepare_audio(data)
    result = _engines().asr.transcribe(
        audio, sample_rate=WHISPER_SAMPLE_RATE, language=whisper_lang
    )

    _LOG.info(
        "Transcribed upload",
        extra={
            "bytes": len(data),
            "characters": len(result.text),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return jsonify(result.to_dict())


@api_blueprint.post("/translate")
def translate() -> Response:
    """Translate text between two languages."""
    started = time.perf_counter()
    payload = parse_request(TranslateRequest, _request_payload())

    settings = _settings()
    if len(payload.text) > settings.max_text_chars:
        raise RequestValidationError(
            f"Text is {len(payload.text)} characters, over the "
            f"{settings.max_text_chars} character limit.",
            details={"characters": len(payload.text), "limit": settings.max_text_chars},
        )

    result = _engines().mt.translate(
        payload.text,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
    )

    _LOG.info(
        "Translated text",
        extra={
            "source_lang": payload.source_lang,
            "target_lang": payload.target_lang,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return jsonify(result.to_dict())


@api_blueprint.post("/speak")
def speak() -> Response:
    """Synthesise speech and return the audio inline.

    The response is the audio itself rather than an attachment, so the browser
    can play it from a blob URL without a download prompt.
    """
    started = time.perf_counter()
    payload = parse_request(SpeakRequest, _request_payload())

    result = _engines().tts.synthesise(payload.text, language=payload.lang)

    _LOG.info(
        "Synthesised speech",
        extra={
            "language": payload.lang,
            "engine": result.engine,
            "bytes": result.size_bytes,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    )

    response = Response(result.audio, mimetype=result.mime_type)
    response.headers["Content-Length"] = str(result.size_bytes)
    response.headers["Cache-Control"] = "no-store"
    # Lets the client show which backend served the audio.
    response.headers["X-TTS-Engine"] = result.engine
    return response


@api_blueprint.post("/pipeline")
def pipeline() -> Response:
    """Run transcribe, translate and optionally synthesise in one request.

    Saves two network round trips on the common "speak and hear it back" flow.
    Audio is returned base64-encoded inside the JSON body so the whole result
    arrives atomically.
    """
    import base64  # noqa: PLC0415 - only needed on this path

    started = time.perf_counter()
    data = _read_audio_upload()
    options = parse_request(PipelineOptions, dict(request.form))

    whisper_lang: str | None = None
    if options.source_lang is not None:
        whisper_lang = require_capability(
            options.source_lang, Capability.TRANSCRIBE
        ).whisper

    audio = _prepare_audio(data)
    engines = _engines()
    transcription = engines.asr.transcribe(
        audio, sample_rate=WHISPER_SAMPLE_RATE, language=whisper_lang
    )

    response: dict[str, Any] = {"transcription": transcription.to_dict()}

    if transcription.is_empty:
        response["translation"] = None
        response["speech"] = None
        response["note"] = "No speech was recognised in the audio."
        return jsonify(response)

    # Whisper reports ISO-639-1; map it back to an application code when the
    # caller asked for auto-detection.
    resolved_source = options.source_lang
    if resolved_source is None:
        resolved_source = _application_code_for_whisper(transcription.language)

    translation = engines.mt.translate(
        transcription.text,
        source_lang=resolved_source,
        target_lang=options.target_lang,
    )
    response["translation"] = translation.to_dict()

    if options.speak and translation.text:
        try:
            speech = engines.tts.synthesise(translation.text, language=options.target_lang)
            response["speech"] = {
                "audio_base64": base64.b64encode(speech.audio).decode("ascii"),
                "mime_type": speech.mime_type,
                "engine": speech.engine,
            }
        except TranslationAppError as exc:
            # Synthesis is the least essential stage; a failure here must not
            # discard a good transcription and translation.
            _LOG.warning(
                "Pipeline synthesis failed; returning text only",
                extra={"language": options.target_lang, "error": exc.message},
            )
            response["speech"] = None
            response["speech_error"] = exc.message
    else:
        response["speech"] = None

    _LOG.info(
        "Completed pipeline",
        extra={
            "target_lang": options.target_lang,
            "spoke": response["speech"] is not None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return jsonify(response)


def _application_code_for_whisper(whisper_language: str) -> str:
    """Map a Whisper ISO-639-1 code back to an application language code.

    Whisper and the registry agree for most languages, but a few application
    codes differ (``zh-Hant``, ``jv``). Falls back to English when no match
    exists, which keeps auto-detection from failing the whole request.
    """
    for language in all_languages():
        if language.whisper == whisper_language and language.code == whisper_language:
            return language.code
    for language in all_languages():
        if language.whisper == whisper_language:
            return language.code
    _LOG.warning(
        "Whisper detected a language with no registry entry; defaulting to English",
        extra={"whisper_language": whisper_language},
    )
    return "en"


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@api_blueprint.get("/languages/<code>")
def language_detail(code: str) -> Response:
    """Return the registry entry for one language code."""
    language = require_language(code)
    return jsonify(
        {
            **language.to_dict(),
            "nllb": language.nllb,
            "whisper": language.whisper,
            "gtts": language.gtts,
            "mms": language.mms,
        }
    )
