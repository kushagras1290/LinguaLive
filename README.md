# LinguaLive

Real-time speech-to-speech translation backend covering **202 languages**,
running entirely on self-hosted models. Audio in one language goes in; a
transcript, a translation and synthesised speech in another language come out —
either as a single request or as live captions over a WebSocket.

Built on Meta's **NLLB-200** for translation, OpenAI's **Whisper** (via
CTranslate2) for speech recognition, and a layered text-to-speech chain that
falls back from Meta's **MMS-TTS** to **gTTS** to a from-scratch formant
synthesiser, so speech output never fails outright.

This repository contains the backend service only.

---

## At a glance

| | |
|---|---|
| Translation | 202 languages (NLLB-200 distilled 600M, int8-quantised on CPU) |
| Speech input | 106 languages (Whisper `small`, int8) |
| Speech output | All 202 languages via a three-tier fallback chain |
| Batch latency | ~7 s speech → speech on an 8-thread CPU, no GPU |
| Live mode | Raw 16 kHz PCM over WebSocket, server-side VAD, partial + final captions |
| Stack | Python 3.12, Flask, flask-sock, pydantic, PyAV, gunicorn + gevent, Docker |
| Tests | 236 unit tests (~8 s, no weights) + 13 opt-in integration tests on real models |

---

## Architecture

```
            ┌──────────────────────────── Flask app ────────────────────────────┐
            │                                                                    │
  HTTP ───▶ │  api/routes.py ──▶ api/schemas.py (pydantic, strict)               │
            │        │                                                           │
  WS   ───▶ │  realtime/ws.py ──▶ realtime/session.py ──▶ realtime/vad.py        │
            │        │                     │                                     │
            │        ▼                     ▼                                     │
            │  ┌────────────── engines/registry.py ──────────────┐               │
            │  │  ASREngine        MTEngine          TTSEngine    │               │
            │  │  faster-whisper   NLLB (local)      chain:       │               │
            │  │  | HF Inference   | HF Inference     MMS → gTTS  │               │
            │  │                                      → formant   │               │
            │  └──────────────────────────────────────────────────┘               │
            │        ▲                                   ▲                        │
            │  audio.py (PyAV decode → float32 [-1,1])   tts/ (normalise,         │
            │  languages.py (202-language registry)           translit, synth)    │
            └────────────────────────────────────────────────────────────────────┘
```

```
backend/app/
  config.py          env-validated settings; refuses to start on bad config
  errors.py          typed exception hierarchy, each mapped to an HTTP status
  logging_conf.py    console or single-line JSON logs with request correlation ids
  languages.py       the 202-language registry — single source of truth
  audio.py           decoding and conditioning (any container → float32 mono)
  api/               routes and strict request schemas
  realtime/          WebSocket streaming: VAD, per-connection session state machine
  engines/
    base.py          ASREngine · MTEngine · TTSEngine contracts
    registry.py      builds the configured engine set
    asr_faster_whisper.py   local Whisper via CTranslate2
    mt_nllb_local.py        local NLLB via transformers, int8-quantised
    tts_mms.py / tts_gtts.py / tts_formant.py / tts_chain.py
    remote/          Hugging Face Inference API adapters
  tts/
    normalise.py     numbers, currency, abbreviations, URLs → spoken form
    translit.py      11 writing systems → Latin
    synth.py         Klatt-style formant synthesiser
backend/scripts/
  download_models.py pre-fetch weights into the model cache
  smoke_test.py      end-to-end checks against a running server, WebSocket included
```

### Engine abstraction

Every model-backed capability sits behind an abstract contract in
`engines/base.py`. Nothing outside `app.engines` imports `torch`,
`transformers`, `faster_whisper` or `gtts`. That makes *where* inference runs a
configuration choice rather than a rewrite:

```bash
ENGINE_ASR=hf_inference   # call a hosted Whisper instead of loading it locally
ENGINE_MT=hf_inference
TTS_CHAIN=gtts,formant    # skip neural voices entirely
```

### Capability-aware language registry

NLLB covers 202 languages, Whisper 100 and gTTS 68, and they do not overlap
neatly. Each registry entry declares its capabilities and external model codes
explicitly (gTTS spells Hebrew `iw`, Santali is `sat_Olck`, not `sat_Beng`),
and the registry rejects duplicate codes at import time. A request for something
a model cannot do — transcribing Bhojpuri, for example — returns a typed `422`,
never a `500`.

### Speech output that cannot fail

The TTS chain tries backends in order until one succeeds:

| Backend | Languages | Notes |
|---|---|---|
| **MMS-TTS** (neural VITS) | ~187 | ~145 MB per voice, loaded and cached on demand |
| **gTTS** | 68 | Covers the CJK languages MMS omits |
| **Formant synthesiser** | any | No weights, no network, no API key |

The last tier is a Klatt-style source-filter synthesiser written for this
project: glottal pulse generation, three cascaded formant resonators, a pitch
contour with declination and phrase-final fall, and rule-based
grapheme-to-phoneme conversion. It runs at ~17× real time. A transliteration
layer maps Devanagari, Cyrillic, Greek, Arabic, Hebrew, kana, Thai, Tibetan,
Myanmar, Ol Chiki and Tifinagh onto Latin so it can pronounce languages that
have no neural voice anywhere (Tibetan, Santali, Shan, Tamazight). Han
ideographs are deliberately not mapped — without a per-language dictionary a
character carries no reading — so those report honestly that no voice exists.

### Live streaming

Clients send one JSON configuration frame, then raw little-endian Int16 PCM at
16 kHz. Raw PCM rather than `MediaRecorder` chunks is deliberate: only the first
WebM chunk carries the container header, so later chunks cannot be decoded on
their own.

Server-side, an energy-based VAD with hysteresis and an adaptive noise floor
finds utterance boundaries. Partials are transcribe-only over a rolling window;
translation runs once at the boundary, because NLLB output churns badly on
half-finished sentences. Sessions are capped in duration and buffer size.

### Operational details

- Every failure — validation, capability, inference, timeout, 404, 405 — uses one
  JSON envelope: `{"error": {"code", "message", "details"}}`.
- Request ids are adopted from the client or minted, logged, and echoed back.
- Uploads are decoded in memory; nothing is written to shared temp paths.
- Upload size, audio duration, text length and inference time are all bounded.
- One gunicorn worker with threads: models live in process memory, so extra
  workers multiply RSS rather than throughput.
- Multi-stage Docker image, non-root user, weights on a mounted volume rather
  than baked into the image, health-checked on `/api/health`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status, engine state, resident memory |
| `GET` | `/api/languages` | All 202 languages with per-capability flags |
| `GET` | `/api/languages/<code>` | One language, including external model codes |
| `POST` | `/api/transcribe` | Audio → text (multipart `audio`) |
| `POST` | `/api/translate` | Text → text (JSON or form) |
| `POST` | `/api/speak` | Text → audio |
| `POST` | `/api/pipeline` | Audio → transcription + translation + audio |
| `WS` | `/ws/stream` | Live streaming captions |

```bash
curl -X POST localhost:5000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Good morning","source_lang":"en","target_lang":"hi"}'
# {"text":"सुप्रभात", ...}
```

```json
{ "code": "bho", "name": "Bhojpuri", "native_name": "भोजपुरी", "script": "Deva",
  "can_translate": true, "can_transcribe": false, "can_speak": true,
  "has_neural_voice": true, "rtl": false }
```

---

## Performance

Measured on an 8-thread CPU, no GPU:

| Operation | Time |
|---|---|
| Transcribe 6.6 s of speech (Whisper small, int8) | ~6 s |
| Translate two sentences (NLLB, int8, beam 4) | **2.3 s** |
| Translate, unquantised fp32 | 11.1 s |
| Synthesise a sentence (MMS) | ~0.8 s |
| Full speech → speech pipeline | **~7 s** |

Dynamic int8 quantisation of NLLB's linear layers is the single biggest win — a
**5× speedup** with no measurable quality loss, on by default on CPU.

Beam width stays at 4 even for streaming. Dropping to 1–2 saves under a second
but measurably degrades output, including a Hindi rendering of "brown fox" that
came out as an obscenity at narrow beams.

---

## Running it

Python 3.12+, ~4 GB free RAM, ~4 GB disk for weights. FFmpeg ships inside PyAV.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env                        # optional; defaults work
python backend/scripts/download_models.py   # ~2.9 GB
python backend/wsgi.py
curl localhost:5000/api/health
```

Or with Docker:

```bash
docker build -t lingualive .
docker run -p 8000:8000 -v lingualive-models:/app/models lingualive
```

Every setting is documented in [.env.example](.env.example).

## Tests

```bash
cd backend
pytest -q                                   # 236 unit tests, fake engines, ~8 s
pytest -m integration -q                    # 13 tests on real weights, ~3 min
python scripts/smoke_test.py --base-url http://127.0.0.1:5000
```

---

## Licensing

| Component | Licence | Commercial use |
|---|---|---|
| NLLB-200-distilled-600M | CC-BY-NC-4.0 | No |
| MMS-TTS voices | CC-BY-NC-4.0 | No |
| Whisper (via faster-whisper) | MIT | Yes |
| gTTS | MIT (wrapper); Google's terms apply to the service | See Google's terms |

The stack as configured is non-commercial because of NLLB and MMS. The engine
abstraction is the swap point: MADLAD-400 or OPUS-MT (Apache-2.0) for
translation and Piper (MIT) for synthesis, at the cost of narrower language
coverage.
