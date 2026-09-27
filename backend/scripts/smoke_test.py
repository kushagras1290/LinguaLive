"""End-to-end smoke test against a running server.

Exercises every endpoint including the WebSocket stream, and prints a table of
results. Use it after a deploy, or after changing engines, to confirm the whole
stack works rather than just the parts unit tests reach.

Usage::

    python backend/scripts/smoke_test.py
    python backend/scripts/smoke_test.py --base-url https://your-app.example.com
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402


@dataclass
class Results:
    """Tally of checks run."""

    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        """Record and print one check."""
        if ok:
            self.passed += 1
            print(f"  PASS  {name}{f'  ({detail})' if detail else ''}")
        else:
            self.failed += 1
            self.failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  {detail}")


def build_reference_speech() -> bytes | None:
    """Synthesise a known sentence with gTTS to use as ASR input."""
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text="The quick brown fox jumps over the lazy dog.", lang="en").write_to_fp(
            buffer
        )
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  reference speech unavailable ({exc})")
        return None


def check_metadata(base_url: str, results: Results) -> None:
    """Health and language endpoints."""
    print("\nMetadata")
    try:
        health = requests.get(f"{base_url}/api/health", timeout=30).json()
        results.record(
            "health",
            health.get("status") == "ok",
            f"{health.get('memory', {}).get('rss_mb')} MB resident",
        )
    except Exception as exc:  # noqa: BLE001
        results.record("health", False, str(exc))
        return

    try:
        languages = requests.get(f"{base_url}/api/languages", timeout=30).json()
        counts = languages["counts"]
        results.record(
            "languages",
            counts["total"] == 202,
            f"{counts['total']} total, {counts['transcribable']} transcribable, "
            f"{counts['speakable']} speakable",
        )
    except Exception as exc:  # noqa: BLE001
        results.record("languages", False, str(exc))


def check_translation(base_url: str, results: Results) -> None:
    """Translation across several scripts."""
    print("\nTranslation")
    source = "Good morning. How are you today?"
    for target, script_range in [
        ("hi", (0x0900, 0x097F)),
        ("ar", (0x0600, 0x06FF)),
        ("ru", (0x0400, 0x04FF)),
        ("es", None),
    ]:
        try:
            started = time.perf_counter()
            response = requests.post(
                f"{base_url}/api/translate",
                json={"text": source, "source_lang": "en", "target_lang": target},
                timeout=180,
            )
            elapsed = time.perf_counter() - started
            text = response.json()["text"]

            ok = bool(text.strip())
            if ok and script_range:
                low, high = script_range
                ok = sum(1 for c in text if low <= ord(c) <= high) > 3

            results.record(f"translate en->{target}", ok, f"{elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            results.record(f"translate en->{target}", False, str(exc))


def check_synthesis(base_url: str, results: Results) -> None:
    """The TTS chain, including languages with no neural voice."""
    print("\nSpeech synthesis")
    cases = [
        ("en", "Hello world", "neural"),
        ("hi", "नमस्ते दुनिया", "neural"),
        ("ja", "こんにちは世界", "gTTS fallback"),
        ("bo", "བཀྲ་ཤིས་བདེ་ལེགས", "formant fallback"),
        ("sat", "ᱡᱚᱦᱟᱨ", "formant fallback"),
    ]
    for lang, text, expectation in cases:
        try:
            response = requests.post(
                f"{base_url}/api/speak", json={"text": text, "lang": lang}, timeout=300
            )
            engine = response.headers.get("X-TTS-Engine", "?")
            results.record(
                f"speak {lang} ({expectation})",
                response.status_code == 200 and len(response.content) > 1000,
                f"{engine}, {len(response.content)} bytes",
            )
        except Exception as exc:  # noqa: BLE001
            results.record(f"speak {lang}", False, str(exc))


def check_transcription(base_url: str, audio: bytes | None, results: Results) -> None:
    """Speech recognition against known reference audio."""
    print("\nTranscription")
    if audio is None:
        return

    try:
        started = time.perf_counter()
        response = requests.post(
            f"{base_url}/api/transcribe",
            files={"audio": ("reference.mp3", audio, "audio/mpeg")},
            data={"source_lang": "en"},
            timeout=300,
        )
        elapsed = time.perf_counter() - started
        text = response.json().get("text", "").lower()
        hits = sum(word in text for word in ("quick", "brown", "fox", "lazy", "dog"))
        results.record("transcribe", hits >= 4, f"{hits}/5 keywords in {elapsed:.1f}s")
    except Exception as exc:  # noqa: BLE001
        results.record("transcribe", False, str(exc))


def check_pipeline(base_url: str, audio: bytes | None, results: Results) -> None:
    """The combined transcribe/translate/speak route."""
    print("\nPipeline")
    if audio is None:
        return

    try:
        started = time.perf_counter()
        response = requests.post(
            f"{base_url}/api/pipeline",
            files={"audio": ("reference.mp3", audio, "audio/mpeg")},
            data={"source_lang": "en", "target_lang": "hi", "speak": "true"},
            timeout=420,
        )
        elapsed = time.perf_counter() - started
        body = response.json()

        results.record(
            "pipeline transcription",
            bool(body.get("transcription", {}).get("text", "").strip()),
        )
        results.record(
            "pipeline translation",
            bool((body.get("translation") or {}).get("text", "").strip()),
        )
        speech = body.get("speech")
        results.record(
            "pipeline speech",
            speech is not None and len(base64.b64decode(speech["audio_base64"])) > 1000,
            f"whole pipeline {elapsed:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        results.record("pipeline", False, str(exc))


def check_errors(base_url: str, results: Results) -> None:
    """Error handling returns the documented status codes."""
    print("\nError handling")
    cases: list[tuple[str, int, dict]] = [
        (
            "unknown language -> 400",
            400,
            {
                "method": "post",
                "path": "/api/translate",
                "json": {"text": "hi", "source_lang": "en", "target_lang": "zzz"},
            },
        ),
        (
            "missing field -> 400",
            400,
            {"method": "post", "path": "/api/translate", "json": {"text": "hi"}},
        ),
        (
            "unknown route -> 404",
            404,
            {"method": "get", "path": "/api/nonexistent"},
        ),
        (
            "wrong method -> 405",
            405,
            {"method": "get", "path": "/api/translate"},
        ),
    ]

    for name, expected_status, spec in cases:
        try:
            method = spec.pop("method")
            path = spec.pop("path")
            response = getattr(requests, method)(
                f"{base_url}{path}", timeout=30, **spec
            )
            body = response.json()
            results.record(
                name,
                response.status_code == expected_status and "error" in body,
                f"got {response.status_code}, code={body.get('error', {}).get('code')}",
            )
        except Exception as exc:  # noqa: BLE001
            results.record(name, False, str(exc))

    # Undecodable audio must be rejected as unsupported media, not crash.
    try:
        response = requests.post(
            f"{base_url}/api/transcribe",
            files={"audio": ("junk.bin", b"not audio at all" * 100, "audio/wav")},
            timeout=60,
        )
        results.record(
            "undecodable audio -> 415", response.status_code == 415, f"got {response.status_code}"
        )
    except Exception as exc:  # noqa: BLE001
        results.record("undecodable audio -> 415", False, str(exc))


def check_websocket(base_url: str, results: Results) -> None:
    """The live streaming endpoint."""
    print("\nLive streaming")
    try:
        from websockets.sync.client import connect
    except ImportError:
        print("  SKIP  websockets package not installed (pip install websockets)")
        return

    import math
    import struct

    url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"

    try:
        with connect(url, open_timeout=30) as socket:
            socket.send(
                json.dumps(
                    {
                        "type": "config",
                        "source_lang": "en",
                        "target_lang": "hi",
                        "speak": False,
                    }
                )
            )
            ready = json.loads(socket.recv(timeout=30))
            results.record(
                "websocket handshake",
                ready.get("type") == "ready",
                f"sample_rate={ready.get('sample_rate')}",
            )

            # A tone is not speech, so no transcript is expected; this checks the
            # transport, framing and VAD rather than recognition quality.
            sample_rate = ready.get("sample_rate", 16000)
            frames_sent = 0
            for block in range(20):
                samples = [
                    int(12000 * math.sin(2 * math.pi * 220 * n / sample_rate))
                    for n in range(block * 1024, (block + 1) * 1024)
                ]
                socket.send(struct.pack(f"<{len(samples)}h", *samples))
                frames_sent += 1

            socket.send(json.dumps({"type": "stop"}))
            results.record("websocket audio frames", frames_sent == 20, f"{frames_sent} frames")

    except Exception as exc:  # noqa: BLE001
        results.record("websocket", False, str(exc))


def main() -> int:
    """Run every check. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Smoke-test a running server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--skip-audio", action="store_true", help="Skip ASR checks.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Smoke testing {base_url}")

    results = Results()
    check_metadata(base_url, results)
    if results.failed:
        print("\nServer is not reachable; aborting.")
        return 1

    check_translation(base_url, results)
    check_synthesis(base_url, results)

    audio = None if args.skip_audio else build_reference_speech()
    check_transcription(base_url, audio, results)
    check_pipeline(base_url, audio, results)

    check_errors(base_url, results)
    check_websocket(base_url, results)

    print(f"\n{'=' * 60}")
    print(f"{results.passed} passed, {results.failed} failed")
    if results.failures:
        print("\nFailures:")
        for failure in results.failures:
            print(f"  - {failure}")
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
