"""Pre-download model weights into the configured cache directory.

Run this once before first use so the first request does not pay the download
cost, and so you can confirm the weights land on the intended drive.

Usage::

    python backend/scripts/download_models.py
    python backend/scripts/download_models.py --whisper large-v3
    python backend/scripts/download_models.py --voices hin,spa,swh
    python backend/scripts/download_models.py --skip-nllb

Every download is written under ``MODEL_CACHE_DIR`` (``models/`` beside this
repository by default), never the system drive.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.errors import TranslationAppError  # noqa: E402
from app.logging_conf import configure_logging, get_logger  # noqa: E402

_LOG = get_logger("download_models")

#: Voices fetched by default: the languages most likely to be demonstrated.
_DEFAULT_VOICES: tuple[str, ...] = ("eng", "spa", "fra", "deu", "hin")


def _directory_size_mb(path: Path) -> float:
    """Return the total size of ``path`` in mebibytes."""
    if not path.exists():
        return 0.0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / (1024 * 1024)


def _download_whisper(model_size: str) -> bool:
    """Download and instantiate a faster-whisper model.

    Returns:
        True on success, False if the download failed.
    """
    settings = get_settings()
    device = settings.resolved_whisper_device()
    compute_type = settings.resolved_whisper_compute_type(device)
    target = settings.model_cache_dir / "whisper"

    _LOG.info(
        "Downloading Whisper", extra={"model": model_size, "compute_type": compute_type}
    )
    started = time.perf_counter()
    try:
        from faster_whisper import WhisperModel

        WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(target),
        )
    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not crash
        _LOG.error("Whisper download failed", extra={"model": model_size, "error": str(exc)})
        return False

    _LOG.info(
        "Whisper ready",
        extra={
            "model": model_size,
            "seconds": round(time.perf_counter() - started, 1),
            "cache_mb": round(_directory_size_mb(target), 1),
        },
    )
    return True


def _download_nllb(model_id: str) -> bool:
    """Download the NLLB translation model and tokenizer.

    Returns:
        True on success, False if the download failed.
    """
    _LOG.info("Downloading NLLB", extra={"model": model_id})
    started = time.perf_counter()
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        _LOG.error("NLLB download failed", extra={"model": model_id, "error": str(exc)})
        return False

    _LOG.info(
        "NLLB ready",
        extra={"model": model_id, "seconds": round(time.perf_counter() - started, 1)},
    )
    return True


def _download_voices(voices: tuple[str, ...]) -> tuple[int, int]:
    """Download MMS-TTS voices.

    Returns:
        A ``(succeeded, attempted)`` pair. Missing voices are not fatal, because
        the TTS chain falls back to gTTS and the built-in formant synthesiser.
    """
    succeeded = 0
    for code in voices:
        model_id = f"facebook/mms-tts-{code}"
        try:
            from transformers import AutoTokenizer, VitsModel

            AutoTokenizer.from_pretrained(model_id)
            VitsModel.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "MMS voice unavailable; the chain will fall back",
                extra={"voice": code, "error": str(exc)[:120]},
            )
            continue
        succeeded += 1
        _LOG.info("MMS voice ready", extra={"voice": code})
    return succeeded, len(voices)


def main() -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Pre-download model weights into the configured cache directory."
    )
    parser.add_argument(
        "--whisper",
        default=None,
        help="Whisper size or model id (default: the configured WHISPER_MODEL).",
    )
    parser.add_argument(
        "--voices",
        default=",".join(_DEFAULT_VOICES),
        help="Comma-separated MMS voice codes (ISO 639-3), or 'none' to skip.",
    )
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper.")
    parser.add_argument("--skip-nllb", action="store_true", help="Skip NLLB.")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except TranslationAppError as exc:
        _LOG.error("Configuration is invalid", extra={"error": exc.message})
        return 2

    configure_logging(level=settings.log_level, json_output=False)

    cache_dir = settings.model_cache_dir
    free_gb = shutil.disk_usage(cache_dir.anchor).free / (1024**3)
    _LOG.info(
        "Cache directory",
        extra={"path": str(cache_dir), "drive": cache_dir.anchor, "free_gb": round(free_gb, 1)},
    )
    if free_gb < 5.0:
        _LOG.warning("Less than 5 GB free on the cache drive", extra={"free_gb": round(free_gb, 1)})

    ok = True
    if not args.skip_whisper:
        ok &= _download_whisper(args.whisper or settings.whisper_model)
    if not args.skip_nllb:
        ok &= _download_nllb(settings.nllb_model)

    if args.voices.strip().lower() != "none":
        codes = tuple(c.strip() for c in args.voices.split(",") if c.strip())
        succeeded, attempted = _download_voices(codes)
        _LOG.info("MMS voices complete", extra={"succeeded": succeeded, "attempted": attempted})

    _LOG.info(
        "Download complete",
        extra={"total_cache_mb": round(_directory_size_mb(cache_dir), 1), "all_succeeded": ok},
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
