# Backend container for Railway, Fly.io, Hugging Face Spaces, or any Docker host.
#
# Two stages: the builder compiles wheels, the runtime carries only what is
# needed to serve. That keeps the final image well under half the size of a
# single-stage build, which matters because the PyTorch CPU wheel alone is
# ~200 MB compressed.
#
# Model weights are NOT baked in. NLLB-600M plus Whisper is ~2.9 GB, which would
# make the image slow to pull and impossible to cache usefully. They download to
# a mounted volume on first run instead — see MODEL_CACHE_DIR below.

# --------------------------------------------------------------------------- #
# Build stage
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile any package without a manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .

# The CPU-only PyTorch index avoids pulling ~2 GB of CUDA libraries that no
# free-tier host can use anyway.
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

# --------------------------------------------------------------------------- #
# Runtime stage
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

# ffmpeg provides the codecs PyAV needs to decode WebM/Opus from the browser.
# libgomp1 is required by the CTranslate2 runtime behind faster-whisper.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Run unprivileged. A container that never needs to write outside its cache has
# no reason to run as root.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app backend/ /app/backend/
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend \
    ENVIRONMENT=production \
    LOG_FORMAT=json \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_CACHE_DIR=/app/models \
    EAGER_LOAD_MODELS=true

# Mount a volume here so weights survive restarts. Without one, every cold start
# re-downloads ~2.9 GB.
VOLUME ["/app/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# One worker, several threads: the models live in process memory, so additional
# workers multiply RSS rather than throughput. gevent is required for the
# WebSocket streaming endpoint to work under gunicorn.
CMD ["sh", "-c", "gunicorn --chdir /app/backend wsgi:app \
     --worker-class gevent --workers 1 --threads 4 \
     --bind 0.0.0.0:${PORT} --timeout 300 --graceful-timeout 30 \
     --access-logfile - --error-logfile -"]
