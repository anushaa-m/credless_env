# Dockerfile  ← root of project, NOT inside server/
FROM python:3.11-slim

WORKDIR /app/env

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app/env
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV ENABLE_WEB_INTERFACE=true

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ── Copy dataset first (separate layer for cache efficiency) ──────────────────

# ── Copy all project files ────────────────────────────────────────────────────
COPY data/ /app/env/data/
COPY . /app/env

# ── Train model at build time ─────────────────────────────────────────────────


RUN echo "Skipping training during build"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
     "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
