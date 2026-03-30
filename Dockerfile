# server/Dockerfile
# ✅ Use plain python image — no dependency on openenv-base:latest
FROM python:3.11-slim

WORKDIR /app/env

# ✅ System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# ✅ Set PYTHONPATH BEFORE any RUN steps that import code
ENV PYTHONPATH=/app/env
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2

# ✅ Install Python dependencies first (layer cache)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ✅ Copy ALL project files
COPY . /app/env

# ✅ Train model (PYTHONPATH already set above)
RUN python credless_model/train.py

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
     "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]