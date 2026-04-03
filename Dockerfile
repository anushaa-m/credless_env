# server/Dockerfile
FROM python:3.11-slim

WORKDIR /app/env

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# ✅ PYTHONPATH set BEFORE any RUN steps that import code
ENV PYTHONPATH=/app/env
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2

# ✅ FIXED: was `server/requirements.txt` — there is only ONE requirements.txt at root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy dataset directory explicitly so training can find it in Docker
COPY data/ /app/env/data/

# Copy all project files
COPY . /app/env

# Train and save model at build time
RUN python credless_model/train.py

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
     "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]




