FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy project
COPY . /app

# Train model
RUN python credless_model/train.py

# Runtime config (IMPORTANT CHANGE)
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV PYTHONPATH=/app

EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start server
CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]