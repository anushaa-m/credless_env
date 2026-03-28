# server/Dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ── Copy all source code ──────────────────────────────────────────────────────
COPY . /app

# ── Pre-build: train and save the CredLess model ─────────────────────────────
RUN python credless_model/train.py

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", \
     "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
```

And the `server/requirements.txt` you need alongside it:
```
# server/requirements.txt
fastapi
uvicorn[standard]
pydantic>=2.0
scikit-learn
joblib
numpy
openai
requests
git+https://github.com/meta-pytorch/OpenEnv.git