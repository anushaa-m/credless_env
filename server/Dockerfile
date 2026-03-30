FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib numpy requests pydantic openai
RUN pip install "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"

RUN python credless_model/train.py

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]