FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/model

COPY Deployment/model/requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY best_model/model_cnn_lstm_isyarat.keras /app/model/model_cnn_lstm_isyarat.keras
COPY Deployment/model/app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
