FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/model/deployment_model/model_cnn_lstm_isyarat.pt \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/model

# Build context adalah folder Deployment/model.
COPY requirements.txt requirements.txt

# Install PyTorch versi CPU lebih dulu supaya image tidak menarik paket CUDA (~2GB).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install -r requirements.txt

# Model dan label disimpan dalam struktur deployment_model agar konsisten
# dengan MODEL_PATH dan fallback resolve_model_path() di app.py.
COPY model_cnn_lstm_isyarat.pt /app/model/deployment_model/model_cnn_lstm_isyarat.pt
COPY labels.json /app/model/deployment_model/labels.json
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]