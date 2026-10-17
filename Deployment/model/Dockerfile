FROM python:3.11-slim

WORKDIR /app

# Instal libgomp1 (dibutuhkan oleh TensorFlow / Scikit-Learn / ONNX untuk multi-threading CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
COPY model_cnn_lstm_isyarat.keras model_cnn_lstm_isyarat.keras

ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]