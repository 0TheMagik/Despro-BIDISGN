FROM python:3.11-slim

WORKDIR /app

# Instal dependensi OS wajib untuk MediaPipe dan OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instal dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi dan file model (.task & model AI)
COPY . .

# Cloud Run menggunakan environment variable PORT (default 8080)
ENV PORT=8080

# Gunakan 1 worker Uvicorn karena Cloud Run mengatur scaling per-container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]