FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EGL_PLATFORM=surfaceless

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-dri \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Salin file dari repo (build context = root repo, lihat docker-compose.yml)
COPY Deployment/handlandmarker/requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Salin kode aplikasi + model hand landmarker
COPY Deployment/handlandmarker/app.py .
COPY handlandmarker/hand_landmarker.task /app/hand_landmarker.task

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]