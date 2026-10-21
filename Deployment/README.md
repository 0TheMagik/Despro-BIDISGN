# Deployment BISINDO

Folder ini memisahkan pipeline deployment menjadi dua service yang bisa dijalankan dengan dua mode:

1. Local Docker untuk testing di mesin sendiri.
2. Cloud Run GCP untuk deployment online.

Alur yang dipakai:

Client atau aplikasi mengirim video ke service handlandmarker. Service ini mengekstrak keypoint tangan menjadi array berukuran tetap, lalu hasilnya dikirim ke service model untuk prediksi label BISINDO.

## Struktur

- `handlandmarker/` - service preprocessing video menjadi keypoint.
- `model/` - service inferensi CNN + LSTM berbasis PyTorch.
- `docker-compose.yml` - menjalankan keduanya secara lokal.

## Jalankan lokal dengan Docker

```bash
docker compose -f Deployment/docker-compose.yml up --build
```

Setelah jalan, service tersedia di:

- Handlandmarker: `http://localhost:8001`
- Model: `http://localhost:8002`

Contoh alur test:

1. Upload video ke `POST /extract` pada service handlandmarker.
2. Ambil field `keypoints` dari response.
3. Kirim `instances: [keypoints]` ke `POST /predict` pada service model.

## Aplikasi realtime client Python

Kalau kamu ingin pengalaman seperti `realtime_detection.py` tetapi memakai endpoint Docker, jalankan file [Deployment/realtime_detection_docker.py](Deployment/realtime_detection_docker.py).

```bash
python Deployment/realtime_detection_docker.py
```

Kalau service model tidak berada di `localhost:8002`, atur URL WebSocket:

```bash
set MODEL_WS_URL=ws://127.0.0.1:8002/ws/predict
```

Script ini tetap realtime dari webcam lokal dan mengirim sequence landmark langsung ke endpoint WebSocket model.

## Cloud Run GCP

Kedua container bisa dipakai di Cloud Run karena masing-masing sudah expose FastAPI di port `8080`.

### Build image

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
docker build -f Deployment/handlandmarker/Dockerfile -t REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/handlandmarker:latest .
docker build -f Deployment/model/Dockerfile -t REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/model:latest .
docker push REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/handlandmarker:latest
docker push REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/model:latest
```

### Deploy handlandmarker ke Cloud Run

```bash
gcloud run deploy handlandmarker-service \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/handlandmarker:latest \
  --region REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Deploy model ke Cloud Run

```bash
gcloud run deploy model-service \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/model:latest \
  --region REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## Endpoint

### Handlandmarker

- `GET /health`
- `POST /extract`

`POST /extract` menerima `multipart/form-data` dengan field `file`.

### Model

- `GET /health`
- `POST /predict`
- `WebSocket /ws/predict`

Kedua endpoint prediksi menerima JSON dengan format:

```json
{
  "landmarks": [
    [[...126 nilai...], ...]
  ]
}
```

Klien realtime memakai `WebSocket /ws/predict`; endpoint HTTP dipakai untuk pengujian satu request.

## Catatan penting

- `realtime_detection.py` tetap berguna untuk local webcam testing, tetapi tidak cocok langsung dipindah ke Cloud Run karena bergantung pada kamera lokal.
- Model service sekarang akan memotong atau padding sequence menjadi 90 frame dan 126 fitur per frame, supaya output handlandmarker tetap valid untuk inferensi.
- Model service memuat `Prototipe kode versi pytorch/best_model/model_cnn_lstm_isyarat.pt` dari checkpoint PyTorch terbaik.
- Urutan dan seluruh label kosakata dibaca dari field `actions` di checkpoint, sehingga tidak ada daftar label hardcode yang dapat tertinggal dari model.
