# Deployment BISINDO

Folder ini memisahkan pipeline deployment menjadi dua service yang bisa dijalankan dengan dua mode:

1. Local Docker untuk testing di mesin sendiri.
2. Cloud Run GCP untuk deployment online.

Alur yang dipakai:

Client atau aplikasi mengirim video ke service handlandmarker. Service ini mengekstrak keypoint tangan menjadi array berukuran tetap, lalu hasilnya dikirim ke service model untuk prediksi label BISINDO.

## Struktur

- `handlandmarker/` - service preprocessing video menjadi keypoint.
- `model/` - service inferensi CNN + LSTM.
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

## Deteksi realtime mandiri (output teks)

Untuk memakai webcam, MediaPipe, dan model CNN-LSTM PyTorch langsung di komputer—tanpa service Docker
atau Cloud—jalankan:

```bash
python realtime_detection.py
```

Secara default, script memuat checkpoint `best_model` dari
`Prototipe kode versi pytorch/best_model/model_cnn_lstm_isyarat.pt`. Untuk membandingkannya dengan
`base_model`, pilih checkpoint melalui PowerShell:

```powershell
$env:REALTIME_MODEL = "base"
python realtime_detection.py

# Ubah kembali untuk menguji best model
$env:REALTIME_MODEL = "best"
python realtime_detection.py
```

Script mengumpulkan sequence webcam sepanjang metadata checkpoint (default 90 frame), menormalisasi
landmark, lalu menjalankan inferensi PyTorch langsung.
Huruf A-Z digabung menjadi teks, sedangkan prediksi kata diberi spasi di panel `Teks`. Checkpoint ini
berbeda dari checkpoint service Docker, jadi kelas yang tersedia dan hasil prediksi bisa berbeda.

- MediaPipe memproses landmark di komputer dan inferensi PyTorch berjalan di thread terpisah.
- Prediksi hanya ditambahkan setelah stabil pada minimal 2 dari 3 prediksi dan confidence di atas 0.35.
- Ambang confidence bisa diubah dengan `REALTIME_CONFIDENCE_THRESHOLD` (misalnya `0.25` untuk lebih sensitif).
- Label yang sama tidak ditambahkan berulang sampai tangan tidak terlihat selama 8 frame.
- Kontrol: `SPACE` menambah spasi, `BACKSPACE` menghapus, `C` reset teks, `Q` keluar.

Gunakan [Deployment/realtime_detection_docker.py](Deployment/realtime_detection_docker.py) bila ingin
client yang memanggil service handlandmarker/model melalui HTTP. Untuk menguji seluruh service di Docker
lokal, jalankan:

```powershell
docker compose -f Deployment\docker-compose.yml up --build -d
$env:MODEL_URL = "http://localhost:8002"
python Deployment\realtime_detection_docker.py --backend docker
```

### Catatan penting

1. Frame kamera asli diproses tanpa mirror; flip hanya untuk preview.
2. Standalone memakai window 90 frame dan normalisasi landmark per frame. Model service Docker me-resample input ke panjang sequence training sebelum inferensi.
3. Label standalone diambil dari field `actions` checkpoint `best_model`; label service Docker berasal dari checkpoint/sidecar berbeda. Jangan menukar urutan label antar-checkpoint.
4. Untuk mengulang label yang sama, lepaskan tangan dari kamera setidaknya 8 frame.

### Pengaturan client (file `Deployment/.env`)

| Variable                     | Default | Keterangan                                                     |
| ---------------------------- | ------- | -------------------------------------------------------------- |
| `LANDMARK_BACKEND`           | `auto`  | `local` (MediaPipe di client), `docker` (service), atau `auto`  |
| `WINDOW_FRAMES`              | 36      | Jumlah frame yang dikirim per prediksi (di-resample ke 90)      |
| `MIN_WINDOW_FRAMES`          | 24      | Minimum frame sebelum prediksi pertama                          |
| `STRIDE_FRAMES`              | 3       | Geser window tiap N frame (makin kecil = makin responsif)       |
| `PREDICTION_THRESHOLD`       | 0.95    | Confidence minimum agar label diterima (0.95 = 95%)             |
| `TOP1_MARGIN`                | 0.20    | Selisih minimum top-1 vs top-2 (menolak prediksi ambigu)        |
| `VOTE_WINDOW`                | 5       | Jumlah prediksi terakhir yang harus sepakat                     |
| `STABLE_PREDICTIONS`         | 5       | Prediksi konsisten berturut-turut sebelum masuk teks            |
| `COMMIT_COOLDOWN_SECONDS`    | 0.6     | Jeda minimum antar-commit                                       |
| `MIN_HOLD_SECONDS`           | 0.3     | Lama gestur harus ditahan stabil sebelum boleh commit           |
| `REQUIRE_GESTURE`            | 1       | Wajib gerakan dulu sebelum commit                               |
| `REQUIRE_STABLE_MOTION`      | 1       | Wajib tangan berhenti stabil sebelum commit                     |
| `GESTURE_MOTION_THRESHOLD`   | 0.020   | Besar gerakan yang dianggap memulai gestur                      |
| `MOVEMENT_THRESHOLD`         | 0.012   | Ambang tangan dianggap diam                                     |
| `STABILITY_FRAMES`           | 6       | Jumlah frame diam berturut-turut agar dianggap stabil           |
| `GESTURE_TIMEOUT_SECONDS`    | 4.0     | Batas waktu dari gerakan sampai commit                          |
| `RELEASE_AFTER_EMPTY_FRAMES` | 8       | Frame tanpa tangan sebelum label "dipegang" dilepas             |
| `MIN_ACTIVE_RATIO`           | 0.6     | Proporsi frame berisi tangan minimum untuk inferensi            |
| `SHOW_STATUS`                | 0       | Tampilkan baris status di video (0 = sembunyikan)               |
| `CAPTURE_WIDTH/HEIGHT`       | 640/480 | Resolusi kamera (lebih kecil = lebih lancar)                    |
| `CAPTURE_FPS`                | 30      | Target FPS kamera                                               |
| `JPEG_QUALITY`               | 80      | Kualitas JPEG saat memakai backend `docker`                     |
| `TEXT_LIMIT`                 | 160     | Panjang maksimum teks yang disimpan                             |
| `DEBUG`                      | 0       | Cetak top-5 prediksi tiap inferensi                             |

### Mode uji tanpa kamera

```bash
python Deployment/realtime_detection_docker.py --video rekaman.mp4 --show
```

Video diproses dengan pipeline yang sama (landmark + sliding window + model) dan hasil teks dicetak di akhir, sehingga bisa dipakai memastikan output sesuai gerakan sebelum menguji webcam.

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
- `POST /extract` — upload video (`multipart/form-data`, field `file`)
- `POST /session/start` — `{"session_id": "..."}`
- `POST /session/frame` — `{"session_id", "frame_data" (base64 JPEG), "timestamp_ms"}` → keypoints 126 + `hand_count`
- `POST /session/end` — `{"session_id": "..."}`

Endpoint `session/*` dipakai client realtime untuk memproses frame satu per satu tanpa menyimpan video.

### Model

- `GET /health`
- `GET /labels` — daftar label + `seq_len`, `features`, `normalize`, `backend`
- `POST /predict`

`POST /predict` menerima JSON:

```json
{
  "instances": [
    [[...126 nilai...], ...]
  ]
}
```

Sequence boleh berapa pun panjangnya (mis. 45 frame); service akan resample ke panjang training, menormalisasi, lalu mengembalikan `label`, `confidence`, `top_k`, dan `scores`.

## Catatan penting

- `realtime_detection.py` memakai checkpoint PyTorch secara lokal (`best_model` secara default; pilih `base` dengan `REALTIME_MODEL`). `Deployment/realtime_detection_docker.py` adalah client terpisah untuk service HTTP lokal atau Cloud.
- Model service Docker memakai checkpoint `Deployment/model/deployment_model/model_cnn_lstm_isyarat.pt` (97 kelas: A-Z + kata). Kedua checkpoint adalah file berbeda dan dapat memiliki label/hasil berbeda. File `.keras` lama hanya 10 kelas.
- Image model memakai PyTorch CPU (bukan TensorFlow), sehingga build lebih ringan dan inferensi cepat tanpa GPU.
- Hasil terbaik didapat jika pencahayaan cukup, satu/dua tangan terlihat penuh, dan gestur ditahan ~1 detik.
