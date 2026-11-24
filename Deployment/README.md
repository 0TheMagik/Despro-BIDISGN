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

## Aplikasi realtime client Python (output teks)

Jalankan [Deployment/realtime_detection_docker.py](Deployment/realtime_detection_docker.py):

```bash
python Deployment/realtime_detection_docker.py
```

Fitur:

- **Cepat**: landmark tangan diproses MediaPipe lokal (default `LANDMARK_BACKEND=auto`) sehingga tidak ada round-trip HTTP per frame. Inferensi model berjalan di thread terpisah, jadi loop kamera tetap lancar. Untuk tetap memakai container: `--backend docker`.
- **Sliding window**: window default 45 frame digeser tiap 6 frame. Service model me-resample ke 90 frame + normalisasi (sama seperti training), sehingga prediksi baru muncul kira-kira tiap 0.2 detik setelah window terisi.
- **Output teks**: huruf A-Z otomatis digabung menjadi kata, label kata otomatis diberi spasi.
- **Anti-noise**: teks hanya dikomit setelah tangan diam sesaat, confidence ≥ 95%, margin top-1 vs top-2 ≥ 20%, dan label konsisten 5 prediksi berturut-turut. Setelah dikomit, label yang sama tidak diulang sampai tangan dilepas/digerakkan lagi.
- Kontrol keyboard: `SPACE` menambah spasi, `BACKSPACE` menghapus, `C` reset teks, `Q` keluar.

Kalau service tidak ada di `localhost`, atur environment variable:

```bash
set HANDLANDMARKER_URL=http://127.0.0.1:8001
set MODEL_URL=http://127.0.0.1:8002
```

### Catatan penting (pipeline harus sama dengan training)

1. Frame dikirim **tanpa mirror** (flip hanya untuk tampilan), supaya koordinat MediaPipe sama persis dengan dataset yang direkam `dataset_collection_V2.py`.
2. Model service melakukan **resample ke 90 frame + normalisasi wrist-relative** persis seperti `normalize_sequence()` pada notebook training PyTorch.
3. Urutan label diambil dari checkpoint `.pt` (`actions`, sorted A-Z lalu kata) dan divalidasi dengan `Deployment/model/deployment_model/labels.json`. Jangan mengubah urutan label secara manual.
4. Service handlandmarker memakai confidence detection `0.2` (sama dengan resep `video_to_numpy.py`). Untuk backend lokal, atur lewat env `HAND_DETECTION_CONFIDENCE`, `HAND_PRESENCE_CONFIDENCE`, `HAND_TRACKING_CONFIDENCE`.
5. Mode ketat bisa dilonggarkan lewat `.env` (turunkan `PREDICTION_THRESHOLD`/`STABLE_PREDICTIONS`, atau set `REQUIRE_GESTURE=0`), tetapi output akan lebih mudah noise. Untuk mengulang huruf/kata yang sama, lepaskan tangan sebentar (minimal `RELEASE_AFTER_EMPTY_FRAMES` frame) lalu sign lagi.

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
docker build -t REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/handlandmarker:latest ./Deployment/handlandmarker
docker build -t REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/model:latest ./Deployment/model
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

- `realtime_detection.py` tetap berguna untuk local webcam testing, tetapi tidak cocok langsung dipindah ke Cloud Run karena bergantung pada kamera lokal.
- Model service memakai `Deployment/model/deployment_model/model_cnn_lstm_isyarat.pt` (97 kelas: A-Z + kata). File `.keras` lama hanya 10 kelas; jika ingin memakainya, set `MODEL_PATH` dan sediakan `labels.json` yang sesuai.
- Image model memakai PyTorch CPU (bukan TensorFlow), sehingga build lebih ringan dan inferensi cepat tanpa GPU.
- Hasil terbaik didapat jika pencahayaan cukup, satu/dua tangan terlihat penuh, dan gestur ditahan ~1 detik.
