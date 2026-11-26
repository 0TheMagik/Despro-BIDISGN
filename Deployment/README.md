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
- **Sliding window per gerakan**: window keypoint di-reset setiap kali gerakan baru dimulai, sehingga prediksi hanya melihat gestur yang sedang berjalan (bukan campuran gestur lama + baru). Window dikirim apa adanya (mis. 36 frame), lalu service model me-resample ke 90 frame + normalisasi seperti training.
- **Output teks**: huruf A-Z otomatis digabung menjadi kata, label kata otomatis diberi spasi.
- **Konsisten dengan gerakan**: probabilitas semua kelas dirata-rata (EMA) antar-prediksi, label harus stabil beberapa prediksi, confidence + margin tinggi, tangan berhenti stabil, dan setelah dikomit decoder **terkunci** sampai tangan digerakkan lagi atau dilepas. Jadi satu gerakan hanya menghasilkan satu huruf/kata (tidak dobel, tidak berubah-ubah).
- **Stabil terhadap jarak kamera**: status "tangan diam" dihitung dari fitur yang sudah dinormalisasi (wrist-relative + skala), bukan koordinat piksel, jadi ambang tidak berubah saat tangan lebih dekat/jauh.
- **Slot tangan konsisten**: urutan tangan kiri/kanan ditempelkan ke frame sebelumnya supaya MediaPipe tidak menukar slot di tengah gestur.
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
5. Mode ketat bisa dilonggarkan lewat `.env` (turunkan `PREDICTION_THRESHOLD`/`MIN_VOTE_SHARE`, atau set `REQUIRE_GESTURE=0`), tetapi output akan lebih mudah noise. Untuk mengulang huruf/kata yang sama: gerakkan tangan sedikit lalu sign lagi (atau lepaskan tangan sebentar) supaya kunci `LOCK_AFTER_COMMIT` terbuka.
6. Kalau tangan sudah dalam posisi huruf sebelum aplikasi jalan, gerakkan dulu sedikit agar sesi gestur baru dimulai (`REQUIRE_GESTURE=1`). Set `REQUIRE_GESTURE=0` untuk mengizinkan gestur statis langsung.

### Pengaturan client (file `Deployment/.env`)

| Variable                     | Default | Keterangan                                                     |
| ---------------------------- | ------- | -------------------------------------------------------------- |
| `LANDMARK_BACKEND`           | `auto`  | `local` (MediaPipe di client), `docker` (service), atau `auto`  |
| `WINDOW_FRAMES`              | 36      | Jumlah frame yang dikirim per prediksi (di-resample ke 90)      |
| `MIN_WINDOW_FRAMES`          | 24      | Minimum frame sebelum prediksi pertama                          |
| `STRIDE_FRAMES`              | 3       | Geser window tiap N frame (makin kecil = makin responsif)       |
| `PREDICTION_THRESHOLD`       | 0.90    | Confidence minimum (setelah smoothing) agar label diterima      |
| `TOP1_MARGIN`                | 0.20    | Selisih minimum top-1 vs top-2 (menolak prediksi ambigu)        |
| `PROB_SMOOTH_ALPHA`          | 0.45    | Bobot prediksi terbaru pada EMA probabilitas (kecil = halus)    |
| `VOTE_WINDOW`                | 4       | Jumlah prediksi terakhir yang dipakai untuk vote                |
| `STABLE_PREDICTIONS`         | 3       | Prediksi konsisten berturut-turut sebelum masuk teks            |
| `MIN_VOTE_SHARE`             | 0.80    | Porsi minimum label yang sama dalam `VOTE_WINDOW`               |
| `LOCK_AFTER_COMMIT`          | 1       | Kunci decoder setelah commit (satu gerakan = satu output)       |
| `REPEAT_COOLDOWN_SECONDS`    | 0.8     | Jeda minimum sebelum gestur berikutnya boleh dikomit            |
| `COMMIT_COOLDOWN_SECONDS`    | 0.4     | Jeda minimum antar-commit                                       |
| `MIN_HOLD_SECONDS`           | 0.25    | Lama gestur harus ditahan stabil sebelum boleh commit           |
| `REQUIRE_GESTURE`            | 1       | Wajib gerakan dulu sebelum commit                               |
| `REQUIRE_STABLE_MOTION`      | 1       | Wajib tangan berhenti stabil sebelum commit                     |
| `GESTURE_MOTION_THRESHOLD`   | 0.020   | Besar gerakan (koordinat gambar) yang dianggap memulai gestur   |
| `STABLE_MOTION_THRESHOLD`    | 0.035   | Ambang tangan dianggap diam (fitur ternormalisasi)              |
| `STABILITY_FRAMES`           | 4       | Jumlah frame diam berturut-turut agar dianggap stabil           |
| `GESTURE_TIMEOUT_SECONDS`    | 5.0     | Batas waktu dari gerakan sampai commit                          |
| `HAND_SLOT_TRACKING`         | 1       | Tempelkan slot tangan ke frame sebelumnya (anti slot tertukar)  |
| `RELEASE_AFTER_EMPTY_FRAMES` | 6       | Frame tanpa tangan sebelum window & kunci dilepas               |
| `MIN_ACTIVE_RATIO`           | 0.6     | Proporsi frame berisi tangan minimum untuk inferensi            |
| `SHOW_STATUS`                | 0       | Tampilkan baris status di video (0 = sembunyikan)               |
| `CAPTURE_WIDTH/HEIGHT`       | 640/480 | Resolusi kamera (lebih kecil = lebih lancar)                    |
| `CAPTURE_FPS`                | 30      | Target FPS kamera                                               |
| `JPEG_QUALITY`               | 80      | Kualitas JPEG saat memakai backend `docker`                     |
| `TEXT_LIMIT`                 | 160     | Panjang maksimum teks yang disimpan                             |
| `DEBUG`                      | 0       | Cetak top-5 prediksi tiap inferensi + nilai motion              |

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

- `realtime_detection.py` tetap berguna untuk local webcam testing, tetapi tidak cocok langsung dipindah ke Cloud Run karena bergantung pada kamera lokal.
- Model service memakai `Deployment/model/deployment_model/model_cnn_lstm_isyarat.pt` (97 kelas: A-Z + kata). File `.keras` lama hanya 10 kelas; jika ingin memakainya, set `MODEL_PATH` dan sediakan `labels.json` yang sesuai.
- Image model memakai PyTorch CPU (bukan TensorFlow), sehingga build lebih ringan dan inferensi cepat tanpa GPU.
- Hasil terbaik didapat jika pencahayaan cukup, satu/dua tangan terlihat penuh, dan gestur ditahan ~1 detik.
