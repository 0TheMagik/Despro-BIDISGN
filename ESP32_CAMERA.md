# Deteksi dari stream ESP32-CAM melalui OpenCV

`realtime_detection.py` dapat memakai kamera lokal seperti sebelumnya atau
langsung menarik MJPEG stream dari ESP32-CAM.

## Menjalankan dengan ESP32-CAM

Pastikan laptop dan ESP32-CAM tersambung ke jaringan Wi-Fi yang sama. Jalankan
perintah berikut dari root repository:

```powershell
$env:CAMERA_SOURCE = "http://10.149.59.6:81/stream"
python realtime_detection.py
```

Ganti alamat IP dengan alamat yang tercetak pada Serial Monitor ESP32-CAM.
Alamat stream harus memakai port `81` dan path `/stream`; halaman kontrol pada
port 80 bukan feed video yang dibaca OpenCV.

Tekan `q` pada jendela video untuk berhenti.

## Kembali ke webcam laptop

Jika `CAMERA_SOURCE` tidak diatur, script tetap menggunakan webcam lokal
indeks 0 dan mencoba indeks 1 serta 2 sebagai fallback:

```powershell
Remove-Item Env:CAMERA_SOURCE -ErrorAction SilentlyContinue
python realtime_detection.py
```

Pipeline MediaPipe tetap memakai `num_hands=2`, menghasilkan 126 fitur per
frame, mengumpulkan 90 frame, lalu menjalankan model lokal PyTorch. Script ini
tidak mengubah Docker atau service Cloud Run.
