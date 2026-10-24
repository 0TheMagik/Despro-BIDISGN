# Client Browser ESP32-CAM

Client ini menjalankan seluruh tahap pemrosesan di browser:

1. Mengambil snapshot JPEG berulang dari ESP32-CAM melalui endpoint `/capture`.
2. Menjalankan MediaPipe HandLandmarker dalam mode `VIDEO`.
3. Mendeteksi maksimal dua tangan.
4. Menghasilkan 126 fitur per frame (`2 x 21 x 3`).
5. Mengirim 90 frame ke endpoint model Cloud Run.

## Menjalankan

Buka `index.html` melalui static server, misalnya:

```powershell
python -m http.server 5500 --directory WebClient
```

Kemudian buka `http://localhost:5500` dari HP/laptop yang berada pada jaringan
yang sama dengan ESP32-CAM. Masukkan alamat dasar kamera, tanpa port, misalnya:

```text
http://192.168.1.50
```

Client menggunakan snapshot:

```text
http://192.168.1.50/capture
```

Metode snapshot dipakai karena halaman kontrol pada port 80 dapat diakses
meskipun browser atau jaringan tidak menampilkan MJPEG multipart pada port 81.
URL `/capture` mengembalikan satu JPEG per request dan client mengambilnya
berulang kali.

Endpoint model dikonfigurasi di `app.js`:

```text
https://despro-bidisgn-model-1031455690132.asia-southeast2.run.app/predict
```

ESP32-CAM tidak perlu mengakses Cloud Run secara langsung. Browser menjadi
penghubung antara stream lokal dan endpoint AI.
