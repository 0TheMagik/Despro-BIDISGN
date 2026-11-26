# Firmware T-Glass — BISINDO & Suara

Sketch Arduino untuk LilyGO **T-Glass** yang menampilkan dua jenis hasil:

- **Mode ISYARAT** — teks hasil translasi BISINDO dari
  `Deployment/realtime_detection_docker.py` (MediaPipe + model CNN+LSTM).
- **Mode SUARA** — teks hasil speech-to-text dari mikrofon PDM bawaan T-Glass,
  diproses oleh `Deployment/glass_bridge.py` di PC (SpeechRecognition, `id-ID`).

T-Glass tidak memiliki kamera, jadi teks isyarat dikirim dari PC melalui Wi-Fi.

Folder sketch juga memuat `build_opt.h` (`-DLV_CONF_INCLUDE_SIMPLE`) agar LVGL
memakai `lv_conf.h` bawaan library LilyGO. Jangan hapus file itu.

```
PC: realtime_detection_docker.py --glass-url http://<IP-PC>:8765
        │  POST /api/sign
        ▼
PC: glass_bridge.py  ──GET /api/sign──►  T-Glass (mode ISYARAT)
        ▲                                    │
        └──── POST /api/speech (PCM) ────────┘  (mode SUARA)
                 SpeechRecognition → teks → T-Glass
```

## 1. Persiapan

### Arduino IDE

Gunakan **Arduino-ESP32 core versi 2.0.5 – 2.0.17** (library LilyGO memakai
driver I2S lama dan `esp_lcd` 2.x; **core 3.x belum didukung**). Board manager
URL (Preferences → Additional boards manager URLs):

```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```

Library yang dibutuhkan — pasang lewat **Tools → Manage Libraries**:

1. Cari `LilyGO T-Wristband and T-Glass` → **Install** (pilih *Install all*
   bila IDE menawarkan dependensi).
2. Cari `lvgl` → pilih versi **8.4.0** (bukan 9.x).
3. Cari `SensorLib` → **0.1.8**.
4. Cari `Madgwick` → **1.2.0**.

Bila library LilyGO tidak muncul di Library Manager, unduh ZIP repo GitHub lalu
**Sketch → Include Library → Add .ZIP Library**, dan salin juga semua folder di
[`libdeps/`](https://github.com/Xinyuan-LilyGO/LilyGO-T-Wristband-and-T-Glass/tree/master/libdeps)
ke `Documents/Arduino/libraries`.

Pengaturan **Tools**:

| Opsi                 | Nilai                                        |
| -------------------- | -------------------------------------------- |
| Board                | **ESP32S3 Dev Module**                       |
| Port                 | port USB board                               |
| USB CDC On Boot      | **Enabled**                                  |
| CPU Frequency        | 240MHz (WiFi)                                |
| Core Debug Level     | None                                         |
| Flash Mode           | QIO 80MHz                                    |
| Flash Size           | **4MB (32Mb)**                               |
| Partition Scheme     | **Huge APP (3MB No OTA/1MB SPIFFS)**         |
| PSRAM                | **QSPI PSRAM**                               |
| Upload Speed         | **921600**                                   |
| USB Mode             | **CDC and JTAG**                             |
| Upload Mode          | **UART0 / Hardware CDC**                     |
| Erase All Flash...   | Disabled                                     |

> LilyGO menyarankan *Default 4M with spiffs (1.2MB APP)* untuk contoh bawaannya.
> Sketch ini menambahkan WiFi + HTTPClient sehingga ukuran firmware lebih besar;
> **Huge APP** memberi ruang 3MB agar tidak kena `Sketch too big`.

### PC

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-speech.txt
```

`SpeechRecognition` memakai layanan Google, jadi PC butuh koneksi internet.
Server bridge binding ke `0.0.0.0:8765` — izinkan Python di Windows Firewall
saat diminta, dan pastikan PC + T-Glass berada di Wi-Fi yang sama.

## 2. Konfigurasi sketch

Ubah bagian atas [TGlass_BISINDO.ino](TGlass_BISINDO.ino):

```cpp
static const char *WIFI_SSID       = "GANTI_SSID";
static const char *WIFI_PASSWORD   = "GANTI_PASSWORD";
static const char *BRIDGE_BASE_URL = "http://192.168.1.10:8765";
static const char *SPEECH_LANGUAGE = "id-ID";

#define TGLASS_VARIANT_V2          0   // 1 jika memakai T-Glass V2
```

T-Glass V1 memakai `flipHorizontal(true)` dan jendela rata kanan-tengah
(contoh `GlassTouchButtonEvent`). T-Glass V2 memakai `setRotation(0)` dan
jendela rata bawah-tengah (contoh `GlassV2/GlassTouchButtonEvent`).

IP PC ditampilkan saat `glass_bridge.py` dijalankan.

## 3. Menjalankan

Urutan yang disarankan:

1. Jalankan bridge di PC:

   ```bash
   python Deployment/glass_bridge.py --port 8765
   ```

   Buka `http://<IP-PC>:8765/` di browser untuk melihat teks isyarat dan suara
   terakhir tanpa kacamata.

2. Jalankan deteksi isyarat dan arahkan teksnya ke bridge:

   ```bash
   python Deployment/realtime_detection_docker.py --glass-url http://<IP-PC>:8765
   ```

   atau set `GLASS_BRIDGE_URL=http://<IP-PC>:8765` di `Deployment/.env`.
   Tanpa opsi ini, script tetap berjalan seperti biasa (tanpa T-Glass).

3. Upload sketch ke T-Glass. Setelah Wi-Fi tersambung, layar menampilkan teks
   isyarat terbaru tiap 400 ms.

## 4. Kontrol tombol sentuh

| Aksi        | Fungsi                                             |
| ----------- | -------------------------------------------------- |
| Klik        | Ganti mode **ISYARAT ⇄ SUARA**                     |
| Klik 2x     | Hapus teks (mode isyarat juga menghapus di bridge) |
| Tekan lama  | Tampilkan IP, sinyal Wi-Fi, status server & mikrofon|
| Klik 3x     | Sambung ulang Wi-Fi                               |

Di mode SUARA, mikrofon mendeteksi suara otomatis (VAD):

- rekaman dimulai setelah 2 frame (60 ms) di atas ambang;
- berhenti setelah 700 ms hening atau maksimal 8 detik;
- rekaman dikirim ke `/api/speech`, hasilnya ditambahkan ke teks di layar.

Rekaman lebih pendek dari 450 ms diabaikan.

## 5. Penyetelan mikrofon

Buka Serial Monitor 115200 baud. Setiap 500 ms firmware mencetak:

```
[MIC] rms=812 floor=240 trigger=672
```

- Suara tidak pernah memicu rekaman → turunkan `MIN_TRIGGER_RMS` (mis. 300)
  atau `TRIGGER_FACTOR`.
- Ruangan berisik memicu rekaman terus → naikkan `MIN_TRIGGER_RMS`.
- Serial juga menampilkan `[SPEECH] <teks>` setelah upload berhasil.

Noise floor diukur otomatis selama 400 ms saat boot, jadi nyalakan kacamata di
ruangan yang relatif tenang.

## 6. Endpoint bridge

| Method | Path                | Isi                                                   |
| ------ | ------------------- | ----------------------------------------------------- |
| GET    | `/api/sign`         | Teks isyarat terbaru (`text/plain`)                   |
| POST   | `/api/sign`         | `{"text": "..."}` dari deteksi isyarat                |
| POST   | `/api/sign/clear`    | Hapus teks isyarat                                    |
| POST   | `/api/speech`       | PCM `s16le` mentah; query `rate`, `width`, `lang`     |
| GET    | `/api/health`       | `ok`                                                  |

## 7. Pemecahan masalah

### Saat compile / upload (Arduino IDE)

| Gejala                                          | Solusi                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `lv_conf.h: No such file or directory`          | Buat file `build_opt.h` di folder sketch berisi `-DLV_CONF_INCLUDE_SIMPLE`, lalu compile ulang.          |
| `#error "Currently not supported 9.x"`          | lvgl 9.x terpasang. Hapus lewat Library Manager, install **lvgl 8.4.0**.                                 |
| `#error "Detected that PSRAM is not turned on"` | Tools → PSRAM = **QSPI PSRAM**.                                                                          |
| `Sketch too big`                                | Tools → Partition Scheme = **Huge APP (3MB No OTA/1MB SPIFFS)**.                                         |
| Error aneh saat upload / port hilang            | Tahan **BOOT**, tekan **RST**, lepas RST, lepas BOOT, lalu Upload lagi.                                  |
| Serial Monitor kosong                           | Tools → **USB CDC On Boot = Enabled** dan baud **115200**.                                               |
| Build gagal di path panjang/berisi spasi        | Salin folder `TGlass_BISINDO` ke `C:\Arduino\TGlass_BISINDO` lalu buka dari sana (hindari OneDrive sync).|
| `i2s_driver_install` / `esp_lcd` error          | Core ESP32 3.x terpasang. Turunkan ke **2.0.x** (mis. 2.0.17).                                           |

### Saat dijalankan

| Gejala                          | Penyebab / solusi                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------- |
| Layar `OFF`                     | Wi-Fi salah / belum terhubung. Cek SSID, password, dan band 2.4 GHz.              |
| Layar `ERR` + `server error`    | `glass_bridge.py` tidak jalan, IP/port salah, atau firewall memblokir port 8765.  |
| Teks isyarat tidak muncul       | Jalankan deteksi dengan `--glass-url`; lihat `[TEKS]` di terminal deteksi.        |
| Mode suara tidak bereaksi       | Cek `[MIC]` di Serial Monitor; sesuaikan `MIN_TRIGGER_RMS`.                       |
| `gagal: ... 503`                | `SpeechRecognition` belum terpasang di PC.                                        |
| `gagal: ... 502`                | PC tidak bisa menghubungi layanan Google Speech.                                  |
| Karakter aneh di layar          | Font Montserrat hanya memuat ASCII; teks non-ASCII bisa tampil kosong.            |
| Board tidak terdeteksi          | Set Core Debug Level ke Error; pastikan PSRAM QSPI dan board T-Glass.             |

## Catatan

- Bridge tidak memakai autentikasi. Gunakan hanya di jaringan lokal tepercaya.
- Audio mikrofon dikirim ke PC, lalu ke layanan Google Speech Recognition.
- Buffer rekaman 8 detik (256 KB) dialokasikan di PSRAM; jika alokasi gagal,
  mode suara dinonaktifkan tetapi mode isyarat tetap berjalan.
