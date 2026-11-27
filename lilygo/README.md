# BISINDO Smart Glasses — firmware LILYGO T-Glass

Firmware kacamata untuk deteksi isyarat BISINDO. Repo ini berisi bagian
firmware kacamata; bagian model ML ada di folder `Deployment/` di root repo.

## Hardware

| Komponen | Spesifikasi |
|---|---|
| Board | LILYGO T-Glass (ESP32-S3 FN4R2) |
| Flash / PSRAM | 4 MB / 2 MB QSPI |
| Layar | AMOLED 1.1" JD9613, panel 294×126, area terlihat 126×126 |
| Mikrofon | PDM on-board, 16 kHz 16-bit mono (I2S0, CLK=6, DATA=5) |
| Sensor | IMU BHI260AP, RTC PCF85063A, vibration motor, tombol sentuh |
| Ekstra | 2× QWIIC, USB-C, baterai LiPo 3.7 V |

> **T-Glass tidak punya kamera on-board.** Kamera OV2640 ada di produk terpisah
> **LILYGO T-Camera-S3**. Lihat bagian "Menambahkan kamera" di bawah.

## Struktur folder

```
lilygo/
├── platformio.ini                 # konfigurasi PlatformIO
├── boards/
│   └── LilyGO-T-Wristband-and-T-Glass.json
├── include/
│   └── config.h                   # WiFi, alamat server, konstanta
└── src/
    └── main.cpp                   # entry point
```

Definisi board diambil apa adanya dari repo resmi LilyGo
(`Xinyuan-LilyGO/LilyGo-T-Wristband-and-T-Glass`). **Jangan** menggantinya
dengan `esp32-s3-devkitc-1` apa adanya: board itu di PlatformIO disetel untuk
modul N8 (8 MB flash, tanpa PSRAM) sedangkan T-Glass memakai 4 MB + PSRAM QSPI.
Salah setelan → firmware tidak muat di flash dan PSRAM tidak terdeteksi.

## Menjalankan

```bash
# build
pio run

# upload ke board
pio run --target upload

# monitor serial (115200)
pio device monitor
```

Dari VS Code: pakai tombol PlatformIO di sidebar, atau Command Palette →
*PlatformIO: Build* / *Upload*.

## Konfigurasi

Semua ada di `include/config.h`:

| Konstanta | Isi |
|---|---|
| `WIFI_SSID` / `WIFI_PASSWORD` | WiFi yang dipakai kacamata |
| `SERVER_HOST` / `SERVER_PORT` | alamat FastAPI di `Deployment/` (IP LAN, bukan localhost) |
| `STREAM_PATH` | endpoint prediksi |
| `GLASS_VIEW_W/H` | area tampilan yang terlihat di kacamata |

## Yang sudah jalan di `src/main.cpp`

- Inisialisasi board, AMOLED, dan LVGL
- UI 126×126 di area terlihat kacamata (judul, status, WiFi, baterai, level mic)
- WiFi STA dengan timeout
- Meter level mikrofon (RMS dari sampel I2S mentah)
- Indikator baterai tiap 5 detik

## Menambahkan kamera (T-Camera-S3)

1. Pasang `LILYGO T-Camera-S3` dan sambungkan ke kacamata (kamera memakai konektor
   FPC, bukan QWIIC — cek pinout di wiki LilyGo untuk wiring T-Glass).
2. Tambahkan driver kamera ke `platformio.ini`:
   ```ini
   lib_deps =
       ...
       espressif/esp32-camera
   ```
3. Set pin kamera (dari board T-Camera-S3) di `include/config.h`, lalu inisialisasi
   `esp_camera_init()` dan ambil frame JPEG → kirim ke `SERVER_HOST`.

## Pin map T-Glass (dari library LilyGo)

```
Display  CS=17  SCK=15  MOSI=16  DC=21  RST=18
I2C      SDA=9   SCL=8
Mic      CLK=6   DATA=5
BHI260AP IRQ=37  CS=36  SCK=35  MISO=34  MOSI=33  RST=47  EN=48
RTC      IRQ=7
Tombol   TOUCH=14  BOOT=0
Baterai  ADC=13
Vibrasi  PIN=38
```
