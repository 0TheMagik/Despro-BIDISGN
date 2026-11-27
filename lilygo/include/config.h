#pragma once

// ------------------------------------------------------------------
//  Kredensial WiFi — ganti dengan WiFi lab/kampus.
//  Jangan commit file ini kalau repo dipublikasikan; ambil dari
//  menu PlatformIO > Project Options > Secret manager bila perlu.
// ------------------------------------------------------------------
#define WIFI_SSID        "NAMA_WIFI_ANDA"
#define WIFI_PASSWORD    "PASSWORD_WIFI_ANDA"
#define WIFI_TIMEOUT_MS  15000

// ------------------------------------------------------------------
//  Server inferensi (repo Deployment/ — FastAPI di port 8000).
//  Pakai IP LAN, bukan localhost, karena kacamata konek via WiFi.
// ------------------------------------------------------------------
#define SERVER_HOST      "192.168.1.100"
#define SERVER_PORT      8000
#define STREAM_PATH      "/predict"

// ------------------------------------------------------------------
//  Geometri layar T-Glass: area yang terlihat pada kacamata ~126x126
//  (panel sebenarnya 294x126, separuh dipakai refresh area).
// ------------------------------------------------------------------
#define GLASS_VIEW_W     126
#define GLASS_VIEW_H     126

// ------------------------------------------------------------------
//  Mikrofon PDM on-board: 16 kHz, 16 bit, mono (fixed di hardware).
// ------------------------------------------------------------------
#define MIC_SAMPLE_RATE  16000
#define MIC_FRAME_SAMPLES 512
