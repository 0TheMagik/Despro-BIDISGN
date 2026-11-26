/**
 * @file      TGlass_BISINDO.ino
 * @brief     Tampilan kacamata T-Glass untuk translasi BISINDO -> teks dan
 *            speech-to-text memakai mikrofon PDM bawaan T-Glass.
 *
 * Alur data:
 *   PC (Deployment/realtime_detection_docker.py)
 *       -> POST /api/sign  -> Deployment/glass_bridge.py
 *   T-Glass -> GET /api/sign (polling) -> tampilkan teks isyarat
 *
 *   T-Glass (mikrofon PDM) -> POST /api/speech -> glass_bridge.py
 *       -> SpeechRecognition (Google, id-ID) -> teks -> T-Glass
 *
 * Tombol sentuh:
 *   klik        : ganti mode ISYARAT <-> SUARA
 *   klik 2x     : hapus teks
 *   tekan lama  : tampilkan info WiFi/server/mikrofon
 *   klik 3x     : sambung ulang WiFi
 *
 * Board  : LilyGO T-Glass (ESP32-S3, 4 MB flash, 2 MB PSRAM)
 * Library: https://github.com/Xinyuan-LilyGO/LilyGO-T-Wristband-and-T-Glass
 *          (lvgl 8.4.0, SensorLib 0.1.8, Madgwick 1.2.0)
 *
 * Pengaturan Arduino IDE:
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 4MB (32Mb)
 *   Partition Scheme : Huge APP (3MB No OTA/1MB SPIFFS)
 *   PSRAM            : QSPI PSRAM
 *   USB CDC On Boot  : Enabled
 */

#include <LilyGo_Wristband.h>
#include <LV_Helper.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ==================== KONFIGURASI ====================
// Sesuaikan dengan jaringan dan PC yang menjalankan glass_bridge.py.
static const char *WIFI_SSID       = "loh";
static const char *WIFI_PASSWORD   = "mautaungga";
static const char *BRIDGE_BASE_URL = "http://10.152.122.31:8765";
static const char *SPEECH_LANGUAGE = "id-ID";

// 0 = T-Glass V1 (flip horizontal + rata kanan-tengah), 1 = T-Glass V2
// (setRotation(0) + rata bawah-tengah, mengikuti contoh GlassV2 LilyGO).
#define TGLASS_VARIANT_V2               0

// Area tampilan T-Glass: 126x126 dengan offset 168 piksel (sama dengan contoh LilyGO).
#define GlassViewable_X_Offset          168
#define GlassViewableWidth              126
#define GlassViewableHeight             126

static const uint8_t SCREEN_BRIGHTNESS = 200;
static const uint16_t TOUCH_THRESHOLD  = 200;

// ==================== PARAMETER JARINGAN ====================
static const uint32_t SIGN_POLL_INTERVAL_MS  = 400;
static const int32_t HTTP_CONNECT_TIMEOUT_MS = 700;
static const int32_t HTTP_READ_TIMEOUT_MS    = 1500;
static const int32_t SPEECH_HTTP_TIMEOUT_MS  = 12000;

// ==================== PARAMETER AUDIO / VAD ====================
// Mikrofon PDM T-Glass: 16 kHz, mono, 16-bit (lihat MIC_I2S_SAMPLE_RATE).
static const uint32_t AUDIO_FRAME_MS      = 30;
static const size_t AUDIO_FRAME_SAMPLES   = MIC_I2S_SAMPLE_RATE / 1000 * AUDIO_FRAME_MS;
static const size_t AUDIO_FRAME_BYTES     = AUDIO_FRAME_SAMPLES * sizeof(int16_t);
static const uint32_t SPEECH_START_FRAMES = 2;
static const uint32_t SPEECH_END_SILENCE_MS = 700;
static const uint32_t SPEECH_MIN_MS       = 450;
static const uint32_t SPEECH_MAX_MS       = 8000;
static const uint8_t PRE_ROLL_FRAMES      = 8;

// Ambang deteksi suara. Naikkan MIN_TRIGGER_RMS jika ruangan berisik,
// turunkan jika suara tidak terdeteksi. Nilai RMS dapat dilihat di Serial Monitor.
static const float MIN_TRIGGER_RMS = 420.0f;
static const float TRIGGER_FACTOR  = 2.8f;
static const float RELEASE_FACTOR  = 0.65f;

static const uint32_t LEVEL_RENDER_MS   = 100;
static const uint32_t STATUS_OVERLAY_MS = 5000;
static const uint32_t WIFI_RETRY_MS     = 5000;

// Batas karakter yang ditampilkan; label 126x126 hanya muat beberapa baris.
static const size_t SIGN_TAIL_CHARS   = 78;
static const size_t SPEECH_TAIL_CHARS = 78;
static const size_t SPEECH_TEXT_LIMIT = 400;

// ==================== WARNA ====================
static const uint32_t COLOR_SIGN   = 0x00E5FF;
static const uint32_t COLOR_SPEECH = 0xFFA726;
static const uint32_t COLOR_OK     = 0x69F0AE;
static const uint32_t COLOR_ERROR  = 0xFF5252;
static const uint32_t COLOR_DIM    = 0x9E9E9E;

// ==================== STATE ====================
enum Mode {
    MODE_SIGN,
    MODE_SPEECH,
};

enum LinkState {
    LINK_UNKNOWN,
    LINK_OK,
    LINK_ERROR,
};

LilyGo_Class amoled;

static Mode mode = MODE_SIGN;
static LinkState linkState = LINK_UNKNOWN;

static String signText;
static String speechText;
static String noticeText;
static uint32_t noticeUntil = 0;
static uint32_t statusOverlayUntil = 0;

static bool micReady = false;
static bool uploading = false;
static bool wifiConnected = false;
static bool uiDirty = true;

static uint32_t lastSignPollMs = 0;
static uint32_t lastWifiAttemptMs = 0;
static uint32_t lastRenderMs = 0;
static uint32_t lastMicDebugMs = 0;

static WiFiClient netClient;

// ==================== AUDIO BUFFER ====================
static int16_t *recordBuf = nullptr;
static size_t recordCapacitySamples = 0;
static size_t recordLenSamples = 0;
static uint32_t recordStartMs = 0;

static int16_t frameBuffer[AUDIO_FRAME_SAMPLES];
static int16_t preRoll[PRE_ROLL_FRAMES][AUDIO_FRAME_SAMPLES];
static uint8_t preRollIndex = 0;
static uint8_t preRollFilled = 0;

static bool speechActive = false;
static uint8_t aboveFrames = 0;
static uint32_t silenceMs = 0;
static float noiseFloor = 200.0f;
static uint8_t levelPercent = 0;

// ==================== LVGL ====================
static lv_obj_t *headerLabel = nullptr;
static lv_obj_t *statusLabel = nullptr;
static lv_obj_t *textLabel = nullptr;
static lv_obj_t *footerLabel = nullptr;
static lv_obj_t *levelBar = nullptr;
static const lv_font_t *bodyFont = nullptr;

// ==================== UTILITAS WAKTU ====================
static bool timerActive(uint32_t deadline)
{
    return deadline != 0 && (int32_t)(millis() - deadline) < 0;
}

static void showNotice(const String &text, uint32_t durationMs)
{
    noticeText = text;
    noticeUntil = millis() + durationMs;
    uiDirty = true;
}

static String shorten(const String &text, size_t maxChars)
{
    if (text.length() <= maxChars) {
        return text;
    }
    return text.substring(0, maxChars);
}

static String tailOf(const String &text, size_t maxChars)
{
    String trimmed = text;
    trimmed.trim();
    if (trimmed.length() <= maxChars) {
        return trimmed;
    }
    return "..." + trimmed.substring(trimmed.length() - maxChars);
}

// ==================== UI ====================
static void setLabel(lv_obj_t *label, const String &value)
{
    const char *current = lv_label_get_text(label);
    if (current != nullptr && value.equals(current)) {
        return;
    }
    lv_label_set_text(label, value.c_str());
}

static void setTextColor(lv_obj_t *obj, uint32_t color)
{
    lv_obj_set_style_text_color(obj, lv_color_hex(color), 0);
}

static void setBodyFont(const lv_font_t *font)
{
    if (bodyFont == font) {
        return;
    }
    bodyFont = font;
    lv_obj_set_style_text_font(textLabel, font, 0);
}

static void buildUi()
{
    lv_obj_set_style_bg_color(lv_scr_act(), lv_color_black(), 0);

    lv_obj_t *window = lv_obj_create(lv_scr_act());
    lv_obj_set_style_bg_color(window, lv_color_black(), 0);
    lv_obj_set_style_border_width(window, 0, 0);
    lv_obj_set_style_radius(window, 0, 0);
    lv_obj_set_style_pad_all(window, 0, 0);
    lv_obj_clear_flag(window, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_size(window, GlassViewableWidth, GlassViewableHeight);
#if TGLASS_VARIANT_V2
    lv_obj_align(window, LV_ALIGN_BOTTOM_MID, 0, 0);
#else
    lv_obj_set_pos(window, 0, GlassViewable_X_Offset);
    lv_obj_align(window, LV_ALIGN_RIGHT_MID, 0, 0);
#endif

    headerLabel = lv_label_create(window);
    lv_obj_set_style_text_font(headerLabel, &lv_font_montserrat_12, 0);
    setTextColor(headerLabel, COLOR_SIGN);
    lv_obj_set_pos(headerLabel, 1, 0);
    lv_label_set_text(headerLabel, "ISYARAT");

    statusLabel = lv_label_create(window);
    lv_obj_set_style_text_font(statusLabel, &lv_font_montserrat_12, 0);
    lv_obj_set_style_text_align(statusLabel, LV_TEXT_ALIGN_RIGHT, 0);
    setTextColor(statusLabel, COLOR_DIM);
    lv_obj_set_width(statusLabel, 52);
    lv_obj_align(statusLabel, LV_ALIGN_TOP_RIGHT, 0, 0);
    lv_label_set_text(statusLabel, "...");

    textLabel = lv_label_create(window);
    lv_obj_set_style_text_font(textLabel, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_color(textLabel, lv_color_white(), 0);
    lv_obj_set_style_text_align(textLabel, LV_TEXT_ALIGN_LEFT, 0);
    lv_label_set_long_mode(textLabel, LV_LABEL_LONG_WRAP);
    lv_obj_set_pos(textLabel, 3, 18);
    lv_obj_set_size(textLabel, 120, 84);
    lv_label_set_text(textLabel, "");
    bodyFont = &lv_font_montserrat_14;

    levelBar = lv_bar_create(window);
    lv_obj_set_size(levelBar, 118, 5);
    lv_obj_set_pos(levelBar, 4, 104);
    lv_bar_set_range(levelBar, 0, 100);
    lv_bar_set_value(levelBar, 0, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(levelBar, lv_color_hex(0x263238), LV_PART_MAIN);
    lv_obj_set_style_bg_color(levelBar, lv_color_hex(COLOR_SPEECH), LV_PART_INDICATOR);
    lv_obj_set_style_radius(levelBar, 2, LV_PART_MAIN);
    lv_obj_set_style_radius(levelBar, 2, LV_PART_INDICATOR);
    lv_obj_add_flag(levelBar, LV_OBJ_FLAG_HIDDEN);

    footerLabel = lv_label_create(window);
    lv_obj_set_style_text_font(footerLabel, &lv_font_montserrat_12, 0);
    setTextColor(footerLabel, COLOR_DIM);
    lv_obj_set_style_text_align(footerLabel, LV_TEXT_ALIGN_CENTER, 0);
    lv_label_set_long_mode(footerLabel, LV_LABEL_LONG_CLIP);
    lv_obj_set_width(footerLabel, GlassViewableWidth);
    lv_obj_set_pos(footerLabel, 0, 111);
    lv_label_set_text(footerLabel, "");
}

static String buildStatusOverlay()
{
    String info;
    info.reserve(120);
    if (WiFi.status() == WL_CONNECTED) {
        info += "WiFi: ";
        info += WiFi.localIP().toString();
        info += "\nSinyal: ";
        info += String(WiFi.RSSI());
        info += " dBm\n";
    } else {
        info += "WiFi: terputus\n";
    }
    info += "Server: ";
    if (linkState == LINK_OK) {
        info += "OK";
    } else if (linkState == LINK_ERROR) {
        info += "GAGAL";
    } else {
        info += "...";
    }
    info += "\nMic: ";
    info += micReady ? "OK" : "GAGAL";
    info += "\nMode: ";
    info += (mode == MODE_SIGN) ? "isyarat" : "suara";
    return info;
}

static const char *statusText()
{
    if (WiFi.status() != WL_CONNECTED) {
        return "OFF";
    }
    if (linkState == LINK_ERROR) {
        return "ERR";
    }
    if (mode == MODE_SPEECH) {
        if (uploading) {
            return "TX";
        }
        if (speechActive) {
            return "REC";
        }
        return (linkState == LINK_OK) ? "OK" : "...";
    }
    return (linkState == LINK_OK) ? "LIVE" : "...";
}

static uint32_t statusColor()
{
    if (WiFi.status() != WL_CONNECTED) {
        return COLOR_DIM;
    }
    if (linkState == LINK_ERROR) {
        return COLOR_ERROR;
    }
    if (uploading || speechActive) {
        return COLOR_ERROR;
    }
    return COLOR_OK;
}

static String footerText()
{
    if (timerActive(statusOverlayUntil)) {
        return "klik: tutup";
    }
    if (timerActive(noticeUntil)) {
        return shorten(noticeText, 26);
    }
    if (mode == MODE_SIGN) {
        if (WiFi.status() != WL_CONNECTED) {
            return "wifi belum siap";
        }
        return (linkState == LINK_ERROR) ? "server error" : "klik: ganti mode";
    }
    if (uploading) {
        return "mengirim audio...";
    }
    if (speechActive) {
        return "merekam...";
    }
    if (WiFi.status() != WL_CONNECTED) {
        return "wifi belum siap";
    }
    return "klik: ganti mode";
}

static void renderScreen()
{
    const bool overlay = timerActive(statusOverlayUntil);
    const bool signMode = (mode == MODE_SIGN);

    setTextColor(statusLabel, statusColor());
    if (overlay) {
        setLabel(statusLabel, "");
    } else {
        setLabel(statusLabel, statusText());
    }

    if (overlay) {
        setTextColor(headerLabel, COLOR_DIM);
        setLabel(headerLabel, "STATUS");
        setBodyFont(&lv_font_montserrat_12);
        setLabel(textLabel, buildStatusOverlay());
    } else if (signMode) {
        setTextColor(headerLabel, COLOR_SIGN);
        setLabel(headerLabel, "ISYARAT");
        setBodyFont(&lv_font_montserrat_14);

        String body = tailOf(signText, SIGN_TAIL_CHARS);
        if (body.isEmpty()) {
            if (WiFi.status() != WL_CONNECTED) {
                body = "menghubungkan wifi";
            } else if (linkState == LINK_ERROR) {
                body = "server tidak terjangkau";
            } else {
                body = "menunggu teks isyarat";
            }
        }
        setLabel(textLabel, body);
    } else {
        setTextColor(headerLabel, COLOR_SPEECH);
        setLabel(headerLabel, "SUARA");
        setBodyFont(&lv_font_montserrat_14);

        String body = tailOf(speechText, SPEECH_TAIL_CHARS);
        if (body.isEmpty()) {
            body = micReady ? "bicara ke mikrofon" : "mikrofon tidak aktif";
        }
        setLabel(textLabel, body);
    }

    if (signMode) {
        lv_obj_add_flag(levelBar, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_clear_flag(levelBar, LV_OBJ_FLAG_HIDDEN);
        lv_bar_set_value(levelBar, levelPercent, LV_ANIM_OFF);
        lv_obj_set_style_bg_color(levelBar,
                                  lv_color_hex(speechActive ? COLOR_ERROR : COLOR_SPEECH),
                                  LV_PART_INDICATOR);
    }

    setLabel(footerLabel, footerText());
}

// ==================== JARINGAN ====================
static String bridgeUrl(const String &path)
{
    String base = BRIDGE_BASE_URL;
    base.trim();
    while (base.endsWith("/")) {
        base.remove(base.length() - 1);
    }
    return base + path;
}

static bool bridgeGet(const String &path, String &body, int &code)
{
    HTTPClient http;
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(HTTP_READ_TIMEOUT_MS);
    http.setReuse(true);

    if (!http.begin(netClient, bridgeUrl(path))) {
        code = -1;
        return false;
    }
    code = http.GET();
    body = http.getString();
    http.end();
    return code > 0;
}

static bool bridgePost(const String &path, const uint8_t *payload, size_t length,
                       const char *contentType, String &body, int &code)
{
    HTTPClient http;
    http.setConnectTimeout(3000);
    http.setTimeout(SPEECH_HTTP_TIMEOUT_MS);
    http.setReuse(true);

    if (!http.begin(netClient, bridgeUrl(path))) {
        code = -1;
        return false;
    }
    http.addHeader("Content-Type", contentType);
    code = http.POST(const_cast<uint8_t *>(payload), length);
    body = http.getString();
    http.end();
    return code > 0;
}

static void pollSign()
{
    if (mode != MODE_SIGN || WiFi.status() != WL_CONNECTED) {
        return;
    }

    String body;
    int code = 0;
    if (!bridgeGet("/api/sign", body, code)) {
        linkState = LINK_ERROR;
        netClient.stop();
    } else if (code == 200) {
        linkState = LINK_OK;
        signText = body;
    } else {
        linkState = LINK_ERROR;
    }
    uiDirty = true;
}

static void clearServerSign()
{
    const char *command = "clear";
    String body;
    int code = 0;
    if (bridgePost("/api/sign/clear", reinterpret_cast<const uint8_t *>(command),
                   strlen(command), "text/plain", body, code)) {
        linkState = (code == 200) ? LINK_OK : LINK_ERROR;
    } else {
        linkState = LINK_ERROR;
        netClient.stop();
    }
}

static bool uploadSpeech(const int16_t *samples, size_t sampleCount, String &text, String &error)
{
    String path = "/api/speech?rate=";
    path += String(MIC_I2S_SAMPLE_RATE);
    path += "&width=2&lang=";
    path += SPEECH_LANGUAGE;

    String body;
    int code = 0;
    const bool reached = bridgePost(path, reinterpret_cast<const uint8_t *>(samples),
                                    sampleCount * sizeof(int16_t),
                                    "application/octet-stream", body, code);
    if (!reached) {
        linkState = LINK_ERROR;
        netClient.stop();
        error = "server tidak terjangkau";
        return false;
    }

    body.trim();
    if (code == 200) {
        linkState = LINK_OK;
        text = body;
        return true;
    }

    linkState = LINK_ERROR;
    error = body.isEmpty() ? ("HTTP " + String(code)) : body;
    return false;
}

static void serviceWifi()
{
    const bool up = (WiFi.status() == WL_CONNECTED);
    if (up != wifiConnected) {
        wifiConnected = up;
        uiDirty = true;
        if (up) {
            Serial.printf("[WIFI] tersambung %s (RSSI %d)\n",
                          WiFi.localIP().toString().c_str(), WiFi.RSSI());
            linkState = LINK_UNKNOWN;
            lastSignPollMs = 0;
        } else {
            Serial.println("[WIFI] terputus");
            linkState = LINK_UNKNOWN;
        }
    }

    if (!up && millis() - lastWifiAttemptMs >= WIFI_RETRY_MS) {
        lastWifiAttemptMs = millis();
        Serial.println("[WIFI] mencoba ulang...");
        WiFi.reconnect();
    }
}

static void startWifi()
{
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiAttemptMs = millis();
    Serial.printf("[WIFI] menghubungkan ke %s...\n", WIFI_SSID);
}

// ==================== AUDIO ====================
static float computeRms(const int16_t *samples, size_t count)
{
    if (count == 0) {
        return 0.0f;
    }
    uint64_t sum = 0;
    for (size_t i = 0; i < count; ++i) {
        const int32_t value = samples[i];
        sum += static_cast<uint64_t>(value * value);
    }
    return sqrtf(static_cast<float>(sum) / static_cast<float>(count));
}

static uint8_t rmsToLevel(float rms, float trigger)
{
    const float low = fmaxf(trigger, 1.0f);
    const float high = low * 6.0f;
    float percent = (rms - low) / (high - low) * 100.0f;
    if (percent < 0.0f) {
        percent = 0.0f;
    }
    if (percent > 100.0f) {
        percent = 100.0f;
    }
    return static_cast<uint8_t>(percent);
}

static void appendSamples(const int16_t *src, size_t count)
{
    if (recordBuf == nullptr || recordLenSamples >= recordCapacitySamples) {
        return;
    }
    size_t room = recordCapacitySamples - recordLenSamples;
    if (count > room) {
        count = room;
    }
    memcpy(recordBuf + recordLenSamples, src, count * sizeof(int16_t));
    recordLenSamples += count;
}

static void resetVad()
{
    speechActive = false;
    aboveFrames = 0;
    silenceMs = 0;
    recordLenSamples = 0;
    preRollIndex = 0;
    preRollFilled = 0;
    levelPercent = 0;
}

static void drainMic()
{
    uint8_t scratch[256];
    size_t readLen = 0;
    for (uint8_t i = 0; i < 64; ++i) {
        if (!amoled.readMicrophone(scratch, sizeof(scratch), &readLen, 0) || readLen == 0) {
            break;
        }
    }
}

static void startRecording()
{
    speechActive = true;
    recordLenSamples = 0;
    silenceMs = 0;
    recordStartMs = millis();

    const size_t frames = preRollFilled;
    const size_t start = (preRollFilled < PRE_ROLL_FRAMES) ? 0 : preRollIndex;
    for (size_t i = 0; i < frames; ++i) {
        appendSamples(preRoll[(start + i) % PRE_ROLL_FRAMES], AUDIO_FRAME_SAMPLES);
    }
    Serial.printf("[MIC] mulai merekam (pre-roll %u frame)\n", static_cast<unsigned>(frames));
}

static void finishRecording()
{
    speechActive = false;
    aboveFrames = 0;
    preRollIndex = 0;
    preRollFilled = 0;
    levelPercent = 0;

    const size_t sampleCount = recordLenSamples;
    recordLenSamples = 0;

    const uint32_t durationMs = static_cast<uint32_t>(
        static_cast<uint64_t>(sampleCount) * 1000ULL / MIC_I2S_SAMPLE_RATE);

    if (sampleCount == 0) {
        uiDirty = true;
        return;
    }
    if (durationMs < SPEECH_MIN_MS) {
        showNotice("terlalu singkat", 1500);
        return;
    }
    if (WiFi.status() != WL_CONNECTED) {
        showNotice("wifi terputus", 2000);
        return;
    }

    uploading = true;
    uiDirty = true;
    renderScreen();
    lv_refr_now(NULL);

    Serial.printf("[MIC] kirim %u sample (%.1f detik)\n",
                  static_cast<unsigned>(sampleCount), durationMs / 1000.0f);

    String text;
    String error;
    const bool ok = uploadSpeech(recordBuf, sampleCount, text, error);
    uploading = false;

    if (!ok) {
        Serial.printf("[MIC] gagal: %s\n", error.c_str());
        showNotice("gagal: " + shorten(error, 18), 3000);
    } else {
        text.trim();
        if (text.isEmpty()) {
            showNotice("suara tidak jelas", 2000);
        } else {
            Serial.printf("[SPEECH] %s\n", text.c_str());
            if (!speechText.isEmpty()) {
                speechText += " ";
            }
            speechText += text;
            if (speechText.length() > SPEECH_TEXT_LIMIT) {
                speechText = speechText.substring(speechText.length() - SPEECH_TEXT_LIMIT);
            }
            showNotice("ok", 1000);
        }
    }

    drainMic();
    resetVad();
    uiDirty = true;
}

static void processAudioFrame(const int16_t *samples, size_t count)
{
    const float rms = computeRms(samples, count);
    const float trigger = fmaxf(noiseFloor * TRIGGER_FACTOR, MIN_TRIGGER_RMS);

    if (!speechActive) {
        if (rms < trigger) {
            noiseFloor = noiseFloor * 0.97f + rms * 0.03f;
        }

        memcpy(preRoll[preRollIndex], samples, AUDIO_FRAME_BYTES);
        preRollIndex = (preRollIndex + 1) % PRE_ROLL_FRAMES;
        if (preRollFilled < PRE_ROLL_FRAMES) {
            preRollFilled++;
        }

        if (rms > trigger) {
            aboveFrames++;
            if (aboveFrames >= SPEECH_START_FRAMES) {
                startRecording();
            }
        } else {
            aboveFrames = 0;
        }
        levelPercent = 0;
    } else {
        appendSamples(samples, count);

        if (rms > trigger * RELEASE_FACTOR) {
            silenceMs = 0;
        } else {
            silenceMs += AUDIO_FRAME_MS;
        }
        levelPercent = rmsToLevel(rms, trigger);

        const uint32_t elapsed = millis() - recordStartMs;
        if (silenceMs >= SPEECH_END_SILENCE_MS ||
                elapsed >= SPEECH_MAX_MS ||
                recordLenSamples >= recordCapacitySamples) {
            finishRecording();
        }
    }

    if (millis() - lastMicDebugMs >= 500) {
        lastMicDebugMs = millis();
        Serial.printf("[MIC] rms=%.0f floor=%.0f trigger=%.0f%s\n",
                      rms, noiseFloor, trigger, speechActive ? " (rekam)" : "");
    }
}

static void serviceSpeech()
{
    size_t readLen = 0;
    const TickType_t timeout = pdMS_TO_TICKS(AUDIO_FRAME_MS + 20);
    if (!amoled.readMicrophone(frameBuffer, AUDIO_FRAME_BYTES, &readLen, timeout)) {
        return;
    }
    if (readLen != AUDIO_FRAME_BYTES) {
        return;
    }
    processAudioFrame(frameBuffer, AUDIO_FRAME_SAMPLES);
}

static void calibrateNoiseFloor()
{
    float sum = 0.0f;
    uint8_t frames = 0;
    const uint32_t started = millis();
    size_t readLen = 0;

    while (millis() - started < 400 && frames < 16) {
        if (amoled.readMicrophone(frameBuffer, AUDIO_FRAME_BYTES, &readLen,
                                  pdMS_TO_TICKS(AUDIO_FRAME_MS + 20)) &&
                readLen == AUDIO_FRAME_BYTES) {
            sum += computeRms(frameBuffer, AUDIO_FRAME_SAMPLES);
            frames++;
        }
    }

    noiseFloor = frames ? fmaxf(sum / frames, 80.0f) : 200.0f;
    Serial.printf("[MIC] noise floor awal %.0f (dari %u frame)\n",
                  noiseFloor, static_cast<unsigned>(frames));
}

// ==================== MODE & TOMBOL ====================
static void setMode(Mode next)
{
    if (next == mode) {
        return;
    }
    mode = next;
    resetVad();

    if (mode == MODE_SPEECH) {
        drainMic();
        showNotice("mode suara", 1200);
    } else {
        lastSignPollMs = 0;
        showNotice("mode isyarat", 1200);
    }
    uiDirty = true;
}

static void clearText()
{
    if (mode == MODE_SIGN) {
        signText = "";
        showNotice("teks dihapus", 1500);
        renderScreen();
        lv_refr_now(NULL);
        clearServerSign();
    } else {
        speechText = "";
        showNotice("teks dihapus", 1500);
    }
    uiDirty = true;
}

void button_event_callback(ButtonState state)
{
    switch (state) {
    case BTN_CLICK_EVENT:
        setMode(mode == MODE_SIGN ? MODE_SPEECH : MODE_SIGN);
        break;
    case BTN_DOUBLE_CLICK_EVENT:
        clearText();
        break;
    case BTN_LONG_PRESSED_EVENT:
        if (timerActive(statusOverlayUntil)) {
            statusOverlayUntil = 0;
        } else {
            statusOverlayUntil = millis() + STATUS_OVERLAY_MS;
        }
        uiDirty = true;
        break;
    case BTN_TRIPLE_CLICK_EVENT:
        showNotice("sambung ulang wifi", 2000);
        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        lastWifiAttemptMs = millis();
        break;
    default:
        break;
    }
}

// ==================== SETUP / LOOP ====================
void setup()
{
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.println("[BOOT] T-Glass BISINDO / Speech-to-text");

    if (!amoled.begin()) {
        while (true) {
            Serial.println("Board tidak terdeteksi. Naikkan Core Debug Level ke Error untuk detail.");
            delay(1000);
        }
    }

    // Orientasi layar: V1 memakai flip vertikal, V2 memakai rotasi 0.
#if TGLASS_VARIANT_V2
    amoled.setRotation(0);
#else
    amoled.flipHorizontal(true);
#endif
    amoled.setBrightness(SCREEN_BRIGHTNESS);
    amoled.setTouchThreshold(TOUCH_THRESHOLD);

    beginLvglHelper(amoled);
    buildUi();
    amoled.setEventCallback(button_event_callback);

    micReady = amoled.initMicrophone();
    if (micReady) {
        recordCapacitySamples = static_cast<size_t>(MIC_I2S_SAMPLE_RATE) * SPEECH_MAX_MS / 1000;
        recordBuf = static_cast<int16_t *>(ps_malloc(recordCapacitySamples * sizeof(int16_t)));
        if (recordBuf == nullptr) {
            Serial.println("[MIC] gagal alokasi buffer PSRAM");
            micReady = false;
        }
    }
    if (!micReady) {
        Serial.println("[MIC] mikrofon tidak siap; mode suara dinonaktifkan");
    } else {
        calibrateNoiseFloor();
    }

    startWifi();
    renderScreen();
}

void loop()
{
    amoled.update();
    lv_timer_handler();
    serviceWifi();

    if (statusOverlayUntil != 0 && !timerActive(statusOverlayUntil)) {
        statusOverlayUntil = 0;
        uiDirty = true;
    }
    if (noticeUntil != 0 && !timerActive(noticeUntil)) {
        noticeUntil = 0;
        uiDirty = true;
    }

    if (mode == MODE_SIGN) {
        if (wifiConnected && millis() - lastSignPollMs >= SIGN_POLL_INTERVAL_MS) {
            lastSignPollMs = millis();
            pollSign();
        }
        delay(5);
    } else {
        if (micReady) {
            serviceSpeech();
        } else {
            delay(20);
        }
    }

    if (uiDirty || (mode == MODE_SPEECH && speechActive &&
                    millis() - lastRenderMs >= LEVEL_RENDER_MS)) {
        uiDirty = false;
        lastRenderMs = millis();
        renderScreen();
    }
}
