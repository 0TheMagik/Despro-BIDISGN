/*
 * @file TGlass_BISINDO_test2.ino
 * @brief T-Glass V2 display for sign-recognition text and speech-to-text.
 *
 * Sign-text flow:
 *   ESP32-CAM -> Cloud recognition -> POST {"text":"..."} to this T-Glass
 *   T-Glass receives the result at POST /text and displays it in ISYARAT mode.
 *
 * Speech-to-text flow:
 *   Built-in PDM microphone -> voice activity detection -> PCM upload to
 *   Deployment/glass_bridge.py at POST /api/speech -> recognized text display.
 *
 * Touch control:
 *   single click : switch ISYARAT / SUARA
 *   double click : clear text for the current mode
 *   long press   : show Wi-Fi, speech bridge, and microphone status
 *   triple click : reconnect Wi-Fi
 *
 * Arduino IDE: ESP32S3 Dev Module, ESP32 Arduino core 2.0.5-2.0.17,
 * LilyGO T-Wristband and T-Glass library, LVGL 8.4.x, SensorLib,
 * ArduinoJson 6.x.
 */

#include <Arduino.h>
#include <LilyGo_Wristband.h>
#include <LV_Helper.h>
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <cmath>
#include <cstring>

// Configure the hotspot and the PC running Deployment/glass_bridge.py.
static const char *WIFI_SSID = "Topbgtdeh";
static const char *WIFI_PASSWORD = "W1f1mama";
static const char *SPEECH_BRIDGE_URL = "http://192.168.1.100:8765";
static const char *SPEECH_LANGUAGE = "id-ID";

constexpr int GLASS_VIEWABLE_WIDTH = 126;
constexpr int GLASS_VIEWABLE_HEIGHT = 126;
constexpr uint8_t SCREEN_BRIGHTNESS = 200;
constexpr uint16_t TOUCH_THRESHOLD = 200;
constexpr size_t SIGN_TEXT_LIMIT = 400;
constexpr size_t SPEECH_TEXT_LIMIT = 400;
constexpr size_t DISPLAY_TAIL_CHARS = 78;

constexpr uint32_t AUDIO_FRAME_MS = 30;
constexpr size_t AUDIO_FRAME_SAMPLES =
    MIC_I2S_SAMPLE_RATE / 1000 * AUDIO_FRAME_MS;
constexpr size_t AUDIO_FRAME_BYTES =
    AUDIO_FRAME_SAMPLES * sizeof(int16_t);
constexpr uint8_t SPEECH_START_FRAMES = 2;
constexpr uint32_t SPEECH_END_SILENCE_MS = 700;
constexpr uint32_t SPEECH_MIN_MS = 450;
constexpr uint32_t SPEECH_MAX_MS = 8000;
constexpr uint8_t PRE_ROLL_FRAMES = 8;
constexpr float MIN_TRIGGER_RMS = 420.0f;
constexpr float TRIGGER_FACTOR = 2.8f;
constexpr float RELEASE_FACTOR = 0.65f;

constexpr uint32_t WIFI_RETRY_MS = 5000;
constexpr uint32_t STATUS_OVERLAY_MS = 5000;
constexpr uint32_t NOTICE_MS = 1800;
constexpr uint32_t LEVEL_RENDER_MS = 100;
constexpr uint32_t BRIDGE_CONNECT_TIMEOUT_MS = 3000;
constexpr uint32_t BRIDGE_READ_TIMEOUT_MS = 12000;

constexpr uint32_t COLOR_SIGN = 0x00E5FF;
constexpr uint32_t COLOR_SPEECH = 0xFFA726;
constexpr uint32_t COLOR_OK = 0x69F0AE;
constexpr uint32_t COLOR_ERROR = 0xFF5252;
constexpr uint32_t COLOR_DIM = 0x9E9E9E;

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
LilyGo_Button bootPin;
WebServer server(80);
WiFiClient bridgeClient;

static Mode mode = MODE_SIGN;
static LinkState bridgeState = LINK_UNKNOWN;
static bool micReady = false;
static bool uploading = false;
static bool wifiWasConnected = false;
static bool uiDirty = true;

static String signText;
static String speechText;
static String noticeText;
static uint32_t noticeUntil = 0;
static uint32_t statusOverlayUntil = 0;
static uint32_t lastWiFiAttemptMs = 0;
static uint32_t lastRenderMs = 0;
static uint32_t lastMicDebugMs = 0;

static lv_obj_t *headerLabel = nullptr;
static lv_obj_t *statusLabel = nullptr;
static lv_obj_t *textLabel = nullptr;
static lv_obj_t *footerLabel = nullptr;
static lv_obj_t *levelBar = nullptr;
static const lv_font_t *bodyFont = nullptr;

static int16_t *recordBuffer = nullptr;
static size_t recordCapacitySamples = 0;
static size_t recordLengthSamples = 0;
static uint32_t recordStartMs = 0;
static int16_t audioFrame[AUDIO_FRAME_SAMPLES];
static int16_t preRoll[PRE_ROLL_FRAMES][AUDIO_FRAME_SAMPLES];
static uint8_t preRollIndex = 0;
static uint8_t preRollFilled = 0;
static bool speechActive = false;
static uint8_t aboveThresholdFrames = 0;
static uint32_t silenceMs = 0;
static float noiseFloor = 200.0f;
static uint8_t levelPercent = 0;

static bool timerActive(uint32_t deadline) {
  return deadline != 0 && static_cast<int32_t>(millis() - deadline) < 0;
}

static String tailOf(const String &text, size_t maxChars) {
  String trimmed = text;
  trimmed.trim();
  if (trimmed.length() <= maxChars) {
    return trimmed;
  }
  return "..." + trimmed.substring(trimmed.length() - maxChars);
}

static void showNotice(const String &text, uint32_t durationMs = NOTICE_MS) {
  noticeText = text;
  noticeUntil = millis() + durationMs;
  uiDirty = true;
}

static void setLabelText(lv_obj_t *label, const String &text) {
  if (label == nullptr) {
    return;
  }
  const char *current = lv_label_get_text(label);
  if (current == nullptr || !text.equals(current)) {
    lv_label_set_text(label, text.c_str());
  }
}

static void setTextColor(lv_obj_t *object, uint32_t color) {
  lv_obj_set_style_text_color(object, lv_color_hex(color), 0);
}

static void setBodyFont(const lv_font_t *font) {
  if (bodyFont != font) {
    bodyFont = font;
    lv_obj_set_style_text_font(textLabel, font, 0);
  }
}

static void buildUi() {
  lv_obj_set_style_bg_color(lv_scr_act(), lv_color_black(), 0);

  lv_obj_t *window = lv_obj_create(lv_scr_act());
  lv_obj_set_style_bg_color(window, lv_color_black(), 0);
  lv_obj_set_style_border_width(window, 0, 0);
  lv_obj_set_style_radius(window, 0, 0);
  lv_obj_set_style_pad_all(window, 0, 0);
  lv_obj_clear_flag(window, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_size(window, GLASS_VIEWABLE_WIDTH, GLASS_VIEWABLE_HEIGHT);
  lv_obj_align(window, LV_ALIGN_BOTTOM_MID, 0, 0);

  headerLabel = lv_label_create(window);
  lv_obj_set_style_text_font(headerLabel, &lv_font_montserrat_12, 0);
  lv_obj_set_pos(headerLabel, 3, 1);
  lv_label_set_text(headerLabel, "ISYARAT");
  setTextColor(headerLabel, COLOR_SIGN);

  statusLabel = lv_label_create(window);
  lv_obj_set_style_text_font(statusLabel, &lv_font_montserrat_12, 0);
  lv_obj_set_style_text_align(statusLabel, LV_TEXT_ALIGN_RIGHT, 0);
  lv_obj_set_width(statusLabel, 52);
  lv_obj_align(statusLabel, LV_ALIGN_TOP_RIGHT, -2, 1);
  lv_label_set_text(statusLabel, "WAIT");

  textLabel = lv_label_create(window);
  lv_obj_set_style_text_font(textLabel, &lv_font_montserrat_14, 0);
  lv_obj_set_style_text_color(textLabel, lv_color_white(), 0);
  lv_obj_set_style_text_align(textLabel, LV_TEXT_ALIGN_LEFT, 0);
  lv_label_set_long_mode(textLabel, LV_LABEL_LONG_WRAP);
  lv_obj_set_pos(textLabel, 3, 18);
  lv_obj_set_size(textLabel, GLASS_VIEWABLE_WIDTH - 6, 82);
  lv_label_set_text(textLabel, "Menunggu teks dari ESP32-CAM");
  bodyFont = &lv_font_montserrat_14;

  levelBar = lv_bar_create(window);
  lv_obj_set_size(levelBar, GLASS_VIEWABLE_WIDTH - 8, 5);
  lv_obj_set_pos(levelBar, 4, 104);
  lv_bar_set_range(levelBar, 0, 100);
  lv_bar_set_value(levelBar, 0, LV_ANIM_OFF);
  lv_obj_set_style_bg_color(levelBar, lv_color_hex(0x263238), LV_PART_MAIN);
  lv_obj_set_style_bg_color(levelBar, lv_color_hex(COLOR_SPEECH),
                            LV_PART_INDICATOR);
  lv_obj_add_flag(levelBar, LV_OBJ_FLAG_HIDDEN);

  footerLabel = lv_label_create(window);
  lv_obj_set_style_text_font(footerLabel, &lv_font_montserrat_12, 0);
  lv_obj_set_style_text_align(footerLabel, LV_TEXT_ALIGN_CENTER, 0);
  lv_label_set_long_mode(footerLabel, LV_LABEL_LONG_CLIP);
  lv_obj_set_width(footerLabel, GLASS_VIEWABLE_WIDTH);
  lv_obj_set_pos(footerLabel, 0, 112);
  lv_label_set_text(footerLabel, "klik: mode | 2x: hapus");
}

static String statusOverlayText() {
  String info;
  info.reserve(120);
  if (WiFi.status() == WL_CONNECTED) {
    info += "WiFi: ";
    info += WiFi.localIP().toString();
    info += "\nRSSI: ";
    info += String(WiFi.RSSI());
    info += " dBm";
  } else {
    info = "WiFi: terputus";
  }
  info += "\nBridge: ";
  if (bridgeState == LINK_OK) {
    info += "OK";
  } else if (bridgeState == LINK_ERROR) {
    info += "GAGAL";
  } else {
    info += "belum dites";
  }
  info += "\nMic: ";
  info += micReady ? "OK" : "GAGAL";
  info += "\nMode: ";
  info += mode == MODE_SIGN ? "ISYARAT" : "SUARA";
  return info;
}

static void renderScreen() {
  const bool showStatus = timerActive(statusOverlayUntil);
  const bool signMode = mode == MODE_SIGN;
  uint32_t statusColor = COLOR_OK;
  String status;

  if (WiFi.status() != WL_CONNECTED) {
    statusColor = COLOR_DIM;
    status = "WIFI";
  } else if (uploading || speechActive) {
    statusColor = COLOR_ERROR;
    status = uploading ? "TX" : "REC";
  } else if (signMode) {
    status = signText.length() ? "TEXT" : "WAIT";
  } else if (!micReady || bridgeState == LINK_ERROR) {
    statusColor = COLOR_ERROR;
    status = "ERR";
  } else {
    status = bridgeState == LINK_OK ? "OK" : "READY";
  }

  setTextColor(statusLabel, statusColor);
  if (showStatus) {
    setLabelText(headerLabel, "STATUS");
    setTextColor(headerLabel, COLOR_DIM);
    setLabelText(statusLabel, "");
    setBodyFont(&lv_font_montserrat_12);
    setLabelText(textLabel, statusOverlayText());
  } else if (signMode) {
    setLabelText(headerLabel, "ISYARAT");
    setTextColor(headerLabel, COLOR_SIGN);
    setLabelText(statusLabel, status);
    setBodyFont(&lv_font_montserrat_14);
    String body = tailOf(signText, DISPLAY_TAIL_CHARS);
    if (body.length() == 0) {
      body = WiFi.status() == WL_CONNECTED
                 ? "Menunggu teks dari ESP32-CAM"
                 : "Menghubungkan Wi-Fi";
    }
    setLabelText(textLabel, body);
  } else {
    setLabelText(headerLabel, "SUARA");
    setTextColor(headerLabel, COLOR_SPEECH);
    setLabelText(statusLabel, status);
    setBodyFont(&lv_font_montserrat_14);
    String body = tailOf(speechText, DISPLAY_TAIL_CHARS);
    if (body.length() == 0) {
      body = micReady ? "Bicara ke mikrofon" : "Mikrofon tidak aktif";
    }
    setLabelText(textLabel, body);
  }

  if (signMode) {
    lv_obj_add_flag(levelBar, LV_OBJ_FLAG_HIDDEN);
  } else {
    lv_obj_clear_flag(levelBar, LV_OBJ_FLAG_HIDDEN);
    lv_bar_set_value(levelBar, levelPercent, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(
        levelBar,
        lv_color_hex(speechActive ? COLOR_ERROR : COLOR_SPEECH),
        LV_PART_INDICATOR);
  }

  String footer;
  if (timerActive(noticeUntil)) {
    footer = noticeText;
  } else if (showStatus) {
    footer = "Tekan lama: tutup";
  } else if (mode == MODE_SIGN) {
    footer = "Klik: mode | 2x: hapus";
  } else if (uploading) {
    footer = "Mengirim audio ke bridge";
  } else if (speechActive) {
    footer = "Merekam suara...";
  } else {
    footer = "Klik: mode | 2x: hapus";
  }
  setLabelText(footerLabel, footer);
}

static String bridgeUrl(const String &path) {
  String base = SPEECH_BRIDGE_URL;
  base.trim();
  while (base.endsWith("/")) {
    base.remove(base.length() - 1);
  }
  return base + path;
}

static void handleText() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json",
                "{\"error\":\"missing JSON body\"}");
    return;
  }

  const String body = server.arg("plain");
  StaticJsonDocument<512> requestDoc;
  DeserializationError error = deserializeJson(requestDoc, body);
  if (error) {
    server.send(400, "application/json",
                "{\"error\":\"invalid JSON\"}");
    return;
  }
  if (!requestDoc["text"].is<const char *>()) {
    server.send(400, "application/json",
                "{\"error\":\"text must be a string\"}");
    return;
  }

  String text = requestDoc["text"].as<String>();
  text.trim();
  if (text.length() > SIGN_TEXT_LIMIT) {
    server.send(413, "application/json",
                "{\"error\":\"text is too long\"}");
    return;
  }

  signText = text;
  Serial.printf("[ISYARAT] %s\n", signText.length() ? signText.c_str() : "(kosong)");
  uiDirty = true;
  server.send(200, "application/json", "{\"status\":\"ok\"}");
}

static void handleHealth() {
  server.send(200, "application/json",
              "{\"status\":\"ok\",\"device\":\"tglass-v2\"}");
}

static void startWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWiFiAttemptMs = millis();
  Serial.printf("[Wi-Fi] Menghubungkan ke hotspot \"%s\"...\n", WIFI_SSID);
}

static void serviceWiFi() {
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (connected != wifiWasConnected) {
    wifiWasConnected = connected;
    uiDirty = true;
    if (connected) {
      Serial.printf("[Wi-Fi] Tersambung. IP T-Glass: %s, RSSI: %d dBm\n",
                    WiFi.localIP().toString().c_str(), WiFi.RSSI());
      bridgeState = LINK_UNKNOWN;
    } else {
      Serial.println("[Wi-Fi] Koneksi terputus.");
      bridgeState = LINK_UNKNOWN;
    }
  }

  if (!connected && millis() - lastWiFiAttemptMs >= WIFI_RETRY_MS) {
    lastWiFiAttemptMs = millis();
    Serial.println("[Wi-Fi] Mencoba menyambung ulang...");
    WiFi.reconnect();
  }
}

static float computeRms(const int16_t *samples, size_t count) {
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

static uint8_t rmsToLevel(float rms, float trigger) {
  const float low = fmaxf(trigger, 1.0f);
  const float high = low * 6.0f;
  float percent = (rms - low) / (high - low) * 100.0f;
  if (percent < 0.0f) {
    percent = 0.0f;
  } else if (percent > 100.0f) {
    percent = 100.0f;
  }
  return static_cast<uint8_t>(percent);
}

static void appendSamples(const int16_t *source, size_t count) {
  if (recordBuffer == nullptr ||
      recordLengthSamples >= recordCapacitySamples) {
    return;
  }
  const size_t room = recordCapacitySamples - recordLengthSamples;
  if (count > room) {
    count = room;
  }
  memcpy(recordBuffer + recordLengthSamples, source,
         count * sizeof(int16_t));
  recordLengthSamples += count;
}

static void resetVad() {
  speechActive = false;
  aboveThresholdFrames = 0;
  silenceMs = 0;
  recordLengthSamples = 0;
  preRollIndex = 0;
  preRollFilled = 0;
  levelPercent = 0;
}

static void drainMicrophone() {
  uint8_t scratch[256];
  size_t readLength = 0;
  for (uint8_t i = 0; i < 64; ++i) {
    if (!amoled.readMicrophone(scratch, sizeof(scratch), &readLength, 0) ||
        readLength == 0) {
      break;
    }
  }
}

static void startRecording() {
  speechActive = true;
  recordLengthSamples = 0;
  silenceMs = 0;
  recordStartMs = millis();

  const size_t start =
      preRollFilled < PRE_ROLL_FRAMES ? 0 : preRollIndex;
  for (size_t i = 0; i < preRollFilled; ++i) {
    appendSamples(preRoll[(start + i) % PRE_ROLL_FRAMES],
                  AUDIO_FRAME_SAMPLES);
  }
  Serial.printf("[Mic] Mulai merekam; pre-roll %u frame.\n",
                static_cast<unsigned>(preRollFilled));
  uiDirty = true;
}

static bool uploadSpeech(const int16_t *samples, size_t sampleCount,
                         String &recognizedText, String &errorText) {
  String path = "/api/speech?rate=";
  path += String(MIC_I2S_SAMPLE_RATE);
  path += "&width=2&lang=";
  path += SPEECH_LANGUAGE;

  HTTPClient http;
  http.setConnectTimeout(BRIDGE_CONNECT_TIMEOUT_MS);
  http.setTimeout(BRIDGE_READ_TIMEOUT_MS);
  if (!http.begin(bridgeClient, bridgeUrl(path))) {
    errorText = "bridge tidak dapat dibuka";
    return false;
  }

  http.addHeader("Content-Type", "application/octet-stream");
  const size_t byteLength = sampleCount * sizeof(int16_t);
  const int status = http.POST(
      reinterpret_cast<uint8_t *>(const_cast<int16_t *>(samples)),
      byteLength);
  const String responseBody = http.getString();
  http.end();

  if (status != 200) {
    bridgeState = LINK_ERROR;
    errorText = responseBody.length()
                    ? responseBody
                    : "HTTP " + String(status);
    return false;
  }

  bridgeState = LINK_OK;
  recognizedText = responseBody;
  recognizedText.trim();
  return true;
}

static void finishRecording() {
  speechActive = false;
  aboveThresholdFrames = 0;
  preRollIndex = 0;
  preRollFilled = 0;
  levelPercent = 0;

  const size_t sampleCount = recordLengthSamples;
  recordLengthSamples = 0;
  const uint32_t durationMs = static_cast<uint32_t>(
      static_cast<uint64_t>(sampleCount) * 1000ULL /
      MIC_I2S_SAMPLE_RATE);

  if (sampleCount == 0) {
    uiDirty = true;
    return;
  }
  if (durationMs < SPEECH_MIN_MS) {
    showNotice("Ucapan terlalu singkat");
    return;
  }
  if (WiFi.status() != WL_CONNECTED) {
    showNotice("Wi-Fi terputus");
    return;
  }

  uploading = true;
  uiDirty = true;
  renderScreen();
  lv_refr_now(nullptr);
  Serial.printf("[Mic] Mengirim audio %.1f detik ke speech bridge.\n",
                durationMs / 1000.0f);

  String recognizedText;
  String errorText;
  const bool ok =
      uploadSpeech(recordBuffer, sampleCount, recognizedText, errorText);
  uploading = false;

  if (!ok) {
    Serial.printf("[Speech] Gagal: %s\n", errorText.c_str());
    showNotice("Bridge: " + errorText.substring(0, 20), 3000);
  } else if (recognizedText.length() == 0) {
    showNotice("Ucapan tidak dikenali");
  } else {
    if (speechText.length() > 0) {
      speechText += ' ';
    }
    speechText += recognizedText;
    if (speechText.length() > SPEECH_TEXT_LIMIT) {
      speechText.remove(0, speechText.length() - SPEECH_TEXT_LIMIT);
    }
    Serial.printf("[Speech] %s\n", recognizedText.c_str());
    showNotice("Speech-to-text berhasil");
  }

  drainMicrophone();
  resetVad();
  uiDirty = true;
}

static void processAudioFrame(const int16_t *samples, size_t count) {
  const float rms = computeRms(samples, count);
  const float trigger = fmaxf(noiseFloor * TRIGGER_FACTOR,
                              MIN_TRIGGER_RMS);

  if (!speechActive) {
    if (rms < trigger) {
      noiseFloor = noiseFloor * 0.97f + rms * 0.03f;
    }

    memcpy(preRoll[preRollIndex], samples, AUDIO_FRAME_BYTES);
    preRollIndex = (preRollIndex + 1) % PRE_ROLL_FRAMES;
    if (preRollFilled < PRE_ROLL_FRAMES) {
      ++preRollFilled;
    }

    if (rms > trigger) {
      ++aboveThresholdFrames;
      if (aboveThresholdFrames >= SPEECH_START_FRAMES) {
        startRecording();
      }
    } else {
      aboveThresholdFrames = 0;
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
        recordLengthSamples >= recordCapacitySamples) {
      finishRecording();
    }
  }

  if (millis() - lastMicDebugMs >= 500) {
    lastMicDebugMs = millis();
    Serial.printf("[Mic] RMS=%.0f noise=%.0f threshold=%.0f%s\n",
                  rms, noiseFloor, trigger,
                  speechActive ? " (merekam)" : "");
  }
}

static void serviceSpeech() {
  size_t readLength = 0;
  const TickType_t timeout = pdMS_TO_TICKS(AUDIO_FRAME_MS + 20);
  if (!amoled.readMicrophone(audioFrame, AUDIO_FRAME_BYTES,
                             &readLength, timeout)) {
    return;
  }
  if (readLength != AUDIO_FRAME_BYTES) {
    return;
  }
  processAudioFrame(audioFrame, AUDIO_FRAME_SAMPLES);
}

static void calibrateNoiseFloor() {
  float sum = 0.0f;
  uint8_t frames = 0;
  const uint32_t started = millis();
  size_t readLength = 0;

  while (millis() - started < 400 && frames < 16) {
    if (amoled.readMicrophone(
            audioFrame, AUDIO_FRAME_BYTES, &readLength,
            pdMS_TO_TICKS(AUDIO_FRAME_MS + 20)) &&
        readLength == AUDIO_FRAME_BYTES) {
      sum += computeRms(audioFrame, AUDIO_FRAME_SAMPLES);
      ++frames;
    }
  }

  noiseFloor = frames ? fmaxf(sum / frames, 80.0f) : 200.0f;
  Serial.printf("[Mic] Noise floor %.0f dari %u frame.\n",
                noiseFloor, static_cast<unsigned>(frames));
}

static void setMode(Mode next) {
  if (next == mode) {
    return;
  }
  mode = next;
  resetVad();
  if (mode == MODE_SPEECH) {
    drainMicrophone();
    showNotice(micReady ? "Mode SUARA" : "Mikrofon tidak aktif");
  } else {
    showNotice("Mode ISYARAT");
  }
  uiDirty = true;
}

static void clearCurrentText() {
  if (mode == MODE_SIGN) {
    signText = "";
    showNotice("Teks isyarat dihapus");
  } else {
    speechText = "";
    showNotice("Teks suara dihapus");
  }
  uiDirty = true;
}

void button_event_callback(ButtonState state) {
  switch (state) {
    case BTN_PRESSED_EVENT:
      Serial.println("[Tombol] pressed");
      break;
    case BTN_RELEASED_EVENT:
      Serial.println("[Tombol] released");
      break;
    case BTN_CLICK_EVENT:
      setMode(mode == MODE_SIGN ? MODE_SPEECH : MODE_SIGN);
      Serial.printf("[Tombol] Mode %s\n",
                    mode == MODE_SIGN ? "ISYARAT" : "SUARA");
      break;
    case BTN_DOUBLE_CLICK_EVENT:
      clearCurrentText();
      Serial.println("[Tombol] Teks mode aktif dihapus.");
      break;
    case BTN_LONG_PRESSED_EVENT:
      if (timerActive(statusOverlayUntil)) {
        statusOverlayUntil = 0;
      } else {
        statusOverlayUntil = millis() + STATUS_OVERLAY_MS;
      }
      uiDirty = true;
      Serial.println("[Tombol] Status ditampilkan/disembunyikan.");
      break;
    case BTN_TRIPLE_CLICK_EVENT:
      showNotice("Menyambung ulang Wi-Fi");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      lastWiFiAttemptMs = millis();
      Serial.println("[Tombol] Memulai ulang koneksi Wi-Fi.");
      break;
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("[BOOT] T-Glass V2 BISINDO display + speech-to-text");

  if (!amoled.begin()) {
    Serial.println("[T-Glass] Inisialisasi board gagal.");
    while (true) {
      delay(1000);
    }
  }

  amoled.setRotation(0);
  amoled.setBrightness(SCREEN_BRIGHTNESS);
  amoled.setTouchThreshold(TOUCH_THRESHOLD);
  beginLvglHelper(amoled);
  buildUi();

  bootPin.init(BOARD_BOOT_PIN);
  bootPin.setEventCallback(button_event_callback);
  amoled.setEventCallback(button_event_callback);

  micReady = amoled.initMicrophone();
  if (micReady) {
    recordCapacitySamples =
        static_cast<size_t>(MIC_I2S_SAMPLE_RATE) * SPEECH_MAX_MS / 1000;
    recordBuffer = static_cast<int16_t *>(
        ps_malloc(recordCapacitySamples * sizeof(int16_t)));
    if (recordBuffer == nullptr) {
      Serial.println("[Mic] Alokasi buffer PSRAM gagal.");
      micReady = false;
    }
  }

  if (micReady) {
    calibrateNoiseFloor();
  } else {
    Serial.println("[Mic] Tidak siap; mode SUARA tidak tersedia.");
  }

  startWiFi();
  server.on("/health", HTTP_GET, handleHealth);
  server.on("/text", HTTP_POST, handleText);
  server.begin();
  Serial.println("[HTTP] Endpoint /health dan /text aktif pada port 80.");
  renderScreen();
}

void loop() {
  bootPin.update();
  delay(1);
  amoled.update();
  lv_timer_handler();
  server.handleClient();
  serviceWiFi();

  if (statusOverlayUntil != 0 && !timerActive(statusOverlayUntil)) {
    statusOverlayUntil = 0;
    uiDirty = true;
  }
  if (noticeUntil != 0 && !timerActive(noticeUntil)) {
    noticeUntil = 0;
    uiDirty = true;
  }

  if (mode == MODE_SPEECH && micReady) {
    serviceSpeech();
  } else {
    delay(5);
  }

  if (uiDirty ||
      (mode == MODE_SPEECH && speechActive &&
       millis() - lastRenderMs >= LEVEL_RENDER_MS)) {
    uiDirty = false;
    lastRenderMs = millis();
    renderScreen();
  }
}
