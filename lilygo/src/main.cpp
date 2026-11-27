/**
 * BISINDO Smart Glasses — firmware entry point
 * Board : LILYGO T-Glass (ESP32-S3 FN4R2, AMOLED 1.1" JD9613 294x126)
 *
 * Yang sudah jalan di file ini:
 *   - inisialisasi board + AMOLED + LVGL
 *   - UI 126x126 (area yang terlihat lewat kacamata)
 *   - WiFi STA
 *   - meter level mikrofon PDM
 *   - indikator baterai
 *
 * Belum ada (butuh hardware tambahan, lihat README):
 *   - kamera OV2640 (T-Camera-S3) untuk capture frame isyarat
 *   - streaming/WS ke server inferensi di Deployment/
 */

#include <Arduino.h>
#include <WiFi.h>
#include <math.h>

#include <LilyGo_Wristband.h>
#include <LV_Helper.h>

#include "config.h"

LilyGo_Class amoled;

static lv_obj_t *stateLabel = nullptr;
static lv_obj_t *netLabel = nullptr;
static lv_obj_t *battLabel = nullptr;
static lv_obj_t *micBar = nullptr;

static int16_t micBuffer[MIC_FRAME_SAMPLES];
static bool micReady = false;
static uint32_t lastBatteryUpdate = 0;

static void buildUi() {
    lv_obj_set_style_bg_color(lv_scr_act(), lv_color_black(), 0);

    // Jendela konten: 126x126, menempel di tengah-bawah panel 294x126
    lv_obj_t *window = lv_obj_create(lv_scr_act());
    lv_obj_set_style_border_width(window, 0, 0);
    lv_obj_set_style_bg_color(window, lv_color_black(), 0);
    lv_obj_set_size(window, GLASS_VIEW_W, GLASS_VIEW_H);
    lv_obj_align(window, LV_ALIGN_BOTTOM_MID, 0, 0);

    lv_obj_t *title = lv_label_create(window);
    lv_obj_set_style_text_color(title, lv_color_hex(0x00ffff), LV_PART_MAIN);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_14, LV_PART_MAIN);
    lv_label_set_text(title, "BISINDO");
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 4);

    stateLabel = lv_label_create(window);
    lv_obj_set_style_text_color(stateLabel, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_text_font(stateLabel, &lv_font_montserrat_10, LV_PART_MAIN);
    lv_label_set_text(stateLabel, "init...");
    lv_obj_align(stateLabel, LV_ALIGN_TOP_MID, 0, 28);

    netLabel = lv_label_create(window);
    lv_obj_set_style_text_color(netLabel, lv_color_hex(0x9e9e9e), LV_PART_MAIN);
    lv_obj_set_style_text_font(netLabel, &lv_font_montserrat_10, LV_PART_MAIN);
    lv_label_set_text(netLabel, "WiFi: --");
    lv_obj_align(netLabel, LV_ALIGN_TOP_MID, 0, 46);

    battLabel = lv_label_create(window);
    lv_obj_set_style_text_color(battLabel, lv_color_hex(0x9e9e9e), LV_PART_MAIN);
    lv_obj_set_style_text_font(battLabel, &lv_font_montserrat_10, LV_PART_MAIN);
    lv_label_set_text(battLabel, "Batt: --%");
    lv_obj_align(battLabel, LV_ALIGN_TOP_MID, 0, 62);

    micBar = lv_bar_create(window);
    lv_obj_set_size(micBar, 100, 8);
    lv_obj_align(micBar, LV_ALIGN_BOTTOM_MID, 0, -8);
    lv_bar_set_value(micBar, 0, LV_ANIM_OFF);
}

static void connectWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < WIFI_TIMEOUT_MS) {
        lv_label_set_text(stateLabel, "WiFi...");
        lv_timer_handler();
        delay(100);
    }

    if (WiFi.status() == WL_CONNECTED) {
        lv_label_set_text(stateLabel, "ready");
        lv_label_set_text_fmt(netLabel, "WiFi: %s", WiFi.SSID().c_str());
        Serial.printf("[BISINDO] WiFi up: %s (%s)\n", WiFi.SSID().c_str(),
                      WiFi.localIP().toString().c_str());
    } else {
        lv_label_set_text(stateLabel, "no wifi");
        lv_label_set_text(netLabel, "WiFi: fail");
        Serial.println("[BISINDO] WiFi timeout");
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println();
    Serial.println("[BISINDO] T-Glass booting...");

    if (!amoled.begin()) {
        Serial.println("[BISINDO] board not detected, check platformio.ini board config");
        while (true) {
            delay(1000);
        }
    }

    amoled.setRotation(0);
    amoled.setBrightness(180);

    beginLvglHelper(amoled);
    buildUi();
    lv_timer_handler();

    connectWifi();

    micReady = amoled.initMicrophone();
    Serial.println(micReady ? "[BISINDO] microphone ready"
                            : "[BISINDO] microphone init failed");
}

void loop() {
    amoled.update();
    lv_timer_handler();

    if (micReady) {
        size_t readLen = 0;
        if (amoled.readMicrophone(micBuffer, sizeof(micBuffer), &readLen)) {
            int samples = (int)(readLen / sizeof(int16_t));
            if (samples > 0) {
                int64_t acc = 0;
                for (int i = 0; i < samples; i++) {
                    acc += (int64_t)micBuffer[i] * micBuffer[i];
                }
                float rms = sqrtf((float)acc / (float)samples);
                lv_bar_set_value(micBar, constrain((int)(rms / 2.0f), 0, 100), LV_ANIM_OFF);
            }
        }
    }

    if (millis() - lastBatteryUpdate >= 5000) {
        lastBatteryUpdate = millis();
        lv_label_set_text_fmt(battLabel, "Batt: %d%%", amoled.getBatteryPercent());
    }

    delay(5);
}
