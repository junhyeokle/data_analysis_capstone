#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

Adafruit_MPU6050 mpu;

#define WIFI_SSID      "000000"
#define WIFI_PASSWORD  "dlwnsgur00"
#define SERVER_URL     "http://3.27.105.83:8000"

#define SDA_PIN 27
#define SCL_PIN 14

#define SAMPLE_INTERVAL_MS 10
#define WINDOW_SIZE 200
#define N_CHANNELS 3
#define N_CLASSES 3

#define DANGER_THRESHOLD 0.80f
#define NORMAL_THRESHOLD 0.25f

#define TENSOR_ARENA_SIZE (56 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::AllOpsResolver tf_resolver;
static tflite::MicroErrorReporter micro_error_reporter;

const tflite::Model*      tf_model       = nullptr;
tflite::MicroInterpreter* tf_interpreter = nullptr;
TfLiteTensor*             tf_input       = nullptr;
TfLiteTensor*             tf_output      = nullptr;

float window_buf[WINDOW_SIZE][N_CHANNELS];
int           window_count   = 0;
int           window_id      = 0;
String        session_id     = "";
unsigned long last_sample_ms = 0;

void preprocess_window(float buf[][N_CHANNELS]) {
  for (int ch = 0; ch < N_CHANNELS; ch++) {
    float sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) sum += buf[i][ch];
    float mean = sum / WINDOW_SIZE;

    float sq_sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      buf[i][ch] -= mean;
      sq_sum += buf[i][ch] * buf[i][ch];
    }

    float std_dev = sqrtf(sq_sum / WINDOW_SIZE);
    if (std_dev < 1e-6f) std_dev = 1e-6f;
    for (int i = 0; i < WINDOW_SIZE; i++) buf[i][ch] /= std_dev;
  }
}

bool run_inference(float input_data[][N_CHANNELS], float softmax[N_CLASSES]) {
  int idx = 0;
  for (int i = 0; i < WINDOW_SIZE; i++)
    for (int ch = 0; ch < N_CHANNELS; ch++)
      tf_input->data.f[idx++] = input_data[i][ch];

  if (tf_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[TFLite] Invoke failed"); Serial.flush();
    return false;
  }

  for (int i = 0; i < N_CLASSES; i++) softmax[i] = tf_output->data.f[i];
  return true;
}

void send_result(float softmax[N_CLASSES], float raw_window[][N_CHANNELS], int pred) {
  if (WiFi.status() != WL_CONNECTED) return;

  StaticJsonDocument<256> doc;
  doc["device_id"]    = session_id;
  doc["window_id"]    = window_id;
  doc["is_confident"] = true;

  JsonArray result_arr = doc.createNestedArray("result");
  for (int i = 0; i < N_CLASSES; i++) result_arr.add(softmax[i]);

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.begin(String(SERVER_URL) + "/result");
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  Serial.printf("[/result] HTTP %d\n", code);
  if (code > 0) Serial.println(http.getString());
  http.end();
}

void send_refine(float raw_window[][N_CHANNELS], float danger_prob) {
  if (WiFi.status() != WL_CONNECTED) return;

  String body;
  body.reserve(5500);
  body  = "{\"device_id\":\"" + session_id + "\","
          "\"window_id\":"    + window_id  + ","
          "\"is_confident\":false,"
          "\"result\":\"주의\","
          "\"confidence\":"   + String(danger_prob, 4) + ","
          "\"readings\":[";

  char tmp[64];
  for (int i = 0; i < WINDOW_SIZE; i++) {
    snprintf(tmp, sizeof(tmp), "{\"t\":%.2f,\"x\":%.4f,\"y\":%.4f,\"z\":%.4f}",
             i * 0.01f, raw_window[i][0], raw_window[i][1], raw_window[i][2]);
    body += tmp;
    if (i < WINDOW_SIZE - 1) body += ',';
  }
  body += "]}";

  HTTPClient http;
  http.begin(String(SERVER_URL) + "/refine");
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  int code = http.POST(body);
  Serial.printf("[/refine] HTTP %d\n", code); Serial.flush();
  if (code > 0) Serial.println(http.getString());
  http.end();
}

void process_window() {
  Serial.printf("\n[WINDOW %d] collected\n", window_id); Serial.flush();

  float processed[WINDOW_SIZE][N_CHANNELS];
  memcpy(processed, window_buf, sizeof(window_buf));
  preprocess_window(processed);

  float softmax[N_CLASSES];
  if (!run_inference(processed, softmax)) return;

  Serial.printf("Softmax [정상, 주의, 위험] = [%.3f, %.3f, %.3f]\n",
                softmax[0], softmax[1], softmax[2]); Serial.flush();

  // argmax로 최고 확률 클래스 결정
  int pred = 0;
  for (int i = 1; i < N_CLASSES; i++)
    if (softmax[i] > softmax[pred]) pred = i;

  float confidence = softmax[pred];

  if (confidence >= 0.60f) {
    // 60% 이상 확신할 때는 바로 판정
    if (pred == 2)      Serial.println("판정: 위험");
    else if (pred == 1) Serial.println("판정: 주의");
    else                Serial.println("판정: 정상");
    send_result(softmax, window_buf, pred);
  } else {
    // 어떤 클래스도 60% 미만일 때만 refine
    Serial.println("판정: 애매함 → /refine");
    send_refine(window_buf, softmax[2]);
  }

  window_id++;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== Boot ==="); Serial.flush();

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!mpu.begin()) {
    Serial.println("[MPU6050] not found"); Serial.flush();
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("[MPU6050] ready"); Serial.flush();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println();
  Serial.print("[WiFi] IP: "); Serial.println(WiFi.localIP()); Serial.flush();

  session_id = "esp32_" + String(millis());

  Serial.printf("[Heap] free: %d\n", ESP.getFreeHeap()); Serial.flush();

  Serial.println("[TF] 1. GetModel"); Serial.flush();
  tf_model = tflite::GetModel(__1d_cnn_model_4_tflite);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TF] schema mismatch!"); Serial.flush();
    while (1) delay(10);
  }

  Serial.println("[TF] 2. new Interpreter"); Serial.flush();
  tf_interpreter = new tflite::MicroInterpreter(
    tf_model, tf_resolver, tensor_arena, TENSOR_ARENA_SIZE,
    &micro_error_reporter
  );

  Serial.println("[TF] 3. AllocateTensors"); Serial.flush();
  if (tf_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TF] AllocateTensors FAILED"); Serial.flush();
    while (1) delay(10);
  }

  tf_input  = tf_interpreter->input(0);
  tf_output = tf_interpreter->output(0);

  Serial.printf("[TF] OK. arena used: %u / %u\n",
                tf_interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE); Serial.flush();

  Serial.print("[TF] input shape: ");
  for (int i = 0; i < tf_input->dims->size; i++) {
    Serial.print(tf_input->dims->data[i]); Serial.print(" ");
  }
  Serial.println(); Serial.flush();

  Serial.printf("[Heap] free: %d\n", ESP.getFreeHeap()); Serial.flush();
  Serial.println("=== Start ==="); Serial.flush();
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_ms < SAMPLE_INTERVAL_MS) return;
  last_sample_ms = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  window_buf[window_count][0] = a.acceleration.x;
  window_buf[window_count][1] = a.acceleration.y;
  window_buf[window_count][2] = a.acceleration.z;
  window_count++;

  if (window_count >= WINDOW_SIZE) {
    process_window();

    const int slide = WINDOW_SIZE / 2;
    for (int i = 0; i < WINDOW_SIZE - slide; i++) {
      window_buf[i][0] = window_buf[i + slide][0];
      window_buf[i][1] = window_buf[i + slide][1];
      window_buf[i][2] = window_buf[i + slide][2];
    }
    window_count = WINDOW_SIZE - slide;
  }
}
