#include <Arduino.h>
#include <Wire.h>
#include <DFRobot_BMI160.h>
#include <WiFi.h>
#include <FirebaseESP32.h>
#include <esp_task_wdt.h>

// =============================================================
//  WIFI & FIREBASE CONFIGURATION
// =============================================================
#define WIFI_SSID "Your WiFi SSID"
#define WIFI_PASSWORD "Your WiFi Password"

#define FIREBASE_HOST "respiratory-system-monitor-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH "Firebase Database Secret or API Key"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
TaskHandle_t FirebaseTaskHandle;

// =============================================================
//  SENSOR CONFIGURATION
// =============================================================
#define I2C_SDA 21
#define I2C_SCL 22

#define ADDR_CHEST 0x68 // Primary Abdomen Sensor
#define ADDR_BACK  0x69 // Reference Back Sensor

#define SAMPLE_RATE_HZ  25
#define SAMPLE_INTERVAL (1000 / SAMPLE_RATE_HZ)

// ── Motion Gating ─────────────────────────────────────────
#define MOTION_THRESHOLD  800.0f 
#define MOTION_LOCKOUT_MS 3000

#define PRINT_INTERVAL    1000

// CRITICAL FIX: Turn off 25Hz raw data printing to prevent UART overflow
// and garbled text while running standalone over WiFi.
#define ENABLE_TELEPLOT   0 

// =============================================================
//  POWER / WIFI STABILITY SETTINGS
// =============================================================
// If your PCB power rail cannot supply full peak current, lowering
// TX power reduces current spikes that cause brownout resets.
// 0 = auto/full power (try this first once hardware power is fixed),
// or use WIFI_POWER_15dBm / WIFI_POWER_11dBm to cut peak draw.
#define REDUCE_WIFI_TX_POWER 1

#define WIFI_RECONNECT_INTERVAL_MS 5000
#define FIREBASE_RETRY_INTERVAL_MS 3000

// How often the status fields (state, progress %, uptime, etc.) get pushed
// to Firebase. Kept separate from the 2s rate/motion upload cadence since
// status changes less often and we don't want to spam the DB.
#define STATUS_UPLOAD_INTERVAL_MS 2000

// =============================================================
//  SYSTEM STATUS TRACKING (for Firebase progress reporting)
// =============================================================
enum SystemState {
  STATE_BOOTING,           // setup() running
  STATE_WIFI_CONNECTING,   // waiting for WiFi
  STATE_SENSOR_INIT,       // initializing BMI160s
  STATE_SENSOR_FAILED,     // sensor init failed, retrying in background
  STATE_GATHERING_WINDOW,  // running, peak-detection buffer still filling
  STATE_RUNNING,           // fully operational, RR values valid
  STATE_MOTION_GATED       // running but currently motion-gated
};

const char* stateToString(SystemState s) {
  switch (s) {
    case STATE_BOOTING:          return "booting";
    case STATE_WIFI_CONNECTING:  return "wifi_connecting";
    case STATE_SENSOR_INIT:      return "sensor_init";
    case STATE_SENSOR_FAILED:    return "sensor_failed";
    case STATE_GATHERING_WINDOW: return "gathering_window";
    case STATE_RUNNING:          return "running";
    case STATE_MOTION_GATED:     return "motion_gated";
    default:                     return "unknown";
  }
}

volatile SystemState systemState = STATE_BOOTING;
volatile int windowProgressPercent = 0; // 0-100, how full the RR peak-detection buffer is

// =============================================================
//  BIQUAD STRUCT (Base element for filters)
// =============================================================
struct Biquad {
  float b0, b1, b2, a1, a2;
  float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  
  Biquad(float _b0, float _b1, float _b2, float _a1, float _a2)
      : b0(_b0), b1(_b1), b2(_b2), a1(_a1), a2(_a2) {}

  float process(float x) {
    float y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
    x2 = x1; x1 = x;
    y2 = y1; y1 = y;
    return y;
  }
  void reset() { x1=x2=y1=y2=0; }
};

// =============================================================
//  STAGE 1: 6th-Order Butterworth BP + Notch + Savitzky-Golay
// =============================================================
struct BPFilter6 {
  Biquad s1 = Biquad( 0.00057f,  0.00000f, -0.00057f, -1.98942f,  0.98944f);
  Biquad s2 = Biquad( 0.00057f,  0.00000f, -0.00057f, -1.97128f,  0.97290f);
  Biquad s3 = Biquad( 0.00057f,  0.00000f, -0.00057f, -1.95524f,  0.95810f);
  
  float process(float x) { return s3.process(s2.process(s1.process(x))); }
  void reset() { s1.reset(); s2.reset(); s3.reset(); }
};

struct NotchFilter1Hz {
  Biquad s1 = Biquad( 0.97546f, -1.89505f,  0.97546f, -1.89505f,  0.95091f);
  float process(float x) { return s1.process(x); }
  void reset() { s1.reset(); }
};

constexpr int SG_WIN = 11;
constexpr float SG_COEF[SG_WIN] = {
    -0.09090909f,  0.06060606f,  0.16883117f,  0.23376623f,  0.25541126f,
     0.23376623f,  0.16883117f,  0.06060606f, -0.09090909f, -0.06060606f,
    -0.09090909f
};

struct SavitzkyGolay {
  static const int WIN = SG_WIN;
  float buf[WIN] = {};
  int   head = 0;
  int   filled = 0;
  
  float process(float x) {
    buf[head] = x;
    head = (head + 1) % WIN;
    if (filled < WIN) { filled++; }
    if (filled < WIN) return x; 

    float out = 0.0f;
    for (int i = 0; i < WIN; i++) {
      int idx = (head + i) % WIN; 
      out += SG_COEF[i] * buf[idx];
    }
    return out;
  }

  void reset() {
    for (int i = 0; i < WIN; i++) buf[i] = 0;
    head = 0; filled = 0;
  }
};

// =============================================================
//  STAGE 2: Time-Domain Peak Detection (Sliding Window)
// =============================================================
struct PeakDetectionRR {
  static const int BUF_LEN      = 300; 
  static const int UPDATE_EVERY = 25; 
  static const int MIN_DIST     = 20; 

  float buf[BUF_LEN] = {};
  int   head = 0;
  bool  full = false;
  int   samplesSinceUpdate = 0;
  float lastRR = 0.0f;

  void push(float v) {
    buf[head] = v;
    head = (head + 1) % BUF_LEN;
    if (!full && head == 0) full = true;
    samplesSinceUpdate++;
  }

  float compute() {
    if (!full) return 0.0f; 
    if (samplesSinceUpdate < UPDATE_EVERY) return lastRR; 
    
    samplesSinceUpdate = 0; 

    float win[BUF_LEN];
    float mean = 0;
    for (int i = 0; i < BUF_LEN; i++) {
      win[i] = buf[(head + i) % BUF_LEN];
      mean += win[i];
    }
    mean /= BUF_LEN; 

    int peakIndices[50]; 
    int peakCount = 0;
    
    for (int i = 1; i < BUF_LEN - 1; i++) {
      if (win[i] > win[i - 1] && win[i] > win[i + 1] && win[i] > mean) {
        if (peakCount == 0 || (i - peakIndices[peakCount - 1]) >= MIN_DIST) {
          peakIndices[peakCount++] = i;
          if (peakCount >= 50) break; 
        } 
        else if (win[i] > win[peakIndices[peakCount - 1]]) {
          peakIndices[peakCount - 1] = i;
        }
      }
    }

    if (peakCount >= 2) {
      float totalIntervals = 0;
      for (int i = 1; i < peakCount; i++) {
        totalIntervals += (peakIndices[i] - peakIndices[i - 1]);
      }
      
      float avgIntervalSamples = totalIntervals / (peakCount - 1);
      float avgIntervalSeconds = avgIntervalSamples / (float)SAMPLE_RATE_HZ; 
      float newRR = 60.0f / avgIntervalSeconds;
      
      if (newRR >= 4.0f && newRR <= 60.0f) {
         lastRR = (lastRR == 0.0f) ? newRR : (0.4f * newRR + 0.6f * lastRR);
      }
    }
    return lastRR;
  }

  void reset() {
    for (int i = 0; i < BUF_LEN; i++) buf[i] = 0;
    head = 0; full = false; samplesSinceUpdate = 0; lastRR = 0;
  }
};

// =============================================================
//  MEDIAN FILTER
// =============================================================
struct MedianFilter {
  static const int SZ = 9;
  float buf[SZ] = {};
  int   idx = 0;

  float process(float v) {
    buf[idx] = v;
    idx = (idx + 1) % SZ;
    float s[SZ];
    for (int i = 0; i < SZ; i++) s[i] = buf[i];
    for (int i = 1; i < SZ; i++) {
      float key = s[i];
      int j = i - 1;
      while (j >= 0 && s[j] > key) { s[j+1] = s[j]; j--; }
      s[j+1] = key;
    }
    return s[SZ / 2];
  }
};

// =============================================================
//  MULTI-AXIS DIFFERENTIAL FUSION
// =============================================================
struct AxisFuser {
  static const int WIN = 50;
  float buf[WIN] = {};
  int   idx = 0;
  bool  full = false;
  float variance = 0;
  
  void push(float v) {
    buf[idx] = v;
    idx = (idx + 1) % WIN;
    if (idx == 0) full = true;
    int n = full ? WIN : idx;
    if (n < 2) { variance = 0; return; }
    float mean = 0;
    for (int i = 0; i < n; i++) mean += buf[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) var += (buf[i]-mean)*(buf[i]-mean);
    variance = var / n;
  }
};

float fusedDiff(float dx, float dy, float dz, float vx, float vy, float vz) {
  float total = vx + vy + vz;
  if (total < 1e-9f) return (dx + dy + dz) / 3.0f;  
  return (dx*vx + dy*vy + dz*vz) / total;
}

// =============================================================
//  GLOBALS & INSTANCES
// =============================================================
DFRobot_BMI160 bmi_chest, bmi_back;
bool bmiReady = false;

MedianFilter    med_cx, med_cy, med_cz;   
MedianFilter    med_bx, med_by, med_bz;
BPFilter6       bp_cx, bp_cy, bp_cz;
BPFilter6       bp_bx, bp_by, bp_bz;
NotchFilter1Hz  notch;
SavitzkyGolay   sgSmooth;            
PeakDetectionRR pdRR; 

AxisFuser fuser_cx, fuser_cy, fuser_cz;
AxisFuser fuser_bx, fuser_by, fuser_bz;

volatile float respiratoryRate  = 0.0f;
volatile bool  motionGated    = false;

unsigned long lastMotionTime = 0;
unsigned long lastRRSample   = 0;
unsigned long lastPrint      = 0;

// WiFi/Firebase state tracking (non-blocking)
volatile bool wifiWasConnected = false;
unsigned long lastWifiAttempt = 0;
bool firebaseConfigured = false;

// =============================================================
//  WIFI EVENT HANDLER (event-driven, no blocking loops)
// =============================================================
void onWifiEvent(WiFiEvent_t event) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.println("[WiFi] Connected to AP.");
      break;
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.print("[WiFi] Got IP: ");
      Serial.println(WiFi.localIP());
      wifiWasConnected = true;
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      Serial.println("[WiFi] Disconnected. Will retry in background.");
      wifiWasConnected = false;
      break;
    default:
      break;
  }
}

// Call periodically from loop() — non-blocking reconnect attempt.
void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  unsigned long now = millis();
  if (now - lastWifiAttempt < WIFI_RECONNECT_INTERVAL_MS) return;
  lastWifiAttempt = now;
  Serial.println("[WiFi] Attempting reconnect...");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

// =============================================================
//  CORE 0: FIREBASE UPLOAD TASK (STANDALONE PROOF)
// =============================================================
void firebaseUploadTask(void *pvParameters) {
  esp_task_wdt_add(NULL); // register this task with the watchdog
  unsigned long lastStatusUpload = 0;

  for (;;) {
    esp_task_wdt_reset();

    // Only upload if properly connected and authenticated.
    // This never blocks the sensor loop on core 1.
    if (WiFi.status() == WL_CONNECTED && Firebase.ready()) {
      bool ok1 = Firebase.setFloat(fbdo, "/respiratory_data/rate", respiratoryRate);
      esp_task_wdt_reset(); // feed the watchdog immediately after this blocking call
      if (!ok1) {
        Serial.print("[Firebase] setFloat failed: ");
        Serial.println(fbdo.errorReason());
      }

      bool ok2 = Firebase.setBool(fbdo, "/respiratory_data/motion_gated", motionGated);
      esp_task_wdt_reset(); // feed the watchdog immediately after this blocking call
      if (!ok2) {
        Serial.print("[Firebase] setBool failed: ");
        Serial.println(fbdo.errorReason());
      }

      // Status/progress fields — uploaded less frequently since they
      // change slower than rate/motion, to keep this loop iteration short.
      unsigned long now = millis();
      if (now - lastStatusUpload >= STATUS_UPLOAD_INTERVAL_MS) {
        lastStatusUpload = now;

        Firebase.setString(fbdo, "/respiratory_data/status/state", stateToString(systemState));
        esp_task_wdt_reset();

        Firebase.setInt(fbdo, "/respiratory_data/status/window_progress_percent", windowProgressPercent);
        esp_task_wdt_reset();

        Firebase.setBool(fbdo, "/respiratory_data/status/wifi_connected", WiFi.status() == WL_CONNECTED);
        esp_task_wdt_reset();

        Firebase.setBool(fbdo, "/respiratory_data/status/sensors_ready", bmiReady);
        esp_task_wdt_reset();

        Firebase.setInt(fbdo, "/respiratory_data/status/uptime_seconds", (int)(millis() / 1000));
        esp_task_wdt_reset();

        Firebase.setInt(fbdo, "/respiratory_data/status/wifi_rssi", WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0);
        esp_task_wdt_reset();
      }
    }

    // Generous delay prevents WDT crash and network stack overflow
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

// =============================================================
//  UPDATE BMI160
// =============================================================
void updateBMI160() {
  unsigned long now = millis();
  if (now - lastRRSample < SAMPLE_INTERVAL) return;
  lastRRSample = now;

  int16_t dc[6] = {}, db[6] = {};
  if (bmi_chest.getAccelGyroData(dc) != 0) return;
  if (bmi_back.getAccelGyroData(db)  != 0) return;
  
  float cxr = (dc[3]/16384.0f) * 1000.0f;
  float cyr = (dc[4]/16384.0f) * 1000.0f;
  float czr = (dc[5]/16384.0f) * 1000.0f;
  
  float bxr = (db[3]/16384.0f) * 1000.0f;
  float byr = (db[4]/16384.0f) * 1000.0f;
  float bzr = (db[5]/16384.0f) * 1000.0f;

  float cx_g = cxr/1000.0f; float cy_g = cyr/1000.0f; float cz_g = czr/1000.0f;
  float bx_g = bxr/1000.0f; float by_g = byr/1000.0f; float bz_g = bzr/1000.0f;

  float magChest = sqrtf(cx_g*cx_g + cy_g*cy_g + cz_g*cz_g);
  float magBack  = sqrtf(bx_g*bx_g + by_g*by_g + bz_g*bz_g);
  
  if (magChest < 0.1f || magBack < 0.1f) return;

  float instMotionChest = fabsf(magChest - 1.0f) * 1000.0f;
  float instMotionBack  = fabsf(magBack - 1.0f) * 1000.0f;
  float motionMag = fmaxf(instMotionChest, instMotionBack);
  
  if (motionMag > MOTION_THRESHOLD) {
    motionGated = true; lastMotionTime = now;
    bp_cx.reset(); bp_cy.reset(); bp_cz.reset();
    bp_bx.reset(); bp_by.reset(); bp_bz.reset();
    notch.reset(); sgSmooth.reset(); pdRR.reset();
  }
  
  if (motionGated && now - lastMotionTime > MOTION_LOCKOUT_MS) motionGated = false;
  
  float cx = med_cx.process(cxr), cy = med_cy.process(cyr), cz = med_cz.process(czr);
  float bx = med_bx.process(bxr), by = med_by.process(byr), bz = med_bz.process(bzr);
  
  float fcx = bp_cx.process(cx), fcy = bp_cy.process(cy), fcz = bp_cz.process(cz);
  float fbx = bp_bx.process(bx), fby = bp_by.process(by), fbz = bp_bz.process(bz);
  
  float dx = fcx - fbx, dy = fcy - fby, dz = fcz - fbz;
  fuser_cx.push(fcx); fuser_cy.push(fcy); fuser_cz.push(fcz);
  fuser_bx.push(fbx); fuser_by.push(fby); fuser_bz.push(fbz);

  float vx = (fuser_cx.variance + fuser_bx.variance);
  float vy = (fuser_cy.variance + fuser_by.variance);
  float vz = (fuser_cz.variance + fuser_bz.variance);
  
  float diff_fused = fusedDiff(dx, dy, dz, vx, vy, vz);
  
  float notched = notch.process(diff_fused);
  float sig     = sgSmooth.process(notched);
  
  if (!motionGated) {
    pdRR.push(sig);
    float rr = pdRR.compute();
    if (rr > 0) respiratoryRate = rr;
  }

  // ── Status tracking for Firebase progress reporting ──
  if (motionGated) {
    systemState = STATE_MOTION_GATED;
  } else if (!pdRR.full) {
    systemState = STATE_GATHERING_WINDOW;
    windowProgressPercent = (pdRR.samplesSinceUpdate * 100) / PeakDetectionRR::BUF_LEN;
  } else {
    systemState = STATE_RUNNING;
    windowProgressPercent = 100;
  }

#if ENABLE_TELEPLOT
  Serial.print(">SigFused_mg:");  Serial.println(sig, 2);
  Serial.print(">Motion_mg:");    Serial.println(motionMag, 1);
  Serial.print(">Gated:");        Serial.println(motionGated ? 10 : 0);
  Serial.print(">RespRate:");     Serial.println(respiratoryRate, 1);
#endif
}

void printStatus() {
  if (millis() - lastPrint < PRINT_INTERVAL) return;
  lastPrint = millis();

  Serial.print("State=");
  Serial.print(stateToString(systemState));
  Serial.print(" Progress=");
  Serial.print(windowProgressPercent);
  Serial.println("%");

  Serial.print("RespRate=");
  if (!pdRR.full) {
    int percent = (pdRR.samplesSinceUpdate * 100) / PeakDetectionRR::BUF_LEN;
    Serial.print("Gathering window ("); Serial.print(percent); Serial.println("%)");
  } else {
    Serial.print(respiratoryRate, 1); Serial.println(" br/min");
  }
  Serial.print("WiFi=");
  Serial.println(WiFi.status() == WL_CONNECTED ? "connected" : "DISCONNECTED");
}

// =============================================================
//  SETUP
// =============================================================
void setup() {
  Serial.begin(115200);
  delay(1500); // let power rails stabilize before WiFi radio init (helps with brownout on marginal supplies)
  Serial.println("\nBooting Standalone System...");
  systemState = STATE_BOOTING;

  // Configure watchdog for the whole app. 20s gives headroom for a slow
  // Firebase/SSL call (bounded to ~4s read timeout, see setup() below)
  // plus WiFi reconnect attempts, without masking a genuine hang.
  esp_task_wdt_init(20, true);

  // 1. Initialize Wi-Fi (event-driven, non-blocking after this point)
  systemState = STATE_WIFI_CONNECTING;
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false); // avoid unnecessary flash writes on every connect

#if REDUCE_WIFI_TX_POWER
  // Lowering TX power reduces peak current draw, which helps a lot if the
  // 3.3V rail is marginal. Remove/raise this once hardware power is solid.
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setTxPower(WIFI_POWER_15dBm);
#else
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
#endif

  // Short bounded wait just so first boot has a chance to come up before
  // we start sensors — but we DO NOT hang forever like the old code did.
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 8000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("Wi-Fi connected on boot.");
  } else {
    Serial.println("Wi-Fi not yet connected — will keep retrying in background.");
  }

  // 2. Initialize Firebase
  fbdo.setBSSLBufferSize(1024, 512); // CRITICAL FIX: Optimize SSL memory
  config.database_url = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.reconnectWiFi(true);

  // Bound how long a single Firebase call is allowed to block. Without this,
  // a stalled SSL handshake or slow network can block setFloat()/setBool()
  // for far longer than the task watchdog timeout and trigger a reboot.
  fbdo.setResponseSize(1024);
  Firebase.setReadTimeout(fbdo, 4000);        // ms
  Firebase.setwriteSizeLimit(fbdo, "tiny");   // small payloads, no need for large limit

  Firebase.begin(&config, &auth);
  firebaseConfigured = true;

  // 3. Start Dual Core Task for Firebase Uploads
  xTaskCreatePinnedToCore(
    firebaseUploadTask,   
    "FirebaseTask",       
    8192,                 
    NULL,                 
    1,                    
    &FirebaseTaskHandle,  
    0);                   

  // 4. Initialize Sensors
  systemState = STATE_SENSOR_INIT;
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  bool sensorsOk = true;
  if (bmi_chest.softReset() != BMI160_OK) { Serial.println("Chest sensor soft reset failed"); sensorsOk = false; }
  delay(100);
  if (sensorsOk && bmi_chest.I2cInit(ADDR_CHEST) != BMI160_OK) { Serial.println("Chest sensor init failed"); sensorsOk = false; }

  delay(100);

  if (sensorsOk && bmi_back.softReset() != BMI160_OK) { Serial.println("Back sensor soft reset failed"); sensorsOk = false; }
  delay(100);
  if (sensorsOk && bmi_back.I2cInit(ADDR_BACK) != BMI160_OK) { Serial.println("Back sensor init failed"); sensorsOk = false; }

  bmiReady = sensorsOk;
  if (!bmiReady) {
    // Don't hard-lock the MCU (while(1)) — that made the whole board
    // unrecoverable without a manual reset. Instead keep WiFi/Firebase
    // alive and keep retrying sensor init in the background.
    systemState = STATE_SENSOR_FAILED;
    Serial.println("Sensors not ready — will retry periodically in loop().");
  } else {
    systemState = STATE_GATHERING_WINDOW;
    Serial.println("System Processing Running Standalone.");
  }
}

// =============================================================
//  LOOP
// =============================================================
unsigned long lastSensorRetry = 0;

void loop() {
  maintainWifi(); // non-blocking reconnect if WiFi dropped

  if (bmiReady) {
    updateBMI160();
  } else {
    systemState = STATE_SENSOR_FAILED;
    // Retry sensor init every 5s without blocking WiFi/Firebase
    unsigned long now = millis();
    if (now - lastSensorRetry > 5000) {
      lastSensorRetry = now;
      Serial.println("Retrying BMI160 init...");
      bool ok = (bmi_chest.softReset() == BMI160_OK) &&
                (bmi_chest.I2cInit(ADDR_CHEST) == BMI160_OK) &&
                (bmi_back.softReset() == BMI160_OK) &&
                (bmi_back.I2cInit(ADDR_BACK) == BMI160_OK);
      bmiReady = ok;
      if (ok) {
        Serial.println("Sensors recovered.");
        systemState = STATE_GATHERING_WINDOW;
      }
    }
  }

  printStatus();
}
