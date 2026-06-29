#include <Arduino.h>
#include <Wire.h>
#include <DFRobot_BMI160.h>
#include <WiFi.h>
#include <FirebaseESP32.h>

// =============================================================
//  WIFI & FIREBASE CONFIGURATION
// =============================================================
#define WIFI_SSID "Isulaaa"
#define WIFI_PASSWORD "00000000"

#define FIREBASE_HOST "https://respiratory-system-monitor-default-rtdb.firebaseio.com/"
#define FIREBASE_AUTH "AIzaSyAhwJpInIiR7JNX8jqS7Z6JTczntVLBF78"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

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
  static const int BUF_LEN      = 750; // 30 seconds at 25 Hz
  static const int UPDATE_EVERY = 250; // Advance interval: 10 seconds at 25 Hz
  static const int MIN_DIST     = 20;  // Minimum samples between peaks (Max ~75 BPM)

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

// Volatile keyword is used so both processor cores can read/write safely
volatile float respiratoryRate  = 0.0f;
volatile bool  motionGated    = false;

unsigned long lastMotionTime = 0;
unsigned long lastRRSample   = 0;
unsigned long lastPrint      = 0;

TaskHandle_t FirebaseTaskHandle;

// =============================================================
//  CORE 0: FIREBASE UPLOAD TASK (Runs in Background)
// =============================================================
void firebaseUploadTask(void *pvParameters) {
  for (;;) {
    // Only upload if WiFi is connected and we actually have a full buffer of data
    if (WiFi.status() == WL_CONNECTED && pdRR.full) {
      
      // Upload Respiratory Rate
      if (Firebase.setFloat(fbdo, "/respiratory_data/rate", respiratoryRate)) {
        // Silent success to keep Serial Monitor clean
      } else {
        Serial.print(">> Firebase Error: ");
        Serial.println(fbdo.errorReason());
      }
      
      // Upload Motion Status (Optional but helpful for UI)
      Firebase.setBool(fbdo, "/respiratory_data/motion_gated", motionGated);
    }
    
    // Wait 2000 milliseconds (2 seconds) before running again
    vTaskDelay(2000 / portTICK_PERIOD_MS);
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

  float bxr_g = bxr/1000.0f; float byr_g = byr/1000.0f;
  float bzr_g = bzr/1000.0f;
  float motionMag = fabsf(sqrtf(bxr_g*bxr_g + byr_g*byr_g + bzr_g*bzr_g) - 1.0f) * 1000.0f;
  
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

  Serial.print(">SigFused_mg:");  Serial.println(sig, 2);
  Serial.print(">Motion_mg:");    Serial.println(motionMag, 1);
  Serial.print(">Gated:");        Serial.println(motionGated ? 10 : 0);
  Serial.print(">RespRate:");     Serial.println(respiratoryRate, 1);
  
  float bufferFillPct = pdRR.full ? 100.0f : ((float)pdRR.samplesSinceUpdate / PeakDetectionRR::BUF_LEN) * 100.0f;
  Serial.print(">BufferFill_%:"); Serial.println(bufferFillPct, 1);
}

// =============================================================
//  SERIAL STATUS
// =============================================================
void printStatus() {
  if (millis() - lastPrint < PRINT_INTERVAL) return;
  lastPrint = millis();
  
  Serial.print("RespRate=");
  if (!pdRR.full) {
    int percent = (pdRR.samplesSinceUpdate * 100) / PeakDetectionRR::BUF_LEN;
    Serial.print("Gathering window ("); Serial.print(percent); Serial.println("%)");
  } else {
    Serial.print(respiratoryRate, 1); Serial.println(" br/min");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Booting...");
  
  // 1. Initialize Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected!");

  // 2. Initialize Firebase
  config.database_url = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.reconnectWiFi(true);
  Firebase.begin(&config, &auth);
  
  // 3. Start Dual Core Task for Firebase Uploads
  xTaskCreatePinnedToCore(
    firebaseUploadTask,   // Task function
    "FirebaseTask",       // Name of task
    8192,                 // Stack size
    NULL,                 // Parameter
    1,                    // Priority
    &FirebaseTaskHandle,  // Task handle
    0);                   // Pin to Core 0

  // 4. Initialize Sensors
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  
  if (bmi_chest.softReset() != BMI160_OK || bmi_chest.I2cInit(ADDR_CHEST) != BMI160_OK) {
    Serial.println("Abdomen BMI160 FAILED"); while(1) delay(1000);
  }
  if (bmi_back.softReset() != BMI160_OK || bmi_back.I2cInit(ADDR_BACK) != BMI160_OK) {
    Serial.println("Back BMI160 FAILED"); while(1) delay(1000);
  }
  
  bmiReady = true;
  Serial.println("All systems GO. Monitoring 25Hz DSP on Core 1, Uploading to Firebase on Core 0.");
}

void loop() {
  // Core 1 (Default Loop): Only handles time-critical sensor math
  if (bmiReady) updateBMI160();
  printStatus();
}