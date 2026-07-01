"""
local_predict.py
─────────────────────────────────────────────────────────────────
Connects to Firebase, watches respiratory_data/rate and spo2 in 
real time, and builds a 5-minute rolling buffer.

Two modes:
  1. Background collection — always running, silently buffering
  2. Manual prediction — press ENTER any time to run a prediction
"""

import os
import sys
import time
import threading
import collections
from datetime import datetime

import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db


# ═════════════════════════════════════════════════════════════
# CONFIGURATION 
# ═════════════════════════════════════════════════════════════

KEY_PATH     = "firebase_key.json"
DATABASE_URL = "https://respiratory-system-monitor-default-rtdb.firebaseio.com"
MODEL_PATH   = "data/respiratory_risk_model.pkl"

RR_PATH   = "respiratory_data/rate"
SPO2_PATH = "respiratory_data/spo2"

PATIENT_AGE    = 25
PATIENT_GENDER = 1

WINDOW_SIZE_SECONDS = 300       # 5 minutes rolling window
LOG_FILE = "data/prediction_log.csv"

# ── TEST MODE & FAILSAFE CONFIG ────────────────────────────────
# Set the expected time (in seconds) between Firebase updates.
# Fake data will be injected at this rate.
EXPECTED_UPDATE_INTERVAL = 2.0  

# Dynamically calculate minimum readings required (e.g., ~60 seconds of data).
# Ensures a slow sensor doesn't mathematically prevent predictions.
MIN_READINGS_TO_PREDICT = max(10, int(60 / EXPECTED_UPDATE_INTERVAL))

TEST_MODE_SPO2 = True
TEST_MODE_FAKE_SPO2 = 97.0   

TEST_MODE_RR = True          
TEST_MODE_FAKE_RR = 14.5     

FEATURE_ORDER = [
    "spo2_mean", "spo2_min", "spo2_std", "spo2_median",
    "spo2_pct_below_95", "spo2_pct_below_92",
    "rr_mean", "rr_max", "rr_min", "rr_std", "rr_median",
    "rr_pct_abnormal", "rr_pct_above_25",
    "hr_mean", "hr_std",
    "age", "gender",
]


# ═════════════════════════════════════════════════════════════
# STARTUP CHECKS
# ═════════════════════════════════════════════════════════════

def startup_checks():
    errors = []
    if not os.path.exists(KEY_PATH):
        errors.append(f"Firebase key not found at '{KEY_PATH}'.")
    if "YOUR-PROJECT-ID" in DATABASE_URL:
        errors.append("DATABASE_URL still contains the placeholder.")
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found at '{MODEL_PATH}'.")
    if not os.path.exists("data"):
        errors.append("'data/' folder not found.")

    if errors:
        print("\n✗ STARTUP CHECKS FAILED:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    print("✓ All startup checks passed")


startup_checks()


# ═════════════════════════════════════════════════════════════
# CONNECT TO FIREBASE & LOAD MODEL
# ═════════════════════════════════════════════════════════════

cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
print(f"✓ Connected to Firebase: {DATABASE_URL}")

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

expected_n_features = model.named_steps["clf"].n_features_in_
if expected_n_features != len(FEATURE_ORDER):
    print("\n✗ FEATURE MISMATCH. Stopping to avoid silently wrong predictions.")
    sys.exit(1)
print(f"✓ Feature count verified: {expected_n_features} features expected.")


# ═════════════════════════════════════════════════════════════
# ROLLING BUFFER
# ═════════════════════════════════════════════════════════════

spo2_buffer = collections.deque()
rr_buffer   = collections.deque()
buffer_lock = threading.Lock()

def prune_old_readings(buf):
    cutoff = time.time() - WINDOW_SIZE_SECONDS
    while buf and buf[0][0] < cutoff:
        buf.popleft()


# ═════════════════════════════════════════════════════════════
# FIREBASE LISTENERS & INJECTION
# ═════════════════════════════════════════════════════════════

def on_spo2_change(event):
    if event.data is None: return
    try:
        val = float(event.data)
        with buffer_lock:
            spo2_buffer.append((time.time(), val))
            prune_old_readings(spo2_buffer)
    except (ValueError, TypeError): pass

def on_rr_change(event):
    if event.data is None: return
    try:
        val = float(event.data)
        with buffer_lock:
            rr_buffer.append((time.time(), val))
            prune_old_readings(rr_buffer)
    except (ValueError, TypeError): pass


last_inject_time = 0

def inject_fake_data_if_needed():
    global last_inject_time
    current_time = time.time()
    
    # Only inject based on the configured interval failsafe
    if current_time - last_inject_time >= EXPECTED_UPDATE_INTERVAL:
        with buffer_lock:
            if TEST_MODE_SPO2:
                spo2_buffer.append((current_time, TEST_MODE_FAKE_SPO2))
                prune_old_readings(spo2_buffer)
            if TEST_MODE_RR:
                rr_buffer.append((current_time, TEST_MODE_FAKE_RR))
                prune_old_readings(rr_buffer)
        last_inject_time = current_time


if not TEST_MODE_SPO2:
    db.reference(SPO2_PATH).listen(on_spo2_change)
    print(f"✓ Listening to Firebase /{SPO2_PATH}")
else:
    print(f"⚠ TEST MODE: Using FAKE SpO2 value {TEST_MODE_FAKE_SPO2}%")

if not TEST_MODE_RR:
    db.reference(RR_PATH).listen(on_rr_change)
    print(f"✓ Listening to Firebase /{RR_PATH}\n")
else:
    print(f"⚠ TEST MODE: Using FAKE RR value {TEST_MODE_FAKE_RR} bpm\n")


# ═════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_features():
    with buffer_lock:
        prune_old_readings(spo2_buffer)
        prune_old_readings(rr_buffer)
        spo2_vals = [v for (_, v) in spo2_buffer]
        rr_vals   = [v for (_, v) in rr_buffer]

    warnings = []

    if len(spo2_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(spo2_vals), [f"Only {len(spo2_vals)} SpO2 buffered (need {MIN_READINGS_TO_PREDICT})."]

    if len(rr_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(rr_vals), [f"Only {len(rr_vals)} RR buffered (need {MIN_READINGS_TO_PREDICT})."]

    spo2 = np.array(spo2_vals, dtype=float)
    rr   = np.array(rr_vals, dtype=float)

    spo2_clean = spo2[(spo2 >= 50) & (spo2 <= 100)]
    rr_clean   = rr[(rr >= 3) & (rr <= 60)]

    if len(spo2_clean) == 0 or len(rr_clean) == 0:
        return None, 0, ["All readings were out of physiological range."]

    features = {
        "spo2_mean":         np.mean(spo2_clean),
        "spo2_min":          np.min(spo2_clean),
        "spo2_std":          np.std(spo2_clean),
        "spo2_median":       np.median(spo2_clean),
        "spo2_pct_below_95": np.mean(spo2_clean < 95) * 100,
        "spo2_pct_below_92": np.mean(spo2_clean < 92) * 100,
        "rr_mean":           np.mean(rr_clean),
        "rr_max":            np.max(rr_clean),
        "rr_min":            np.min(rr_clean),
        "rr_std":            np.std(rr_clean),
        "rr_median":         np.median(rr_clean),
        "rr_pct_abnormal":   np.mean((rr_clean < 12) | (rr_clean > 20)) * 100,
        "rr_pct_above_25":   np.mean(rr_clean > 25) * 100,
        "hr_mean":           85.0,   
        "hr_std":            5.0,    
        "age":               float(PATIENT_AGE),
        "gender":            float(PATIENT_GENDER),
    }

    X = np.array([[features[col] for col in FEATURE_ORDER]])
    return X, len(spo2_clean), warnings


# ═════════════════════════════════════════════════════════════
# RUN A SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════

def run_prediction():
    X, n_readings, warnings = extract_features()

    print(f"\n{'─'*55}")
    print(f"  Prediction requested at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'─'*55}")

    if X is None:
        print(f"  ✗ Cannot predict yet.")
        for w in warnings:
            print(f"    - {w}")
        print(f"{'─'*55}\n")
        return None

    for w in warnings:
        print(f"  ⚠ {w}")

    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1]) * 100
    risk_label  = "AT RISK" if prediction == 1 else "NOT AT RISK"

    is_any_test_mode = TEST_MODE_SPO2 or TEST_MODE_RR

    result = {
        "prediction":       prediction,
        "risk_label":       risk_label,
        "risk_probability": round(probability, 1),
        "spo2_mean":        round(float(X[0][0]), 1),
        "spo2_min":         round(float(X[0][1]), 1),
        "rr_mean":          round(float(X[0][6]), 1),
        "rr_max":           round(float(X[0][7]), 1),
        "readings_used":    n_readings,
        "window_seconds":   WINDOW_SIZE_SECONDS,
        "timestamp":        int(time.time() * 1000),
        "source":           "local_predict.py (manual)",
        "is_test_data":     is_any_test_mode,
    }

    print(f"\n  RESULT: {risk_label}  ({probability:.1f}% probability)")
    print(f"  Mean SpO2: {result['spo2_mean']}%  (min: {result['spo2_min']}%)"
          f"{'  [FAKE]' if TEST_MODE_SPO2 else ''}")
    print(f"  Mean RR:   {result['rr_mean']} bpm  (max: {result['rr_max']})"
          f"{'  [FAKE]' if TEST_MODE_RR else ''}")
    print(f"  Based on {n_readings} readings")

    try:
        db.reference("prediction").set(result)
        print(f"  ✓ Written to Firebase at /prediction")
    except Exception as e:
        print(f"  ✗ Failed to write to Firebase: {e}")

    log_prediction_locally(result)
    print(f"{'─'*55}\n")
    return result

def log_prediction_locally(result):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            f.write("timestamp,datetime,risk_label,risk_probability,"
                    "spo2_mean,spo2_min,rr_mean,rr_max,readings_used,is_test_data\n")
        dt_str = datetime.fromtimestamp(result["timestamp"] / 1000).isoformat()
        f.write(f"{result['timestamp']},{dt_str},{result['risk_label']},"
                f"{result['risk_probability']},{result['spo2_mean']},"
                f"{result['spo2_min']},{result['rr_mean']},{result['rr_max']},"
                f"{result['readings_used']},{result['is_test_data']}\n")


# ═════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ═════════════════════════════════════════════════════════════

def print_status():
    with buffer_lock:
        prune_old_readings(spo2_buffer)
        prune_old_readings(rr_buffer)
        n_spo2 = len(spo2_buffer)
        n_rr   = len(rr_buffer)

    ready = (n_spo2 >= MIN_READINGS_TO_PREDICT and n_rr >= MIN_READINGS_TO_PREDICT)
    status = "✓ ready to predict" if ready else "buffering..."
    tag = " [TEST]" if (TEST_MODE_SPO2 or TEST_MODE_RR) else ""

    print(f"\r  SpO2 buffer: {n_spo2:3d}   RR buffer: {n_rr:3d}   "
          f"Req: {MIN_READINGS_TO_PREDICT}  {tag} [{status}]   ", end="", flush=True)


# ═════════════════════════════════════════════════════════════
# MAIN 
# ═════════════════════════════════════════════════════════════

def status_loop():
    while True:
        inject_fake_data_if_needed()
        print_status()
        time.sleep(0.5)  # Runs fast to keep UI responsive, while data logic relies on EXPECTED_UPDATE_INTERVAL

def main():
    print("\n" + "="*55)
    print("  VitaSync Local Predictor")
    if TEST_MODE_SPO2 or TEST_MODE_RR:
        print("  ⚠⚠⚠  TEST MODE ACTIVE  ⚠⚠⚠")
    print("="*55)
    print(f"  Window size:        {WINDOW_SIZE_SECONDS}s")
    print(f"  Min to predict:     {MIN_READINGS_TO_PREDICT} readings")
    print("="*55)
    print("\n  Buffering live data from Firebase in the background.")
    print("  Press ENTER at any time to run a prediction.")
    print("  Press Ctrl+C to quit.\n")

    status_thread = threading.Thread(target=status_loop, daemon=True)
    status_thread.start()

    try:
        while True:
            input()   
            run_prediction()
    except KeyboardInterrupt:
        print("\n\n  Shutting down. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()