"""
local_predict.py
─────────────────────────────────────────────────────────────────
VitaSync — Local ML Prediction Script
─────────────────────────────────────────────────────────────────

Firebase tree this script reads from:
  biomedical_data/
    bpm:        heart rate (beats per minute)
    spo2:       blood oxygen saturation (%)
    timestamp:  device-side timestamp (ignored — we use Mac time)
  respiratory_data/
    rate:       respiratory rate (breaths per minute)
    motion_gated: bool (read but not used in model)

Prediction output is written to:
  prediction/   (top-level node, written after every manual ENTER)

How it works:
  - Listens to all three sensor paths in real time
  - Builds a 5-minute rolling buffer on your Mac for each signal
  - If a sensor stops updating (ESP32 disconnected), fake placeholder
    values are injected automatically, clearly labeled
  - If no real data arrives within FALLBACK_DELAY_SECONDS of startup,
    fake placeholders kick in for all missing sensors automatically
  - Press ENTER at any time to run a prediction once 60s of data exists
  - Ctrl+C to quit

Important: This script must stay running continuously to maintain
the 5-minute window. Closing it clears all buffers.
─────────────────────────────────────────────────────────────────
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

# ── Firebase paths (confirmed from console screenshot) ────────
SPO2_PATH = "biomedical_data/spo2"
BPM_PATH  = "biomedical_data/bpm"
RR_PATH   = "respiratory_data/rate"

# ── Prediction output path ────────────────────────────────────
PREDICTION_PATH = "prediction"

# ── Patient demographics ──────────────────────────────────────
PATIENT_AGE    = 25   # age of the person wearing the device
PATIENT_GENDER = 1    # 1 = Male, 0 = Female, -1 = Unknown

# ── Buffering and prediction thresholds ──────────────────────
WINDOW_SIZE_SECONDS    = 300   # 5 minutes — matches model training window
MIN_READINGS_TO_PREDICT = 60   # need at least 60s of data before predicting

# ── Disconnection / stale detection ──────────────────────────
# If a sensor hasn't sent a NEW value in this many seconds,
# we treat it as disconnected and inject fake placeholders instead.
STALE_THRESHOLD_SECONDS = 10

# ── First-run fallback ────────────────────────────────────────
# If a sensor has received NO real data at all within this many
# seconds of startup, we assume the ESP32 is not connected and
# start injecting fake placeholders automatically.
FALLBACK_DELAY_SECONDS = 15

# ── Fake placeholder values (injected when disconnected) ──────
# These are clinically "normal" values so the model doesn't
# produce misleading high-risk results purely from fake data.
# They are clearly labeled everywhere they appear.
FAKE_SPO2 = 97.0    # % — healthy baseline
FAKE_RR   = 16.0    # breaths/min — healthy baseline
FAKE_BPM  = 75.0    # beats/min — healthy baseline

# ── Local log ─────────────────────────────────────────────────
LOG_FILE = "data/prediction_log.csv"

# ── Feature order — must match Notebook 3 exactly ─────────────
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
        errors.append(
            f"Firebase key not found at '{KEY_PATH}'.\n"
            f"     Download from Firebase Console → Project Settings → "
            f"Service Accounts → Generate new private key."
        )

    if "YOUR-PROJECT-ID" in DATABASE_URL:
        errors.append(
            "DATABASE_URL still contains the placeholder. "
            "Edit it to your real Firebase Realtime Database URL."
        )

    if not os.path.exists(MODEL_PATH):
        errors.append(
            f"Model file not found at '{MODEL_PATH}'.\n"
            f"     Make sure you are running from inside ~/respiratory_ml/ "
            f"and that Notebook 3 was completed."
        )

    if not os.path.exists("data"):
        errors.append(
            "'data/' folder not found. Run this script from "
            "inside ~/respiratory_ml/."
        )

    if errors:
        print("\n✗ STARTUP CHECKS FAILED — fix these before continuing:\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}\n")
        sys.exit(1)

    print("✓ All startup checks passed")


startup_checks()


# ═════════════════════════════════════════════════════════════
# FIREBASE CONNECTION
# ═════════════════════════════════════════════════════════════

cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
print(f"✓ Connected to Firebase: {DATABASE_URL}")


# ═════════════════════════════════════════════════════════════
# MODEL LOADING AND VALIDATION
# ═════════════════════════════════════════════════════════════

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

expected_n_features = model.named_steps["clf"].n_features_in_
if expected_n_features != len(FEATURE_ORDER):
    print(
        f"\n✗ FEATURE MISMATCH: model expects {expected_n_features} features "
        f"but FEATURE_ORDER has {len(FEATURE_ORDER)}.\n"
        f"  The model file does not match this script. "
        f"Stopping to avoid silent wrong predictions."
    )
    sys.exit(1)
print(f"✓ Feature count verified: {expected_n_features} features")


# ═════════════════════════════════════════════════════════════
# SENSOR STATE TRACKING
# ═════════════════════════════════════════════════════════════
# For each sensor we track:
#   last_real_time  — Mac timestamp of the last REAL value received
#                     None means no real reading has ever arrived
#   is_fake         — True if we are currently injecting fake values
#                     for this sensor

class SensorState:
    def __init__(self, name, fake_value):
        self.name           = name
        self.fake_value     = fake_value
        self.last_real_time = None   # None = never received a real value
        self.is_fake        = False  # True = currently using fake data

spo2_state = SensorState("SpO2", FAKE_SPO2)
rr_state   = SensorState("RR",   FAKE_RR)
bpm_state  = SensorState("BPM",  FAKE_BPM)

script_start_time = time.time()


# ═════════════════════════════════════════════════════════════
# ROLLING BUFFERS
# ═════════════════════════════════════════════════════════════
# Each buffer stores (mac_timestamp, value) pairs.
# Pruning is by actual elapsed time, not reading count,
# so irregular update rates don't corrupt the window.

spo2_buffer = collections.deque()
rr_buffer   = collections.deque()
bpm_buffer  = collections.deque()

buffer_lock = threading.Lock()


def prune_old_readings(buf):
    """Drop readings older than WINDOW_SIZE_SECONDS."""
    cutoff = time.time() - WINDOW_SIZE_SECONDS
    while buf and buf[0][0] < cutoff:
        buf.popleft()


# ═════════════════════════════════════════════════════════════
# FIREBASE LISTENERS
# ═════════════════════════════════════════════════════════════

def _append(buf, value, state):
    """Append a real value to a buffer and update sensor state."""
    with buffer_lock:
        buf.append((time.time(), value))
        prune_old_readings(buf)
    state.last_real_time = time.time()
    if state.is_fake:
        state.is_fake = False
        print(f"\n  ✓ Real {state.name} data resumed from ESP32")


def on_spo2_change(event):
    if event.data is None:
        return
    try:
        _append(spo2_buffer, float(event.data), spo2_state)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed spo2 value ignored: {event.data!r}")


def on_rr_change(event):
    if event.data is None:
        return
    try:
        _append(rr_buffer, float(event.data), rr_state)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed rr value ignored: {event.data!r}")


def on_bpm_change(event):
    if event.data is None:
        return
    try:
        _append(bpm_buffer, float(event.data), bpm_state)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed bpm value ignored: {event.data!r}")


db.reference(SPO2_PATH).listen(on_spo2_change)
db.reference(RR_PATH).listen(on_rr_change)
db.reference(BPM_PATH).listen(on_bpm_change)
print(f"✓ Listening to {SPO2_PATH}")
print(f"✓ Listening to {RR_PATH}")
print(f"✓ Listening to {BPM_PATH}\n")


# ═════════════════════════════════════════════════════════════
# DISCONNECTION DETECTION AND FAKE INJECTION
# ═════════════════════════════════════════════════════════════

def check_and_inject_fake(buf, state):
    """
    Called once per second from the background loop.
    Determines whether a sensor is disconnected or stale,
    and if so injects a fake placeholder reading.

    A sensor is considered disconnected / stale if:
      - It has NEVER sent a real value AND FALLBACK_DELAY_SECONDS
        have passed since the script started, OR
      - It last sent a real value more than STALE_THRESHOLD_SECONDS ago

    Returns True if a fake value was injected this tick.
    """
    now = time.time()
    never_received = state.last_real_time is None
    stale = (
        not never_received and
        (now - state.last_real_time) > STALE_THRESHOLD_SECONDS
    )
    first_run_timeout = (
        never_received and
        (now - script_start_time) > FALLBACK_DELAY_SECONDS
    )

    should_fake = stale or first_run_timeout

    if should_fake:
        if not state.is_fake:
            # First tick we switch to fake — announce it
            reason = "never connected" if never_received else "disconnected"
            print(f"\n  ⚠ {state.name} sensor {reason} — "
                  f"injecting fake placeholder ({state.fake_value}) "
                  f"until real data resumes")
            state.is_fake = True

        with buffer_lock:
            buf.append((now, state.fake_value))
            prune_old_readings(buf)
        return True

    return False


# ═════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_features():
    """
    Reads current buffer state, applies physiological sanity
    filtering, and computes the 17 features the model was
    trained on.

    Returns: (X, n_spo2, n_rr, warnings, fake_sensors)
      X            — numpy array shape (1, 17), ready for model.predict()
      n_spo2       — number of valid SpO2 readings used
      n_rr         — number of valid RR readings used
      warnings     — list of warning strings to display
      fake_sensors — list of sensor names currently using fake data
    """
    with buffer_lock:
        prune_old_readings(spo2_buffer)
        prune_old_readings(rr_buffer)
        prune_old_readings(bpm_buffer)
        spo2_vals = [v for (_, v) in spo2_buffer]
        rr_vals   = [v for (_, v) in rr_buffer]
        bpm_vals  = [v for (_, v) in bpm_buffer]

    warnings     = []
    fake_sensors = []

    if spo2_state.is_fake:
        fake_sensors.append("SpO2")
    if rr_state.is_fake:
        fake_sensors.append("RR")
    if bpm_state.is_fake:
        fake_sensors.append("BPM/HR")

    # ── Minimum data check ──
    if len(spo2_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(spo2_vals), len(rr_vals), \
            [f"Only {len(spo2_vals)} SpO2 readings buffered "
             f"(need {MIN_READINGS_TO_PREDICT})."], fake_sensors

    if len(rr_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(spo2_vals), len(rr_vals), \
            [f"Only {len(rr_vals)} RR readings buffered "
             f"(need {MIN_READINGS_TO_PREDICT})."], fake_sensors

    # ── Physiological sanity filtering ──
    spo2 = np.array(spo2_vals, dtype=float)
    rr   = np.array(rr_vals,   dtype=float)
    bpm  = np.array(bpm_vals,  dtype=float) if bpm_vals else np.array([FAKE_BPM])

    spo2_clean = spo2[(spo2 >= 50)  & (spo2 <= 100)]
    rr_clean   = rr[(rr   >= 3)    & (rr   <= 60)]
    bpm_clean  = bpm[(bpm  >= 20)   & (bpm  <= 250)]

    dropped_spo2 = len(spo2) - len(spo2_clean)
    dropped_rr   = len(rr)   - len(rr_clean)
    if dropped_spo2 > 0:
        warnings.append(f"Dropped {dropped_spo2} out-of-range SpO2 readings")
    if dropped_rr > 0:
        warnings.append(f"Dropped {dropped_rr} out-of-range RR readings")

    if len(spo2_clean) == 0 or len(rr_clean) == 0:
        return None, 0, 0, \
            ["All readings are physiologically impossible — "
             "check sensor connection."], fake_sensors

    if len(spo2_clean) < len(spo2) * 0.5 or len(rr_clean) < len(rr) * 0.5:
        warnings.append(
            "More than half of readings were filtered as invalid — "
            "sensor may be malfunctioning."
        )

    # ── Use BPM if available, else fall back to training-set mean ──
    hr_mean = float(np.mean(bpm_clean)) if len(bpm_clean) > 0 else 85.0
    hr_std  = float(np.std(bpm_clean))  if len(bpm_clean) > 1 else 5.0
    if len(bpm_clean) == 0:
        warnings.append("No valid BPM readings — using training-set mean (85 bpm)")

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
        "hr_mean":           hr_mean,
        "hr_std":            hr_std,
        "age":               float(PATIENT_AGE),
        "gender":            float(PATIENT_GENDER),
    }

    X = np.array([[features[col] for col in FEATURE_ORDER]])
    return X, len(spo2_clean), len(rr_clean), warnings, fake_sensors


# ═════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════

def run_prediction():
    result = extract_features()

    # extract_features returns 5 values when successful, 4 when not ready
    if len(result) == 4:
        _, n_spo2, n_rr, warnings = result
        fake_sensors = []
    else:
        X, n_spo2, n_rr, warnings, fake_sensors = result

    has_fake = len(fake_sensors) > 0

    print(f"\n{'═'*55}")
    print(f"  Prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*55}")

    if has_fake:
        print(f"  ⚠ FAKE DATA IN USE: {', '.join(fake_sensors)}")
        print(f"  ⚠ This prediction is NOT clinically valid.")
        print(f"  ⚠ Real device readings required for a real prediction.")

    # X is None if not enough data
    if 'X' not in dir() or X is None:
        print(f"\n  ✗ Cannot predict yet:")
        for w in warnings:
            print(f"    - {w}")
        print(f"{'═'*55}\n")
        return None

    for w in warnings:
        print(f"  ⚠ {w}")

    prediction  = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1]) * 100
    risk_label  = "AT RISK" if prediction == 1 else "NOT AT RISK"

    # ── Collect per-sensor source labels for transparency ──
    spo2_source = "FAKE" if spo2_state.is_fake else "real"
    rr_source   = "FAKE" if rr_state.is_fake   else "real"
    bpm_source  = "FAKE" if bpm_state.is_fake  else "real"

    result_dict = {
        "prediction":        prediction,
        "risk_label":        risk_label,
        "risk_probability":  round(probability, 1),
        "spo2_mean":         round(float(X[0][0]), 1),
        "spo2_min":          round(float(X[0][1]), 1),
        "rr_mean":           round(float(X[0][6]), 1),
        "rr_max":            round(float(X[0][7]), 1),
        "hr_mean":           round(float(X[0][13]), 1),
        "spo2_readings":     n_spo2,
        "rr_readings":       n_rr,
        "window_seconds":    WINDOW_SIZE_SECONDS,
        "timestamp":         int(time.time() * 1000),
        "source":            "local_predict.py (manual)",
        "has_fake_data":     has_fake,
        "fake_sensors":      fake_sensors,
        "spo2_source":       spo2_source,
        "rr_source":         rr_source,
        "bpm_source":        bpm_source,
    }

    # ── Terminal output ──
    print(f"\n  RESULT:    {risk_label}")
    print(f"  Risk prob: {probability:.1f}%")
    print(f"  SpO2 mean: {result_dict['spo2_mean']}%  "
          f"(min: {result_dict['spo2_min']}%)  [{spo2_source}]")
    print(f"  RR mean:   {result_dict['rr_mean']} bpm  "
          f"(max: {result_dict['rr_max']})  [{rr_source}]")
    print(f"  HR mean:   {result_dict['hr_mean']} bpm  [{bpm_source}]")
    print(f"  SpO2 readings used: {n_spo2}")
    print(f"  RR readings used:   {n_rr}")

    # ── Write to Firebase ──
    try:
        db.reference(PREDICTION_PATH).set(result_dict)
        tag = "  (contains fake data — flagged)" if has_fake else ""
        print(f"\n  ✓ Written to Firebase /{PREDICTION_PATH}{tag}")
    except Exception as e:
        print(f"\n  ✗ Failed to write to Firebase: {e}")
        print(f"    Prediction was still logged locally.")

    # ── Write to local log ──
    log_prediction_locally(result_dict)

    print(f"{'═'*55}\n")
    return result_dict


def log_prediction_locally(r):
    """Append every prediction to a local CSV for permanent local history."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            f.write(
                "timestamp,datetime,risk_label,risk_probability,"
                "spo2_mean,spo2_min,rr_mean,rr_max,hr_mean,"
                "spo2_readings,rr_readings,has_fake_data,fake_sensors\n"
            )
        dt = datetime.fromtimestamp(r["timestamp"] / 1000).isoformat()
        fake_list = "|".join(r["fake_sensors"]) if r["fake_sensors"] else "none"
        f.write(
            f"{r['timestamp']},{dt},{r['risk_label']},{r['risk_probability']},"
            f"{r['spo2_mean']},{r['spo2_min']},{r['rr_mean']},{r['rr_max']},"
            f"{r['hr_mean']},{r['spo2_readings']},{r['rr_readings']},"
            f"{r['has_fake_data']},{fake_list}\n"
        )
    print(f"  ✓ Logged locally to {LOG_FILE}")


# ═════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ═════════════════════════════════════════════════════════════

def sensor_tag(state):
    """Returns [real] or [FAKE] label for a sensor."""
    return "[FAKE]" if state.is_fake else "[real]"


def print_status():
    with buffer_lock:
        prune_old_readings(spo2_buffer)
        prune_old_readings(rr_buffer)
        prune_old_readings(bpm_buffer)
        n_spo2 = len(spo2_buffer)
        n_rr   = len(rr_buffer)
        n_bpm  = len(bpm_buffer)

    ready = (
        n_spo2 >= MIN_READINGS_TO_PREDICT and
        n_rr   >= MIN_READINGS_TO_PREDICT
    )
    status = "✓ ready — press ENTER to predict" if ready else "buffering..."

    print(
        f"\r  SpO2{sensor_tag(spo2_state)}: {n_spo2:3d}  "
        f"RR{sensor_tag(rr_state)}: {n_rr:3d}  "
        f"BPM{sensor_tag(bpm_state)}: {n_bpm:3d}  "
        f"[{status}]   ",
        end="", flush=True
    )


# ═════════════════════════════════════════════════════════════
# BACKGROUND LOOP
# ═════════════════════════════════════════════════════════════

def background_loop():
    """
    Runs every second:
    1. Check each sensor for disconnection / first-run timeout
    2. Inject fake placeholders if needed
    3. Refresh the status line
    """
    while True:
        check_and_inject_fake(spo2_buffer, spo2_state)
        check_and_inject_fake(rr_buffer,   rr_state)
        check_and_inject_fake(bpm_buffer,  bpm_state)
        print_status()
        time.sleep(1)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  VitaSync Local Predictor")
    print("═"*55)
    print(f"  SpO2 path:          {SPO2_PATH}")
    print(f"  RR path:            {RR_PATH}")
    print(f"  BPM path:           {BPM_PATH}")
    print(f"  Prediction path:    /{PREDICTION_PATH}")
    print(f"  Window:             {WINDOW_SIZE_SECONDS}s")
    print(f"  Min to predict:     {MIN_READINGS_TO_PREDICT} readings")
    print(f"  Stale threshold:    {STALE_THRESHOLD_SECONDS}s without update")
    print(f"  Fallback delay:     {FALLBACK_DELAY_SECONDS}s before fake injection")
    print(f"  Patient:            Age {PATIENT_AGE} / "
          f"{'M' if PATIENT_GENDER==1 else 'F' if PATIENT_GENDER==0 else 'Unknown'}")
    print(f"  Fake placeholders:  SpO2={FAKE_SPO2}%  "
          f"RR={FAKE_RR}bpm  BPM={FAKE_BPM}bpm")
    print("═"*55)
    print()
    print("  Listening for live sensor data from Firebase.")
    print(f"  If a sensor goes silent for >{STALE_THRESHOLD_SECONDS}s,")
    print(f"  fake values inject automatically and are clearly labeled.")
    print(f"  If no data arrives within {FALLBACK_DELAY_SECONDS}s of startup,")
    print(f"  fake values start automatically for missing sensors.")
    print()
    print("  Press ENTER to run a prediction.")
    print("  Press Ctrl+C to quit.")
    print()

    bg = threading.Thread(target=background_loop, daemon=True)
    bg.start()

    try:
        while True:
            input()
            run_prediction()
    except KeyboardInterrupt:
        print("\n\n  Shutting down. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()