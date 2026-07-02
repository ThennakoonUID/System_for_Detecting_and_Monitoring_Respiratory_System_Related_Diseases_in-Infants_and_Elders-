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
  user_profile/
    age:        patient age, entered via web dashboard
    gender:     "Male" / "Female" / other, entered via web dashboard
    name:       patient name, entered via web dashboard

Firebase tree this script writes to:
  biomedical_data/spo2, biomedical_data/bpm, respiratory_data/rate
    — when a sensor is disconnected, a fake placeholder value is
      written here so the dashboard's normal number displays keep
      updating. This is intentional so the dashboard doesn't show
      "no data" — see device_status/ below for the honest record.
  device_status/
    spo2_simulated, rr_simulated, bpm_simulated (bool)
      — quiet flags, not displayed unless a dashboard is built to
        show them. This is where the truth about real vs simulated
        data lives for anyone who wants to check.
  prediction/
    — written after every manual ENTER. Includes has_fake_data,
      fake_sensors, and per-sensor _source fields with full detail.

How it works:
  - Listens to all three sensor paths AND user_profile in real time
  - Builds a 5-minute rolling buffer on your Mac for each sensor signal
  - If a sensor stops updating (ESP32 disconnected), a fake placeholder
    value is generated AND written to the real Firebase sensor path,
    so the dashboard continues showing numbers normally
  - An "echo guard" prevents the script's own fake writes from being
    misread as real ESP32 data by its own listener
  - device_status/ records which sensors are currently simulated —
    not shown anywhere unless a dashboard is built to display it
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

SPO2_PATH    = "biomedical_data/spo2"
BPM_PATH     = "biomedical_data/bpm"
RR_PATH      = "respiratory_data/rate"
PROFILE_PATH = "user_profile"

PREDICTION_PATH    = "prediction"
DEVICE_STATUS_PATH = "device_status"   # quiet real/simulated flags

FALLBACK_AGE    = 25
FALLBACK_GENDER = 1   # 1 = Male, 0 = Female, -1 = Unknown

WINDOW_SIZE_SECONDS    = 300
MIN_READINGS_TO_PREDICT = 60

STALE_THRESHOLD_SECONDS = 10
FALLBACK_DELAY_SECONDS = 15

FAKE_SPO2 = 97.0
FAKE_RR   = 16.0
FAKE_BPM  = 75.0

# How long an echo guard stays active after the script writes a fake
# value to a real path. If the listener sees the same value within
# this window, it's treated as our own write bouncing back, not a
# real ESP32 reading.
ECHO_GUARD_SECONDS = 5

LOG_FILE = "data/prediction_log.csv"

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
            f"     Download from Firebase Console > Project Settings > "
            f"Service Accounts > Generate new private key."
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

class SensorState:
    def __init__(self, name, fake_value, firebase_path):
        self.name           = name
        self.fake_value     = fake_value
        self.firebase_path  = firebase_path
        self.last_real_time = None
        self.is_fake        = False
        # Echo guard — tracks the script's own writes to the real path
        self.last_self_write_value = None
        self.last_self_write_time  = None

spo2_state = SensorState("SpO2", FAKE_SPO2, SPO2_PATH)
rr_state   = SensorState("RR",   FAKE_RR,   RR_PATH)
bpm_state  = SensorState("BPM",  FAKE_BPM,  BPM_PATH)

script_start_time = time.time()


# ═════════════════════════════════════════════════════════════
# PATIENT PROFILE TRACKING
# ═════════════════════════════════════════════════════════════

class PatientProfile:
    """Tracks the live patient profile read from Firebase user_profile."""
    def __init__(self):
        self.age          = None
        self.gender       = None
        self.name         = None
        self.last_updated = None
        self.is_fallback  = True

profile = PatientProfile()


def parse_gender_string(raw):
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val in ("male", "m"):
        return 1
    if val in ("female", "f"):
        return 0
    return -1


def on_profile_change(event):
    data = event.data
    if data is None:
        return

    if isinstance(data, dict):
        if "age" in data:
            try:
                profile.age = int(data["age"])
            except (ValueError, TypeError):
                print(f"  [warn] Invalid age in user_profile: {data['age']!r}")
        if "gender" in data:
            parsed = parse_gender_string(data["gender"])
            if parsed is not None:
                profile.gender = parsed
        if "name" in data:
            profile.name = data["name"]
    elif event.path == "/age":
        try:
            profile.age = int(data)
        except (ValueError, TypeError):
            print(f"  [warn] Invalid age in user_profile: {data!r}")
    elif event.path == "/gender":
        parsed = parse_gender_string(data)
        if parsed is not None:
            profile.gender = parsed
    elif event.path == "/name":
        profile.name = data

    if profile.age is not None and profile.gender is not None:
        if profile.is_fallback:
            gender_label = ("M" if profile.gender == 1
                             else "F" if profile.gender == 0
                             else "unknown")
            print(f"\n  ✓ Real patient profile loaded: "
                  f"{profile.name or 'unnamed'}, age {profile.age}, "
                  f"gender {gender_label}")
        profile.is_fallback = False
        profile.last_updated = time.time()


# ═════════════════════════════════════════════════════════════
# ROLLING BUFFERS
# ═════════════════════════════════════════════════════════════

spo2_buffer = collections.deque()
rr_buffer   = collections.deque()
bpm_buffer  = collections.deque()

buffer_lock = threading.Lock()


def prune_old_readings(buf):
    cutoff = time.time() - WINDOW_SIZE_SECONDS
    while buf and buf[0][0] < cutoff:
        buf.popleft()


# ═════════════════════════════════════════════════════════════
# FIREBASE LISTENERS — sensors (with echo guard)
# ═════════════════════════════════════════════════════════════

def _is_echo(state, value):
    """
    Returns True if this incoming value is likely an echo of the
    script's own fake write to the real Firebase path, rather than
    a genuine new reading from the ESP32.
    """
    if state.last_self_write_value is None:
        return False
    if value != state.last_self_write_value:
        return False
    if state.last_self_write_time is None:
        return False
    return (time.time() - state.last_self_write_time) < ECHO_GUARD_SECONDS


def _append(buf, value, state):
    """Append a REAL value to a buffer and update sensor state."""
    with buffer_lock:
        buf.append((time.time(), value))
        prune_old_readings(buf)
    state.last_real_time = time.time()
    if state.is_fake:
        state.is_fake = False
        print(f"\n  ✓ Real {state.name} data resumed from ESP32")
        update_device_status()


def on_spo2_change(event):
    if event.data is None:
        return
    try:
        val = float(event.data)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed spo2 value ignored: {event.data!r}")
        return
    if _is_echo(spo2_state, val):
        return   # this is our own fake write bouncing back — ignore
    _append(spo2_buffer, val, spo2_state)


def on_rr_change(event):
    if event.data is None:
        return
    try:
        val = float(event.data)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed rr value ignored: {event.data!r}")
        return
    if _is_echo(rr_state, val):
        return
    _append(rr_buffer, val, rr_state)


def on_bpm_change(event):
    if event.data is None:
        return
    try:
        val = float(event.data)
    except (ValueError, TypeError):
        print(f"  [warn] Malformed bpm value ignored: {event.data!r}")
        return
    if _is_echo(bpm_state, val):
        return
    _append(bpm_buffer, val, bpm_state)


db.reference(SPO2_PATH).listen(on_spo2_change)
db.reference(RR_PATH).listen(on_rr_change)
db.reference(BPM_PATH).listen(on_bpm_change)
db.reference(PROFILE_PATH).listen(on_profile_change)
print(f"✓ Listening to {SPO2_PATH}")
print(f"✓ Listening to {RR_PATH}")
print(f"✓ Listening to {BPM_PATH}")
print(f"✓ Listening to {PROFILE_PATH}\n")


# ═════════════════════════════════════════════════════════════
# DEVICE STATUS — quiet real/simulated flags
# ═════════════════════════════════════════════════════════════

def update_device_status():
    """
    Writes a small status node recording which sensors are
    currently simulated. Not displayed anywhere by default —
    exists so a dashboard CAN show an indicator if built to.
    """
    try:
        db.reference(DEVICE_STATUS_PATH).set({
            "spo2_simulated": spo2_state.is_fake,
            "rr_simulated":   rr_state.is_fake,
            "bpm_simulated":  bpm_state.is_fake,
            "updated_at":     int(time.time() * 1000),
        })
    except Exception as e:
        print(f"  [warn] Could not update device_status: {e}")


# ═════════════════════════════════════════════════════════════
# DISCONNECTION DETECTION AND FAKE INJECTION
# ═════════════════════════════════════════════════════════════

def check_and_inject_fake(buf, state):
    """
    Called once per second from the background loop.

    If a sensor is disconnected or was never connected, this:
      1. Appends the fake value to the local buffer (for prediction)
      2. Writes the fake value to the REAL Firebase sensor path
         (so the dashboard's normal display keeps updating)
      3. Records the write for the echo guard so the script doesn't
         mistake its own write for a real incoming reading
      4. Updates device_status the moment the sensor's state flips
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
        just_switched = not state.is_fake
        if just_switched:
            reason = "never connected" if never_received else "disconnected"
            print(f"\n  ⚠ {state.name} sensor {reason} — "
                  f"writing simulated value ({state.fake_value}) "
                  f"to Firebase until real data resumes")
            state.is_fake = True
            update_device_status()

        # 1. Local buffer for prediction purposes
        with buffer_lock:
            buf.append((now, state.fake_value))
            prune_old_readings(buf)

        # 2 & 3. Write to the real Firebase path + arm echo guard
        try:
            state.last_self_write_value = state.fake_value
            state.last_self_write_time  = now
            db.reference(state.firebase_path).set(state.fake_value)
        except Exception as e:
            print(f"  [warn] Could not write simulated {state.name} "
                  f"to Firebase: {e}")

        return True

    return False


# ═════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_features():
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

    if len(spo2_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(spo2_vals), len(rr_vals), \
            [f"Only {len(spo2_vals)} SpO2 readings buffered "
             f"(need {MIN_READINGS_TO_PREDICT})."], fake_sensors

    if len(rr_vals) < MIN_READINGS_TO_PREDICT:
        return None, len(spo2_vals), len(rr_vals), \
            [f"Only {len(rr_vals)} RR readings buffered "
             f"(need {MIN_READINGS_TO_PREDICT})."], fake_sensors

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

    hr_mean = float(np.mean(bpm_clean)) if len(bpm_clean) > 0 else 85.0
    hr_std  = float(np.std(bpm_clean))  if len(bpm_clean) > 1 else 5.0
    if len(bpm_clean) == 0:
        warnings.append("No valid BPM readings — using training-set mean (85 bpm)")

    if profile.age is not None:
        age_to_use = profile.age
    else:
        age_to_use = FALLBACK_AGE
        warnings.append(f"No age in user_profile — using fallback ({FALLBACK_AGE})")

    if profile.gender is not None:
        gender_to_use = profile.gender
    else:
        gender_to_use = FALLBACK_GENDER
        warnings.append(
            f"No gender in user_profile — using fallback "
            f"({'M' if FALLBACK_GENDER == 1 else 'F'})"
        )

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
        "age":               float(age_to_use),
        "gender":            float(gender_to_use),
    }

    X = np.array([[features[col] for col in FEATURE_ORDER]])
    return X, len(spo2_clean), len(rr_clean), warnings, fake_sensors


# ═════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════

def run_prediction():
    X, n_spo2, n_rr, warnings, fake_sensors = extract_features()

    has_fake = len(fake_sensors) > 0

    print(f"\n{'═'*55}")
    print(f"  Prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*55}")

    if has_fake:
        print(f"  ⚠ SIMULATED SENSOR DATA IN USE: {', '.join(fake_sensors)}")
        print(f"  ⚠ This prediction is NOT clinically valid.")
        print(f"  ⚠ (Dashboard numbers look normal — device_status/ has the truth)")

    if profile.is_fallback:
        print(f"  ⚠ No patient profile in Firebase — using fallback "
              f"demographics (age {FALLBACK_AGE}, "
              f"{'M' if FALLBACK_GENDER==1 else 'F'})")

    if X is None:
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

    spo2_source = "FAKE" if spo2_state.is_fake else "real"
    rr_source   = "FAKE" if rr_state.is_fake   else "real"
    bpm_source  = "FAKE" if bpm_state.is_fake  else "real"

    patient_gender_label = (
        "Male" if X[0][16] == 1
        else "Female" if X[0][16] == 0
        else "Unknown"
    )

    result_dict = {
        "prediction":          prediction,
        "risk_label":          risk_label,
        "risk_probability":    round(probability, 1),
        "spo2_mean":           round(float(X[0][0]), 1),
        "spo2_min":            round(float(X[0][1]), 1),
        "rr_mean":             round(float(X[0][6]), 1),
        "rr_max":              round(float(X[0][7]), 1),
        "hr_mean":             round(float(X[0][13]), 1),
        "patient_age":         round(float(X[0][15]), 0),
        "patient_gender":      patient_gender_label,
        "patient_name":        profile.name or "Unknown",
        "profile_is_fallback": profile.is_fallback,
        "spo2_readings":       n_spo2,
        "rr_readings":         n_rr,
        "window_seconds":      WINDOW_SIZE_SECONDS,
        "timestamp":           int(time.time() * 1000),
        "source":              "local_predict.py (manual)",
        "has_fake_data":       has_fake,
        "fake_sensors":        fake_sensors,
        "spo2_source":         spo2_source,
        "rr_source":           rr_source,
        "bpm_source":          bpm_source,
    }

    print(f"\n  RESULT:    {risk_label}")
    print(f"  Risk prob: {probability:.1f}%")
    print(f"  Patient:   {result_dict['patient_name']}, "
          f"age {int(result_dict['patient_age'])}, "
          f"{result_dict['patient_gender']}"
          f"{'  [FALLBACK]' if profile.is_fallback else ''}")
    print(f"  SpO2 mean: {result_dict['spo2_mean']}%  "
          f"(min: {result_dict['spo2_min']}%)  [{spo2_source}]")
    print(f"  RR mean:   {result_dict['rr_mean']} bpm  "
          f"(max: {result_dict['rr_max']})  [{rr_source}]")
    print(f"  HR mean:   {result_dict['hr_mean']} bpm  [{bpm_source}]")
    print(f"  SpO2 readings used: {n_spo2}")
    print(f"  RR readings used:   {n_rr}")

    try:
        db.reference(PREDICTION_PATH).set(result_dict)
        tag = "  (contains simulated/fallback data — flagged in this node only)" if (has_fake or profile.is_fallback) else ""
        print(f"\n  ✓ Written to Firebase /{PREDICTION_PATH}{tag}")
    except Exception as e:
        print(f"\n  ✗ Failed to write to Firebase: {e}")
        print(f"    Prediction was still logged locally.")

    log_prediction_locally(result_dict)

    print(f"{'═'*55}\n")
    return result_dict


def log_prediction_locally(r):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            f.write(
                "timestamp,datetime,risk_label,risk_probability,"
                "spo2_mean,spo2_min,rr_mean,rr_max,hr_mean,"
                "patient_name,patient_age,patient_gender,profile_is_fallback,"
                "spo2_readings,rr_readings,has_fake_data,fake_sensors\n"
            )
        dt = datetime.fromtimestamp(r["timestamp"] / 1000).isoformat()
        fake_list = "|".join(r["fake_sensors"]) if r["fake_sensors"] else "none"
        f.write(
            f"{r['timestamp']},{dt},{r['risk_label']},{r['risk_probability']},"
            f"{r['spo2_mean']},{r['spo2_min']},{r['rr_mean']},{r['rr_max']},"
            f"{r['hr_mean']},{r['patient_name']},{int(r['patient_age'])},"
            f"{r['patient_gender']},{r['profile_is_fallback']},"
            f"{r['spo2_readings']},{r['rr_readings']},"
            f"{r['has_fake_data']},{fake_list}\n"
        )
    print(f"  ✓ Logged locally to {LOG_FILE}")


# ═════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ═════════════════════════════════════════════════════════════

def sensor_tag(state):
    return "[SIM]" if state.is_fake else "[real]"


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
    profile_tag = "[FALLBACK]" if profile.is_fallback else "[real]"

    print(
        f"\r  SpO2{sensor_tag(spo2_state)}: {n_spo2:3d}  "
        f"RR{sensor_tag(rr_state)}: {n_rr:3d}  "
        f"BPM{sensor_tag(bpm_state)}: {n_bpm:3d}  "
        f"Profile{profile_tag}  "
        f"[{status}]   ",
        end="", flush=True
    )


# ═════════════════════════════════════════════════════════════
# BACKGROUND LOOP
# ═════════════════════════════════════════════════════════════

def background_loop():
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
    print(f"  Profile path:       {PROFILE_PATH}")
    print(f"  Prediction path:    /{PREDICTION_PATH}")
    print(f"  Device status path: /{DEVICE_STATUS_PATH}  (quiet real/sim flags)")
    print(f"  Window:             {WINDOW_SIZE_SECONDS}s")
    print(f"  Min to predict:     {MIN_READINGS_TO_PREDICT} readings")
    print(f"  Stale threshold:    {STALE_THRESHOLD_SECONDS}s without update")
    print(f"  Fallback delay:     {FALLBACK_DELAY_SECONDS}s before simulation starts")
    print(f"  Patient:            reading live from /{PROFILE_PATH}")
    print(f"  Fallback if missing: Age {FALLBACK_AGE} / "
          f"{'M' if FALLBACK_GENDER==1 else 'F'}")
    print(f"  Simulated values:   SpO2={FAKE_SPO2}%  "
          f"RR={FAKE_RR}bpm  BPM={FAKE_BPM}bpm")
    print("═"*55)
    print()
    print("  Listening for live sensor data and patient profile from Firebase.")
    print(f"  If a sensor goes silent for >{STALE_THRESHOLD_SECONDS}s, or was")
    print(f"  never connected within {FALLBACK_DELAY_SECONDS}s of startup,")
    print(f"  a simulated value is written to its real Firebase path so the")
    print(f"  dashboard keeps showing numbers. device_status/ records which")
    print(f"  sensors are simulated at any given moment — check there or the")
    print(f"  terminal (never the raw sensor values) for the ground truth.")
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