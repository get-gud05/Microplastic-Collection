# =========================
# FINAL FASTAPI BACKEND
# =========================

from fastapi import FastAPI, Request
import numpy as np
import joblib
import time

app = FastAPI()

# -------------------------
# LOAD MODELS
# -------------------------
rf = joblib.load("fusion_rf.pkl")
pca = joblib.load("fusion_pca.pkl")
threshold = joblib.load("fusion_threshold.pkl")

# -------------------------
# STATE
# -------------------------
state = {
    "land": {"data": None, "ts": 0},
    "water": {"data": None, "ts": 0},
    "air": {"data": None, "ts": 0}
}

labels_state = {
    "land": None,
    "water": None,
    "air": None
}

last_result = {
    "risk_score": 0.0,
    "decision": 0
}

# =========================
# TIME SLOT
# =========================
def get_time_slot(ts):
    t = int(ts) % 6
    if t < 2:
        return "land"
    elif t < 4:
        return "water"
    else:
        return "air"

# =========================
# VALIDATION
# =========================
def validate_data(slot, payload):
    try:
        if slot == "water":
            return 0 <= payload["turbidity"] <= 4095
        elif slot == "air":
            return 0 <= payload["gas"] <= 4095
        elif slot == "land":
            return True
    except:
        return False

# =========================
# PREPROCESSING
# =========================
def preprocess_data(slot, payload):

    if slot == "water":
        turb = payload["turbidity"]
        turb_norm = turb / 4095
        return {
            "water_emb": [turb_norm, 0, 0],
            "water_probs": [1 - turb_norm, turb_norm]
        }

    elif slot == "air":
        gas = payload["gas"]
        gas_norm = gas / 4095
        return {
            "air_emb": [gas_norm, 0, 0],
            "air_score": gas_norm
        }

    elif slot == "land":
        return {
            "land_emb": [0.1, 0.2, 0.3]  # replace later with CNN
        }

# =========================
# LABELING
# =========================
def label_data(slot, payload):

    if slot == "water":
        turb = payload["turbidity"]
        if turb < 1000: return "clean"
        elif turb < 2500: return "moderate"
        else: return "polluted"

    elif slot == "air":
        gas = payload["gas"]
        if gas < 1000: return "good"
        elif gas < 2500: return "moderate"
        else: return "hazardous"

    elif slot == "land":
        return "image_received"

# =========================
# NORMALIZE
# =========================
def normalize(x):
    return x / (np.linalg.norm(x) + 1e-8)

# =========================
# FUSION
# =========================
def run_fusion():

    land = state["land"]["data"]
    water = state["water"]["data"]
    air = state["air"]["data"]

    air_emb = normalize(np.array(air["air_emb"]))
    air_score = air["air_score"]

    water_emb = normalize(np.array(water["water_emb"]))
    water_probs = np.array(water["water_probs"])

    land_emb = normalize(np.array(land["land_emb"]))

    x = np.hstack([
        air_emb,
        [air_score],
        water_emb,
        water_probs,
        land_emb
    ])

    x = pca.transform([x])

    prob = rf.predict_proba(x)[0][1]
    decision = int(prob > threshold)

    return prob, decision

# =========================
# SYNC CHECK (FIXED)
# =========================
def is_synced(max_delay=2):
    valid_ts = [state[s]["ts"] for s in state if state[s]["data"] is not None]

    if len(valid_ts) < 3:
        return False

    return max(valid_ts) - min(valid_ts) <= max_delay

# =========================
# INGEST (FINAL)
# =========================
@app.post("/ingest")
async def ingest(request: Request):

    content_type = request.headers.get("content-type")
    ts = int(request.headers.get("x-timestamp", time.time()))

    # =========================
    # IMAGE (ESP / IP CAM)
    # =========================
    if content_type and "application/octet-stream" in content_type:

        data = await request.body()

        # -------- IMAGE VALIDATION --------
        if len(data) < 1000:
            return {"status": "error", "message": "Image corrupted / too small"}

        # -------- SAFE FILE NAME --------
        filename = f"image_{ts}_{int(time.time()*1000)}.jpg"

        with open(filename, "wb") as f:
            f.write(data)

        # -------- FORCE LAND SLOT --------
        slot = "land"

        state["land"] = {
            "data": preprocess_data("land", {}),
            "ts": ts
        }

        labels_state["land"] = "image_received"

        response = {
            "slot": "land",
            "status": "image stored",
            "size": len(data)
        }

    # =========================
    # SENSOR DATA (ESP32)
    # =========================
    else:

        data = await request.json()

        payload = data["data"]
        sensor_type = data["type"]
        ts = data["timestamp"]

        slot = get_time_slot(ts)

        if sensor_type != slot:
            return {"status": "error", "message": f"Expected {slot}, got {sensor_type}"}

        if not validate_data(slot, payload):
            return {"status": "error", "message": "Invalid data"}

        label = label_data(slot, payload)
        labels_state[slot] = label

        processed = preprocess_data(slot, payload)

        state[slot] = {
            "data": processed,
            "ts": ts
        }

        response = {
            "slot": slot,
            "label": label,
            "status": "stored"
        }

    # =========================
    # FUSION
    # =========================
    if all(state[s]["data"] is not None for s in state) and is_synced():
        prob, decision = run_fusion()

        last_result["risk_score"] = float(prob)
        last_result["decision"] = int(decision)

        response["fusion"] = last_result

    return response

# =========================
# OTHER ENDPOINTS
# =========================
@app.get("/predict")
def predict():
    return {
        "risk_score": last_result["risk_score"],
        "microplastic_detected": bool(last_result["decision"])
    }

@app.get("/collect")
def collect():
    if last_result["decision"] == 1:
        return {"action": "COLLECTION ACTIVATED"}
    return {"action": "NO ACTION"}

@app.get("/status")
def status():
    return {
        "labels": labels_state,
        "risk_score": last_result["risk_score"],
        "decision": last_result["decision"]
    }