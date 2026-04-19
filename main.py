# =========================
# FINAL FASTAPI BACKEND (IP CAMERA VERSION)
# =========================

from fastapi import FastAPI, Request
import numpy as np
import joblib
import time
import requests

app = FastAPI()

# -------------------------
# LOAD MODELS
# -------------------------
rf = joblib.load("fusion_rf.pkl")
pca = joblib.load("fusion_pca.pkl")
threshold = joblib.load("fusion_threshold.pkl")

# -------------------------
# IP CAMERA URL (IPv6 supported)
# -------------------------
IP_CAMERA_URL = "http://[2401:4900:bbab:7818:289b:13ff:fe5f:af6]:8080"

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
# FETCH IMAGE FROM CAMERA
# =========================
def fetch_image_from_camera():
    print("📡 Trying to fetch image...")

    try:
        with requests.get(IP_CAMERA_URL, stream=True, timeout=5) as stream:

            print("Status:", stream.status_code)

            if stream.status_code != 200:
                print("❌ Bad response")
                return None

            bytes_data = b""

            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk

                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    print("✅ Frame extracted")
                    return bytes_data[a:b+2]

        return None

    except Exception as e:
        print("❌ Camera error:", e)
        return None

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
        return "image_captured"

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
# SYNC CHECK
# =========================
def is_synced(max_delay=2):
    valid_ts = [state[s]["ts"] for s in state if state[s]["data"] is not None]

    if len(valid_ts) < 3:
        return False

    return max(valid_ts) - min(valid_ts) <= max_delay

# =========================
# INGEST
# =========================
@app.post("/ingest")
async def ingest(request: Request):

    ts = int(time.time())

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
    # WATER TRIGGER → IMAGE FETCH
    # =========================
    if slot == "water":

        turbidity_value = processed["water_emb"][0]

        # trigger condition (you will replace with ML model)
        if turbidity_value > 0.5:

            image_data = fetch_image_from_camera()

            if image_data is not None:

                filename = f"image_{ts}_{int(time.time()*1000)}.jpg"

                with open(filename, "wb") as f:
                    f.write(image_data)

                state["land"] = {
                    "data": preprocess_data("land", {}),
                    "ts": ts
                }

                labels_state["land"] = "image_captured"

                response["image"] = "captured"

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