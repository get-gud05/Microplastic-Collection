# =========================
# MAIN BACKEND (FASTAPI)
# =========================

from fastapi import FastAPI
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
# GLOBAL STATE (LATEST DATA)
# -------------------------
state = {
    "land": None,
    "water": None,
    "air": None
}

last_result = {
    "risk_score": 0.0,
    "decision": 0
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def normalize(x):
    return x / (np.linalg.norm(x) + 1e-8)

def run_fusion(land, water, air):
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


# -------------------------
# OPTIONAL: TIME SLOT LOGIC
# -------------------------
def get_current_slot():
    t = int(time.time()) % 6

    if 0 <= t < 2:
        return "land"
    elif 2 <= t < 4:
        return "water"
    else:
        return "air"


# =========================
# 1. INGEST (MAIN ENTRY)
# =========================
@app.post("/ingest")
def ingest(data: dict):

    # OPTION A (recommended): slot from hardware
    slot = data.get("slot")

    # OPTION B: backend decides slot
    if slot is None:
        slot = get_current_slot()

    payload = data["data"]

    state[slot] = payload

    response = {
        "active_slot": slot,
        "status": "stored"
    }

    # -------------------------
    # RUN FUSION IF ALL AVAILABLE
    # -------------------------
    if all(state.values()):
        prob, decision = run_fusion(
            state["land"],
            state["water"],
            state["air"]
        )

        last_result["risk_score"] = float(prob)
        last_result["decision"] = int(decision)

        response["fusion"] = last_result

    return response


# =========================
# 2. PREDICT
# =========================
@app.get("/predict")
def predict():
    return {
        "risk_score": last_result["risk_score"],
        "microplastic_detected": bool(last_result["decision"])
    }


# =========================
# 3. COLLECTION TRIGGER
# =========================
@app.get("/collect")
def collect():

    if last_result["decision"] == 1:
        # HERE: trigger ESP / relay / motor
        return {
            "action": "COLLECTION ACTIVATED",
            "risk_score": last_result["risk_score"]
        }

    return {
        "action": "NO ACTION",
        "risk_score": last_result["risk_score"]
    }


# =========================
# 4. STATUS (FRONTEND)
# =========================
@app.get("/status")
def status():

    return {
        "air": state["air"],
        "water": state["water"],
        "land": state["land"],
        "risk_score": last_result["risk_score"],
        "decision": last_result["decision"]
    }