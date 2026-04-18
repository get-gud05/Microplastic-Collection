from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# -------------------------
# LOAD MODELS
# -------------------------
rf = joblib.load("fusion_rf.pkl")
pca = joblib.load("fusion_pca.pkl")
threshold = joblib.load("fusion_threshold.pkl")

# -------------------------
# HELPER
# -------------------------
def normalize(x):
    return x / (np.linalg.norm(x) + 1e-8)

def run_fusion(air_emb, air_score, water_emb, water_probs, land_emb):
    air_emb = normalize(air_emb)
    water_emb = normalize(water_emb)
    land_emb = normalize(land_emb)

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


@app.post("/predict")
def predict(data: dict):

    air_emb = np.array(data["air_emb"])
    air_score = data["air_score"]
    water_emb = np.array(data["water_emb"])
    water_probs = np.array(data["water_probs"])
    land_emb = np.array(data["land_emb"])

    prob, decision = run_fusion(
        air_emb, air_score, water_emb, water_probs, land_emb
    )

    return {
        "risk_score": float(prob),
        "microplastic_detected": bool(decision)
    }


@app.post("/collect")
def collect(data: dict):

    air_emb = np.array(data["air_emb"])
    air_score = data["air_score"]
    water_emb = np.array(data["water_emb"])
    water_probs = np.array(data["water_probs"])
    land_emb = np.array(data["land_emb"])

    prob, decision = run_fusion(
        air_emb, air_score, water_emb, water_probs, land_emb
    )

    if decision == 1:
        # HERE you trigger hardware (ESP / relay)
        return {
            "action": "COLLECTION ACTIVATED",
            "risk_score": float(prob)
        }
    else:
        return {
            "action": "NO ACTION",
            "risk_score": float(prob)
        }



@app.post("/status")
def status(data: dict):

    air_score = data["air_score"]
    water_probs = data["water_probs"]
    
    land_signal = float(np.mean(data["land_emb"]))

    return {
        "air_quality_score": float(air_score),
        "water_cluster_probs": water_probs,
        "land_signal": land_signal
    }