from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "feature_columns.json"

model = joblib.load(MODEL_PATH)
with FEATURES_PATH.open("r", encoding="utf-8") as handle:
    feature_columns = json.load(handle)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features", [])
    if len(features) != len(feature_columns):
        return jsonify({"error": f"Expected {len(feature_columns)} features"}), 400

    df = pd.DataFrame([features], columns=feature_columns)
    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(model.predict(df)[0])
    return jsonify({"prediction": prediction, "probability": probability})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
