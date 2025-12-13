import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and encoders
model = joblib.load("disease_model.joblib")
mlb = joblib.load("symptom_encoder.joblib")
le = joblib.load("label_encoder.joblib")

@app.route("/")
def home():
    return "Disease Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "Symptoms not provided"}), 400

    # Normalize input
    symptoms = [str(s).strip().lower() for s in data["symptoms"] if str(s).strip()]

    # Keep only known symptoms
    known_symptoms = set(mlb.classes_)
    valid_symptoms = [s for s in symptoms if s in known_symptoms]

    if len(valid_symptoms) == 0:
        return jsonify({
            "error": "No valid symptoms provided",
            "received": symptoms
        }), 400

    # Encode & predict
    X = mlb.transform([valid_symptoms])
    probabilities = model.predict_proba(X)[0]

    top_indices = np.argsort(probabilities)[::-1][:5]

    results = []
    for i in top_indices:
        results.append({
            "disease": le.inverse_transform([i])[0],
            "probability": float(probabilities[i])
        })

    return jsonify({
        "selected_symptoms": valid_symptoms,
        "predictions": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
