import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow requests from any origin (you can restrict to your Lovable domain later)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# --- Load model & encoders once at startup ---
clf = joblib.load('disease_model.pkl')
mlb = joblib.load('symptom_encoder.pkl')
le  = joblib.load('disease_encoder.pkl')

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "service": "disease-detector"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    if not isinstance(symptoms, list):
        return jsonify({"error": "symptoms must be a list of strings"}), 400

    # Normalize symptoms (lowercase/strip); optional simple aliasing
    cleaned = []
    for s in symptoms:
        if not s:
            continue
        s_norm = s.strip().lower()
        # quick common mappings from natural labels to dataset tokens
        alias = {
            "skin rash or itching": ["skin_rash", "itching"],
            "sore throat": ["sore_throat"],
            "shortness of breath": ["shortness_of_breath"],
            "chest pain": ["chest_pain"],
            "joint pain": ["joint_pain"],
            "stomach pain": ["stomach_pain"],
            "loss of appetite": ["loss_of_appetite"],
            "frequent urination": ["polyuria", "frequent_urination"],  # keep both in case
            "back pain": ["back_pain"],
            "leg swelling": ["swelling_of_stomach","swollen_legs"],   # adapt if in your dataset
        }
        if s_norm in alias:
            cleaned.extend(alias[s_norm])
        else:
            # replace spaces with underscores for simple cases
            cleaned.append(s_norm.replace(" ", "_"))

    # Keep only symptoms that the model knows
    known = set(mlb.classes_.tolist())
    filtered = [s for s in cleaned if s in known]

    if not filtered:
        return jsonify({
            "error": "no recognized symptoms",
            "received": symptoms,
            "normalized": cleaned,
            "known_symptoms_example": list(sorted(list(known))[:20])  # small sample for debugging
        }), 400

    X = mlb.transform([filtered])
    y_pred = clf.predict(X)
    disease = le.inverse_transform(y_pred)[0]

    return jsonify({
        "predicted_disease": disease,
        "recognized_symptoms": filtered
    })

if __name__ == "__main__":
    # Render sets PORT; default to 10000 if not present
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)
