import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow requests from any origin (can restrict to Lovable domain later)
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

    # --- Alias mapping: UI-friendly → dataset tokens ---
    alias = {
        "cough": ["cough"],
        "sore throat": ["sore_throat"],
        "stomach pain": ["stomach_pain"],
        "diarrhea": ["diarrhea"],
        "constipation": ["constipation"],
        "joint pain": ["joint_pain"],
        "skin rash or itching": ["skin_rash", "itching"],
        "shortness of breath": ["shortness_of_breath"],
        "chest pain": ["chest_pain"],
        "fatigue": ["fatigue"],
        "unexplained weight loss": ["weight_loss"],
        "burning urination": ["burning_micturition"],
        "loss of appetite": ["loss_of_appetite"],
        "leg swelling": ["swollen_legs", "swelling_of_stomach"],
        "vision problems": ["blurred_and_distorted_vision"],
        "frequent urination": ["polyuria"],
        "nausea and vomiting": ["nausea", "vomiting"],
        "back pain": ["back_pain"]
    }

    # Normalize and map
    cleaned = []
    for s in symptoms:
        if not s:
            continue
        s_norm = s.strip().lower()
        if s_norm in alias:
            cleaned.extend(alias[s_norm])
        else:
            cleaned.append(s_norm.replace(" ", "_"))

    # Keep only known symptoms
    known = set(mlb.classes_.tolist())
    filtered = [s for s in cleaned if s in known]

    if not filtered:
        return jsonify({
            "error": "no recognized symptoms",
            "received": symptoms,
            "normalized": cleaned,
            "known_symptoms_example": list(sorted(list(known))[:20])
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
