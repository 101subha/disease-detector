from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ✅ Load trained model and encoders
model = joblib.load("disease_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
symptom_encoder = joblib.load("symptom_encoder.joblib")


@app.route('/')
def home():
    return "Disease Detector API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        # ✅ Convert symptoms → numerical format
        input_vector = symptom_encoder.transform([symptoms])

        # ✅ Get probabilities
        probabilities = model.predict_proba(input_vector)[0]

        # ✅ Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]

        results = []
        for i in top_indices:
            disease = label_encoder.inverse_transform([i])[0]
            prob = float(probabilities[i])

            results.append({
                "disease": disease,
                "probability": prob
            })

        return jsonify({
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Run locally (Render will ignore this)
if __name__ == "__main__":
    app.run(debug=True)
