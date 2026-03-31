from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ ADD THIS

app = Flask(__name__)
CORS(app)   # ✅ ENABLE CORS

@app.route('/')
def home():
    return "Disease Detector API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # ✅ Get symptoms from frontend
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # 🔥 Simple demo logic (replace with ML model later)
    # Just example mapping
    if "vomiting" in symptoms and "fatigue" in symptoms:
        disease = "Food Poisoning"
        probability = 0.85
    elif "headache" in symptoms:
        disease = "Migraine"
        probability = 0.75
    else:
        disease = "Common Cold"
        probability = 0.60

    # ✅ IMPORTANT: return in frontend format
    return jsonify({
        "predictions": [
            {"disease": disease, "probability": probability}
        ]
    })
