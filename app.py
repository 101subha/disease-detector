from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ✅ Load ML model
model = joblib.load("disease_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
symptom_encoder = joblib.load("symptom_encoder.joblib")

# ✅ Disease Info (Top 15 Common)
disease_info = {

    "Fungal infection": {
        "description": "A fungal infection affects the skin, hair, or nails caused by fungi.",
        "precautions": [
            "Keep affected area clean and dry",
            "Use antifungal creams",
            "Avoid sharing personal items",
            "Wear breathable clothing"
        ]
    },

    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract.",
        "precautions": [
            "Drink warm fluids",
            "Take proper rest",
            "Wash hands frequently",
            "Avoid cold exposure"
        ]
    },

    "Migraine": {
        "description": "A neurological condition causing intense headaches.",
        "precautions": [
            "Avoid bright light and noise",
            "Maintain sleep schedule",
            "Stay hydrated",
            "Reduce stress"
        ]
    },

    "Malaria": {
        "description": "A mosquito-borne disease causing fever and chills.",
        "precautions": [
            "Use mosquito nets",
            "Apply insect repellent",
            "Avoid stagnant water",
            "Wear full sleeves"
        ]
    },

    "Dengue": {
        "description": "A viral infection transmitted by mosquitoes.",
        "precautions": [
            "Avoid mosquito bites",
            "Stay hydrated",
            "Use mosquito repellents",
            "Keep surroundings clean"
        ]
    },

    "Typhoid": {
        "description": "A bacterial infection spread through contaminated food and water.",
        "precautions": [
            "Drink clean water",
            "Maintain hygiene",
            "Avoid street food",
            "Wash hands regularly"
        ]
    },

    "Diabetes": {
        "description": "A chronic condition affecting blood sugar levels.",
        "precautions": [
            "Maintain healthy diet",
            "Exercise regularly",
            "Monitor blood sugar",
            "Avoid sugary foods"
        ]
    },

    "Hypertension": {
        "description": "A condition of high blood pressure.",
        "precautions": [
            "Reduce salt intake",
            "Exercise regularly",
            "Avoid stress",
            "Maintain healthy weight"
        ]
    },

    "Asthma": {
        "description": "A respiratory condition causing breathing difficulty.",
        "precautions": [
            "Avoid allergens",
            "Use inhaler as prescribed",
            "Stay in clean environment",
            "Avoid smoke"
        ]
    },

    "Pneumonia": {
        "description": "An infection that inflames air sacs in the lungs.",
        "precautions": [
            "Take proper rest",
            "Stay hydrated",
            "Avoid smoking",
            "Consult doctor early"
        ]
    },

    "Allergy": {
        "description": "A reaction of the immune system to substances.",
        "precautions": [
            "Avoid allergens",
            "Take antihistamines",
            "Keep surroundings clean",
            "Use masks if needed"
        ]
    },

    "Food Poisoning": {
        "description": "Illness caused by contaminated food.",
        "precautions": [
            "Eat fresh food",
            "Maintain hygiene",
            "Drink safe water",
            "Avoid stale food"
        ]
    },

    "Acne": {
        "description": "A skin condition causing pimples and spots.",
        "precautions": [
            "Keep face clean",
            "Avoid oily products",
            "Drink water",
            "Avoid touching face"
        ]
    },

    "Arthritis": {
        "description": "Inflammation of joints causing pain and stiffness.",
        "precautions": [
            "Exercise regularly",
            "Maintain weight",
            "Apply hot/cold therapy",
            "Consult doctor"
        ]
    },

    "Jaundice": {
        "description": "A condition causing yellowing of skin and eyes.",
        "precautions": [
            "Drink clean water",
            "Eat healthy food",
            "Avoid alcohol",
            "Take proper rest"
        ]
    }
}


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
        # ✅ Convert symptoms
        input_vector = symptom_encoder.transform([symptoms])

        # ✅ Predict probabilities
        probabilities = model.predict_proba(input_vector)[0]

        # ✅ Top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]

        results = []
        for i in top_indices:
            disease = label_encoder.inverse_transform([i])[0]
            prob = float(probabilities[i])

            info = disease_info.get(disease, {
                "description": "Information not available. Please consult a doctor.",
                "precautions": ["Consult a healthcare professional."]
            })

            results.append({
                "disease": disease,
                "probability": prob,
                "description": info["description"],
                "precautions": info["precautions"]
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
