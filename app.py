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
    "description": "A condition affecting skin, hair, or nails caused by fungi.",
    "precautions": [
        "Keep affected area clean and dry",
        "Use antifungal medication",
        "Avoid sharing personal items",
        "Wear breathable clothing"
    ]
},

"Allergy": {
    "description": "An immune reaction to allergens like dust, pollen, or food.",
    "precautions": [
        "Avoid known allergens",
        "Keep surroundings clean",
        "Use antihistamines if needed",
        "Wear mask in dusty areas"
    ]
},

"GERD": {
    "description": "A condition where stomach acid flows back into the food pipe.",
    "precautions": [
        "Avoid spicy and oily food",
        "Eat smaller meals",
        "Do not lie down after eating",
        "Maintain healthy weight"
    ]
},

"Chronic cholestasis": {
    "description": "A liver condition where bile flow is reduced or blocked.",
    "precautions": [
        "Avoid alcohol consumption",
        "Follow low-fat diet",
        "Consult doctor regularly",
        "Maintain proper hygiene"
    ]
},

"Drug Reaction": {
    "description": "An unwanted reaction caused by certain medications.",
    "precautions": [
        "Stop the triggering drug",
        "Consult a doctor immediately",
        "Avoid self-medication",
        "Monitor symptoms carefully"
    ]
},

"Peptic ulcer diseae": {
    "description": "Sores that develop in the lining of the stomach or intestine.",
    "precautions": [
        "Avoid spicy food",
        "Reduce stress",
        "Avoid alcohol and smoking",
        "Take prescribed medicines"
    ]
},

"AIDS": {
    "description": "A condition caused by HIV that weakens the immune system.",
    "precautions": [
        "Practice safe sex",
        "Avoid sharing needles",
        "Get regular medical checkups",
        "Follow prescribed treatment"
    ]
},

"Gastroenteritis": {
    "description": "Inflammation of stomach and intestines causing diarrhea.",
    "precautions": [
        "Drink clean and safe water",
        "Maintain proper hygiene",
        "Avoid contaminated food",
        "Stay hydrated"
    ]
},

"Bronchial Asthma": {
    "description": "A respiratory condition causing difficulty in breathing.",
    "precautions": [
        "Avoid dust and allergens",
        "Use inhaler as prescribed",
        "Avoid smoking",
        "Keep environment clean"
    ]
},

"Migraine": {
    "description": "A neurological condition causing severe headaches.",
    "precautions": [
        "Avoid stress and bright lights",
        "Maintain proper sleep",
        "Stay hydrated",
        "Identify and avoid triggers"
    ]
},

"Cervical spondylosis": {
    "description": "A condition affecting neck bones due to aging or strain.",
    "precautions": [
        "Maintain good posture",
        "Avoid heavy lifting",
        "Exercise regularly",
        "Use proper neck support"
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
},

"Malaria": {
    "description": "A mosquito-borne disease causing fever and chills.",
    "precautions": [
        "Use mosquito nets",
        "Apply repellents",
        "Avoid stagnant water",
        "Wear full-sleeve clothes"
    ]
},

"Dengue": {
    "description": "A viral infection spread by mosquitoes causing high fever.",
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
        "Avoid street food",
        "Maintain hygiene",
        "Wash hands regularly"
    ]
},

"Pneumonia": {
    "description": "A lung infection causing breathing difficulty.",
    "precautions": [
        "Take proper rest",
        "Stay hydrated",
        "Avoid smoking",
        "Seek medical attention early"
    ]
},

"Common Cold": {
    "description": "A viral infection affecting nose and throat.",
    "precautions": [
        "Drink warm fluids",
        "Take proper rest",
        "Maintain hygiene",
        "Avoid cold exposure"
    ]
},

"Tuberculosis": {
    "description": "A bacterial infection affecting the lungs.",
    "precautions": [
        "Cover mouth while coughing",
        "Complete full treatment",
        "Avoid crowded places",
        "Maintain proper hygiene"
    ]
}
}

# ✅ Doctor Mapping
doctor_map = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist / General Physician",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "General Physician",
    "Peptic ulcer diseae": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Gastroenteritis": "General Physician",
    "Bronchial Asthma": "Pulmonologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic",
    "Jaundice": "Hepatologist",
    "Malaria": "General Physician",
    "Dengue": "General Physician",
    "Typhoid": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Common Cold": "General Physician",
    "Tuberculosis": "Pulmonologist"
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

            # ✅ Clean fallback
            info = disease_info.get(disease, {
                "description": f"{disease} may require proper medical diagnosis.",
                "precautions": [
                    "Monitor symptoms",
                    "Maintain hygiene",
                    "Stay hydrated",
                    "Consult a doctor"
                ]
            })

            doctor = doctor_map.get(disease, "General Physician")

            results.append({
                "disease": disease,
                "probability": prob,
                "description": info["description"],
                "precautions": info["precautions"],
                "doctor": doctor
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
