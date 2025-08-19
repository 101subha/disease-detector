from flask import Flask, request, jsonify
import joblib

# Load model and encoders
model = joblib.load("disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Disease Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "Please provide symptoms"}), 400

    # Convert symptoms into input vector
    try:
        symptom_vector = symptom_encoder.transform([symptoms])
        prediction = model.predict(symptom_vector)
        disease = disease_encoder.inverse_transform(prediction)[0]
        return jsonify({"disease": disease})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
