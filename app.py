from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Disease Detector API is running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        features = data.get("features")
        if isinstance(features, str):
            features = [float(x) for x in features.split(",")]
        # Example model logic
        result = "Positive" if sum(features) > 2 else "Negative"
        return jsonify({"prediction": result})
    else:
        return '''
            <form method="POST">
                Enter features (comma separated): <input name="features">
                <input type="submit">
            </form>
        '''