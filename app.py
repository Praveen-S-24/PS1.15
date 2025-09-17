from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

@app.route("/")
def home():
    return "🌱 Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            data["Temparature"],
            data["Humidity"],
            data["Moisture"],
            data["Nitrogen"],
            data["Potassium"],
            data["Phosphorous"],
            data["Soil Type"]
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        crop_name = label_encoder.inverse_transform([pred])[0]

        return jsonify({"recommended_crop": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
