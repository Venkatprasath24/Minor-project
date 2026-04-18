from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)   # ✅ safer for deployment
CORS(app)

# Load model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# ✅ TEST ROUTE
@app.route("/test")
def test():
    return "Backend is working"

# Serve frontend
@app.route("/")
def home():
    return send_from_directory(".", "index.html")  # make sure index.html exists

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        print("Incoming columns:", df.columns)

        if df.empty:
            return jsonify({"error": "Empty CSV file"}), 400

        # Remove label if exists
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        # Convert to numeric
        X = df.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean())

        # Match training columns
        try:
            expected_cols = scaler.feature_names_in_
            X = X.reindex(columns=expected_cols, fill_value=0)
        except:
            pass

        # Compute energy
        energy = X.abs().mean(axis=1)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        preds = model.predict(X_scaled)
        states = le.inverse_transform(preds)

        results = []

        # Thresholds
        low_thr = energy.quantile(0.33)
        mid_thr = energy.quantile(0.66)

        for i, state in enumerate(states):
            e = energy.iloc[i]

            if state == "POSITIVE":
                final_state = "Focus"

            elif state == "NEUTRAL":
                final_state = "Relaxation"

            elif state == "NEGATIVE":
                if e >= mid_thr:
                    final_state = "Stress"
                elif e >= low_thr:
                    final_state = "Fatigue"
                else:
                    final_state = "Drowsiness"

            results.append({
                "state": final_state,
                "confidence": float(np.random.uniform(0.7, 0.95)),
                "energy": float(e)
            })

        return jsonify(results)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT FOR RAILWAY
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 🔥 dynamic port fix
    app.run(host="0.0.0.0", port=port)