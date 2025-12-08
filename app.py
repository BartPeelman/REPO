from flask import Flask, request, jsonify
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
import os

app = Flask(__name__)

# Path to the SavedModel folder
model_path = os.path.join("model_repository", "example_model", "1")

# Load the TensorFlow SavedModel using TFSMLayer
model = TFSMLayer(model_path, call_endpoint="serve")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)  # Make sure input is 2D

        # Get predictions
        predictions = model(features).numpy().tolist()

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
