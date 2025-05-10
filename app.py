import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Suppress TensorFlow GPU and optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning logs (errors will still show)

# Configure Flask logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow warnings in logs

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for predict endpoint

# Load the model and tools
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nutrition_model.h5')
label_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_encoder.pkl')
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')

try:
    model = load_model(model_path, compile=False)  # Compile False avoids the mse bug
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    
    # Load normalized label classes for validation
    label_classes_normalized = [label.lower() for label in label_encoder.classes_]
    app.logger.info("Model and preprocessing tools loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model files: {str(e)}")
    model = None
    label_encoder = None
    scaler = None

@app.route('/')
def home():
    return jsonify({'status': 'Nutrition Model API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, label_encoder, scaler]):
        app.logger.error("Model or pre-processing tools not loaded properly")
        return jsonify({'error': 'Model or pre-processing tools not loaded properly'}), 500

    try:
        app.logger.info("Received predict request")
        data = request.get_json()
        if not data:
            app.logger.error("No data received")
            return jsonify({'error': 'No data received'}), 400

        food_name = data.get('food', '').strip().lower()
        app.logger.info(f"Input: food_name={food_name}")

        if not food_name:
            app.logger.error("No food name provided")
            return jsonify({'error': 'No food name provided'}), 400

        if food_name not in label_classes_normalized:
            similar = [food for food in label_classes_normalized if food_name in food][:3]
            app.logger.warning(f"'{food_name}' not found in database, suggesting similar: {similar}")
            return jsonify({
                'error': f"'{food_name}' not found in database",
                'similar': similar
            }), 404

        # Encode food name and check shape
        app.logger.info("Encoding food name")
        food_encoded = label_encoder.transform([food_name])
        food_features = np.array(food_encoded).reshape(1, -1)
        app.logger.info(f"Encoded food features shape: {food_features.shape}")

        # Get prediction and apply scaler
        app.logger.info("Making prediction")
        prediction_scaled = model.predict(food_features, verbose=0)
        app.logger.info("Inverse transforming prediction")
        prediction = scaler.inverse_transform(prediction_scaled)

        # Process the nutrients
        app.logger.info("Processing nutrients")
        nutrients = {
            col: round(float(val), 4)
            for col, val in zip([
                'Water (g)', 'Protein (g)', 'Fat (g)', 'Total carbohydrate (g)',
                'Cholesterol (mg)', 'Phytosterols (mg)', 'SFA (g)', 'MUFA (g)', 'PUFA (g)'
            ], prediction[0])
        }

        app.logger.info(f"Prediction successful for {food_name}: {nutrients}")
        return jsonify({
            'food': food_name,
            'nutrients': nutrients,
            'status': 'success'
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)