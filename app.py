import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Disable oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

    data = request.get_json()
    food_name = data.get('food', '').strip().lower()

    if not food_name:
        app.logger.error("No food name provided")
        return jsonify({'error': 'No food name provided'}), 400

    if food_name not in label_classes_normalized:
        # Suggest similar matches
        similar = [food for food in label_classes_normalized if food_name in food][:3]
        app.logger.warning(f"Food '{food_name}' not found, suggesting: {similar}")
        return jsonify({
            'error': f"'{food_name}' not found in database",
            'similar': similar
        }), 404

    try:
        app.logger.info(f"Processing prediction for food: {food_name}")
        # Encode food name and check shape
        food_encoded = label_encoder.transform([food_name])
        food_features = np.array(food_encoded).reshape(1, -1)
        app.logger.debug(f"Encoded features shape: {food_features.shape}")

        # Get prediction and apply scaler
        app.logger.info("Making prediction")
        prediction_scaled = model.predict(food_features, verbose=0)
        app.logger.debug(f"Prediction scaled: {prediction_scaled}")
        prediction = scaler.inverse_transform(prediction_scaled)
        app.logger.debug(f"Prediction after scaling: {prediction}")

        # Process the nutrients
        nutrients = {
            col: round(float(val), 4)
            for col, val in zip([
                'Water (g)', 'Protein (g)', 'Fat (g)', 'Total carbohydrate (g)',
                'Cholesterol (mg)', 'Phytosterols (mg)', 'SFA (g)', 'MUFA (g)', 'PUFA (g)'
            ], prediction[0])
        }
        app.logger.info(f"Nutrients predicted: {nutrients}")

        return jsonify({
            'food': food_name,
            'nutrients': nutrients,
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '_main_':
    port = int(os.environ.get("PORT", 5001))  # Use a different port for this app
    app.run(host="0.0.0.0", port=port, debug=False)