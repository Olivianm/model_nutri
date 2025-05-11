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
try:
    app.logger.info("Loading model and preprocessing tools")
    model = load_model('nutrition_model.h5', compile=False)  # Load .h5 file directly
    label_encoder = joblib.load('label_encoder.pkl')  # Load .pkl file directly
    scaler = joblib.load('scaler.pkl')  # Load .pkl file directly
    
    # Load normalized label classes for validation
    label_classes_normalized = [label.lower() for label in label_encoder.classes_]
    app.logger.info("Model and preprocessing tools loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model files: {str(e)}")
    exit()

@app.route('/')
def home():
    return jsonify({'status': 'Nutrition Model API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received predict request")
        data = request.get_json() if request.is_json else request.form
        if not data:
            app.logger.error("No data received")
            return jsonify({'error': 'No data received'}), 400

        food_name = data.get('food', '').strip().lower()
        app.logger.info(f"Input: food={food_name}")

        if not food_name:
            app.logger.error("No food name provided")
            return jsonify({'error': 'No food name provided'}), 400

        if food_name not in label_classes_normalized:
            app.logger.warning(f"'{food_name}' not found in database")
            # Suggest similar matches
            similar = [food for food in label_classes_normalized if food_name in food][:3]
            return jsonify({
                'error': f"'{food_name}' not found in database",
                'similar': similar
            }), 404

        app.logger.info("Encoding food name")
        food_encoded = label_encoder.transform([food_name])
        food_features = np.array(food_encoded).reshape(1, -1)

        app.logger.info("Making prediction")
        prediction_scaled = model.predict(food_features, verbose=0)
        app.logger.info("Scaling prediction")
        prediction = scaler.inverse_transform(prediction_scaled)

        app.logger.info("Processing nutrients")
        nutrients = {
            col: round(float(val), 4)
            for col, val in zip([
                'Water (g)', 'Protein (g)', 'Fat (g)', 'Total carbohydrate (g)',
                'Cholesterol (mg)', 'Phytosterols (mg)', 'SFA (g)', 'MUFA (g)', 'PUFA (g)'
            ], prediction[0])
        }

        app.logger.info("Prediction successful")
        return jsonify({
            'food': food_name,
            'nutrients': nutrients,
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Use a different port for this app
    app.run(host="0.0.0.0", port=port, debug=False)