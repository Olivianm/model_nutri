from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load model and tools
try:
    app.logger.info("Loading model and preprocessing tools")
    model = joblib.load('nutrition_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Normalize label classes for matching input
    label_classes_normalized = [label.lower() for label in label_encoder.classes_]
    app.logger.info("Model and preprocessing tools loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model files: {str(e)}")
    exit()

@app.route('/')
def home():
    return jsonify({'status': 'Nutrition Model API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or request.form
        food_name = data.get('food', '').strip().lower()

        if not food_name:
            return jsonify({'error': 'No food name provided'}), 400

        if food_name not in label_classes_normalized:
            similar = [f for f in label_classes_normalized if food_name in f][:3]
            return jsonify({'error': f"'{food_name}' not found", 'similar': similar}), 404

        # Encode and reshape
        food_encoded = label_encoder.transform([food_name])
        food_features = np.array(food_encoded).reshape(1, -1)

        # Predict
        prediction = model.predict(food_features)

        # Format output
        nutrients = {
            col: round(float(val), 4)
            for col, val in zip([
                'Water (g)', 'Protein (g)', 'Fat (g)', 'Total carbohydrate (g)',
                'Cholesterol (mg)', 'Phytosterols (mg)', 'SFA (g)', 'MUFA (g)', 'PUFA (g)'
            ], prediction[0])
        }

        return jsonify({'food': food_name, 'nutrients': nutrients, 'status': 'success'})

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)