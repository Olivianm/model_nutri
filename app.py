from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model and preprocessing artifacts
model = joblib.load('nutrition_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Create a reverse mapping for food names
food_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
food_options = sorted([food.title() for food in food_names])

@app.route('/')
def home():
    return render_template('index.html', food_options=food_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get food name from form
        food_name = request.form['food_name'].lower().strip()
        
        # Encode the food name
        try:
            food_encoded = label_encoder.transform([food_name])[0]
        except ValueError:
            return jsonify({'error': 'Food not found in database'})
        
        # Scale the feature
        X_scaled = scaler.transform([[food_encoded]])
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Prepare results
        nutrients = [
            'Water (g)', 'Protein (g)', 'Fat (g)', 'Total carbohydrate (g)',
            'Cholesterol (mg)', 'Phytosterols (mg)', 'SFA (g)', 'MUFA (g)', 'PUFA (g)'
        ]
        
        results = {
            'food': food_name.title(),
            'nutrients': {nutrient: round(value, 2) for nutrient, value in zip(nutrients, prediction)}
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)