from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: heart_disease_model.pkl not found")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract values from form
        age = float(data.get('age'))
        sex = float(data.get('sex'))
        chest_pain = int(float(data.get('chestpain')))
        restbp = float(data.get('restbp'))
        chol = float(data.get('chol'))
        fastingbs = float(data.get('fastingbs'))
        restecg = int(float(data.get('restecg')))
        maxhr = float(data.get('maxhr'))
        angina = float(data.get('angina'))
        oldpeak = float(data.get('oldpeak'))
        stslope = int(float(data.get('stslope')))
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Apply log transformation to match training data
        chol_log = np.log1p(chol)
        oldpeak_log = np.log1p(oldpeak)
        
        # Create one-hot encoded features to match training transformation
        # ChestPainType: 1, 2, 3, 4 -> ChestPainType_2, ChestPainType_3, ChestPainType_4 (drop_first=True)
        chest_pain_2 = 1 if chest_pain == 2 else 0
        chest_pain_3 = 1 if chest_pain == 3 else 0
        chest_pain_4 = 1 if chest_pain == 4 else 0
        
        # RestingECG: 0, 1, 2 -> RestingECG_1, RestingECG_2 (drop_first=True)
        restecg_1 = 1 if restecg == 1 else 0
        restecg_2 = 1 if restecg == 2 else 0
        
        # ST_Slope: 1, 2, 3 -> ST_Slope_2, ST_Slope_3 (drop_first=True)
        stslope_2 = 1 if stslope == 2 else 0
        stslope_3 = 1 if stslope == 3 else 0
        
        # Prepare 15 features in EXACT order from predict.py:
        # Age, Sex, Cholesterol, RestingBP, FastingBS, MaxHR, Oldpeak, ExerciseAngina,
        # ChestPainType_2, ChestPainType_3, ChestPainType_4, RestingECG_1, RestingECG_2, ST_Slope_2, ST_Slope_3
        features = np.array([[
            age, sex, chol_log, restbp, fastingbs, maxhr, oldpeak_log, angina,
            chest_pain_2, chest_pain_3, chest_pain_4, restecg_1, restecg_2, stslope_2, stslope_3
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Risk percentage (probability of disease)
        risk_percentage = probability[1] * 100
        
        # Determine risk level and color
        if risk_percentage < 30:
            risk_level = "Low Risk"
            risk_color = "green"
        elif risk_percentage < 60:
            risk_level = "Moderate Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        return jsonify({
            'success': True,
            'risk_percentage': round(risk_percentage, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probability_healthy': round(probability[0] * 100, 2),
            'probability_disease': round(probability[1] * 100, 2)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
