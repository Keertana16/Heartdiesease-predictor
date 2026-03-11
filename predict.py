import pickle
import numpy as np

# Load saved model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\n" + "="*70)
print("HEART DISEASE RISK ASSESSMENT - PATIENT PREDICTION")
print("="*70)

# Encoding mappings
sex_map = {'male': 1, 'female': 0}
chest_pain_map = {'typical': 1, 'atypical': 2, 'non-anginal': 3, 'asymptomatic': 4}
restecg_map = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
st_slope_map = {'up': 1, 'flat': 2, 'down': 3}
fastingbs_map = {'no': 0, 'yes': 1}
angina_map = {'no': 0, 'yes': 1}

print("\nEnter patient information:")

# Age and numeric values
age = float(input("Age (years): "))
chol = float(input("Cholesterol (mg/dL): "))
restbp = float(input("Resting Blood Pressure (mmHg): "))
maxhr = float(input("Maximum Heart Rate Achieved: "))
oldpeak = float(input("ST depression induced by exercise: "))

# String-based categorical inputs with clear options
print("\nSex: (Male/Female)")
sex_input = input("Enter Sex: ").lower().strip()
sex = sex_map.get(sex_input, 1)  # Default to Male if invalid

print("\nChest Pain Type: (Typical, Atypical, Non-anginal, Asymptomatic)")
chest_pain_input = input("Enter Chest Pain Type: ").lower().strip()
chest_pain = chest_pain_map.get(chest_pain_input, 1)  # Default to Typical

print("\nResting ECG: (Normal, ST-T abnormality, LV hypertrophy)")
restecg_input = input("Enter Resting ECG: ").lower().strip()
restecg = restecg_map.get(restecg_input, 0)  # Default to Normal

print("\nST Segment Slope: (Up, Flat, Down)")
st_slope_input = input("Enter ST Segment Slope: ").lower().strip()
st_slope = st_slope_map.get(st_slope_input, 1)  # Default to Up

print("\nFasting Blood Sugar > 120 mg/dL?: (Yes/No)")
fastingbs_input = input("Answer: ").lower().strip()
fastingbs = fastingbs_map.get(fastingbs_input, 0)  # Default to No

print("\nExercise Induced Angina?: (Yes/No)")
angina_input = input("Answer: ").lower().strip()
exercise_angina = angina_map.get(angina_input, 0)  # Default to No

# Apply log transformation (same as training data)
chol_transformed = np.log1p(chol)
oldpeak_transformed = np.log1p(oldpeak)

# Create feature array matching model's 15 features
patient_features = np.array([[
    age,
    sex,
    chol_transformed,
    restbp,
    fastingbs,
    maxhr,
    oldpeak_transformed,
    exercise_angina,  # ExerciseAngina
    1 if chest_pain == 2 else 0,  # ChestPainType_ATA
    1 if chest_pain == 3 else 0,  # ChestPainType_NAP
    1 if chest_pain == 4 else 0,  # ChestPainType_TA
    1 if restecg == 1 else 0,     # RestingECG_Normal
    1 if restecg == 2 else 0,     # RestingECG_ST
    1 if st_slope == 2 else 0,    # ST_Slope_Flat
    1 if st_slope == 3 else 0     # ST_Slope_Up
]])

# Make prediction
prediction = model.predict(patient_features)
prediction_prob = model.predict_proba(patient_features)

print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

# Safety-critical threshold = 0.7
threshold = 0.7
prob_disease = prediction_prob[0][1]

print(f"\nDisease Probability: {prob_disease:.2%}")
print(f"Threshold Used: {threshold:.0%}")

if prob_disease > threshold:
    print("\n[ALERT] Patient MAY have heart disease")
    print("        Recommend IMMEDIATE medical consultation")
    print("        Schedule: ECG, Stress test, Cardiology review")
elif prob_disease > 0.5:
    print("\n[CAUTION] Moderate risk detected")
    print("          Patient should consult cardiologist soon")
    print("          Monitor symptoms closely")
else:
    print("\n[HEALTHY] Low risk detected")
    print("          Continue regular health check-ups")
    print("          Maintain healthy lifestyle")

print(f"\nProbability Breakdown:")
print(f"  - Healthy: {prediction_prob[0][0]:.2%}")
print(f"  - Disease Risk: {prediction_prob[0][1]:.2%}")
print("="*70)
