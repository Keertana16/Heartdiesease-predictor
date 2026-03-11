import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*70)
print("HEART DISEASE MODEL TRAINING")
print("="*70)

# Load data
print("\n[1/7] Loading data...")
data = pd.read_csv('heart.csv')
data = data.dropna()
print(f"Data shape: {data.shape}")

# Data cleaning - Replace zero values with median
print("\n[2/7] Cleaning data...")
data["Cholesterol"] = data["Cholesterol"].replace(0, data["Cholesterol"].median())
data["RestingBP"] = data["RestingBP"].replace(0, data["RestingBP"].median())
data["Oldpeak"] = data["Oldpeak"].fillna(data["Oldpeak"].median())

# Handle outliers - Cap outliers using IQR method
print("\n[3/7] Handling outliers...")
def cap_outliers(col, data):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower, upper)

for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
    cap_outliers(col, data)

# Log transform skewed features
print("\n[4/7] Transforming skewed features...")
data["Oldpeak"] = np.log1p(data["Oldpeak"])
data["Cholesterol"] = np.log1p(data["Cholesterol"])

# Encode categorical variables
print("\n[5/7] Encoding categorical variables...")
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=True)

# Prepare features and target
print("\n[6/7] Preparing features and target...")
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train model
print("\n[7/7] Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
print("\n[Saving Model]")
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✓ Model saved as 'heart_disease_model.pkl'")
print("="*70)
