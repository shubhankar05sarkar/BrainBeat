import sys
import os
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data.window_ecg import X, y
from models.train_ecg_knn import train_ecg_knn

print("\nECG feature matrix shape:", X.shape)
print("ECG labels shape:", y.shape)

# Train multiple models and select best
best_ecg_model = train_ecg_knn(X, y)

# Save model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_ecg_model, "saved_models/ecg_knn_model.pkl")

print("\nECG model saved at saved_models/ecg_knn_model.pkl")
