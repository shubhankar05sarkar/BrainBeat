import joblib
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data.window_eeg import get_eeg_windows
from feature_extraction.extract_eeg_features_real import extract_features_from_windows

EEG_MODEL_PATH = "saved_models/eeg_knn_model.pkl"
eeg_model = joblib.load(EEG_MODEL_PATH)


def eeg_predict(edf_path, seizure_ratio_threshold=0.30):
    
    #Window EEG
    eeg_windows = get_eeg_windows(edf_path)

    if len(eeg_windows) == 0:
        return False

    #Feature extraction
    eeg_features = extract_features_from_windows(eeg_windows)

    # Predict
    preds = eeg_model.predict(eeg_features)

    #Majority logic
    seizure_ratio = np.sum(preds == 1) / len(preds)

    # Clinical decision
    return seizure_ratio >= seizure_ratio_threshold