import joblib
import numpy as np
import os
import sys
import neurokit2 as nk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data.window_ecg import get_ecg_windows
from feature_extraction.ecg_features_real import extract_ecg_features

ECG_MODEL_PATH = "saved_models/ecg_knn_model.pkl"
ecg_model = joblib.load(ECG_MODEL_PATH)


def ecg_predict(record_path, abnormal_ratio_threshold=0.30):

    windows, fs = get_ecg_windows(record_path)

    if len(windows) == 0:
        return False

    feature_list = []

    for window in windows:
        try:
            ecg_clean = nk.ecg_clean(window, sampling_rate=fs)
            _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
            rpeaks = info["ECG_R_Peaks"]

            # Skip poor-quality windows
            if len(rpeaks) < 5:
                continue

            features = extract_ecg_features(ecg_clean, rpeaks, fs)
            feature_list.append(features)

        except Exception:
            continue

    if len(feature_list) == 0:
        return False

    X = np.array(feature_list)
    preds = ecg_model.predict(X)

    # ---- Majority logic ----
    abnormal_ratio = np.sum(preds == 1) / len(preds)

    return abnormal_ratio >= abnormal_ratio_threshold
