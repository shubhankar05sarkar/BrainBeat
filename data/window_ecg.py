import wfdb
import neurokit2 as nk
import numpy as np
import os
import sys

def get_ecg_windows(record_path, window_sec=10):
    import wfdb
    import numpy as np

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    fs = record.fs

    window_samples = window_sec * fs
    windows = []

    for start in range(0, len(signal) - window_samples, window_samples):
        end = start + window_samples
        windows.append(signal[start:end])

    return np.array(windows), fs


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from feature_extraction.ecg_features_real import extract_ecg_features

ECG_DIR = "data/ecg/mit_bih"
RECORDS = ["100", "101", "102"]
WINDOW_SEC = 10
FS = 360

X = []
y = []

for record_id in RECORDS:
    print(f"\nProcessing ECG record: {record_id}")

    record_path = os.path.join(ECG_DIR, record_id)

    # Load ECG
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, 0]

    # Load annotations
    ann = wfdb.rdann(record_path, "atr")

    # Convert annotation samples to a set for fast lookup
    abnormal_beats = set(
        ann.sample[i]
        for i, sym in enumerate(ann.symbol)
        if sym not in ["N"]  
    )

    window_samples = WINDOW_SEC * FS
    total_samples = len(ecg_signal)

    for start in range(0, total_samples - window_samples, window_samples):
        end = start + window_samples
        window = ecg_signal[start:end]

        try:
            
            ecg_clean = nk.ecg_clean(window, sampling_rate=FS)

           
            signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=FS)
            rpeaks = info["ECG_R_Peaks"]

            
            if len(rpeaks) < 5:
                continue

            # Extract HRV features
            features = extract_ecg_features(ecg_clean, rpeaks, FS)

            # Label window
            label = 0
            for peak in rpeaks:
                global_peak = start + peak
                if global_peak in abnormal_beats:
                    label = 1
                    break

            X.append(features)
            y.append(label)

        except Exception:
            continue

X = np.array(X)
y = np.array(y)

print("\n----------------------------")
print("Total ECG windows:", len(X))
print("Normal windows:", np.sum(y == 0))
print("Abnormal windows:", np.sum(y == 1))
print("Feature shape:", X.shape)


