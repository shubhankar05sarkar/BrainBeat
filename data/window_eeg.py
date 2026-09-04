import mne
import numpy as np
import sys
import os
import joblib


def get_eeg_windows(edf_path, window_sec=5, fs=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()

    window_samples = window_sec * fs
    windows = []

    for start in range(0, data.shape[1] - window_samples, window_samples):
        end = start + window_samples
        windows.append(data[:, start:end])

    return np.array(windows)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# PARAMETERS
EDF_FILES = [
    "data/eeg/chb01/chb01_03.edf",
    "data/eeg/chb01/chb01_04.edf",
    "data/eeg/chb01/chb01_15.edf",
    "data/eeg/chb01/chb01_16.edf",
    "data/eeg/chb01/chb01_18.edf",
    "data/eeg/chb01/chb01_21.edf",
    "data/eeg/chb01/chb01_26.edf",
]

SUMMARY_PATH = "data/eeg/chb01/chb01-summary.txt"
WINDOW_SEC = 5
FS = 256 


def get_seizures_from_summary(summary_path, target_edf):
    seizures = []
    inside_target = False
    start_time = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("File Name:"):
                inside_target = target_edf in line
                continue

            if inside_target:
                if "Seizure" in line and "Start Time" in line:
                    start_time = int(line.split(":")[1].split()[0])

                if "Seizure" in line and "End Time" in line and start_time is not None:
                    end_time = int(line.split(":")[1].split()[0])
                    seizures.append((start_time, end_time))
                    start_time = None

    return seizures

X_all = []
y_all = []

for edf_path in EDF_FILES:
    print(f"\nProcessing file: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()  
    n_samples = data.shape[1]

    print("EEG shape:", data.shape)

    edf_name = os.path.basename(edf_path)
    seizures = get_seizures_from_summary(SUMMARY_PATH, edf_name)
    print("Seizure intervals (s):", seizures)

    seizure_samples = [(s * FS, e * FS) for s, e in seizures]
    window_samples = WINDOW_SEC * FS

    for start in range(0, n_samples - window_samples, window_samples):
        end = start + window_samples
        window = data[:, start:end]

        label = 0
        for sz_start, sz_end in seizure_samples:
            if start >= sz_start and end <= sz_end:
                label = 1
                break

        X_all.append(window)
        y_all.append(label)

X_all = np.array(X_all)
y_all = np.array(y_all)

print("\n----------------------------")
print("Total windows:", len(X_all))
print("Seizure windows:", np.sum(y_all))
print("Non-seizure windows:", len(y_all) - np.sum(y_all))
print("----------------------------")

from feature_extraction.extract_eeg_features_real import extract_features_from_windows

X_features = extract_features_from_windows(X_all)
print("Feature matrix shape:", X_features.shape)

from models.train_eeg_knn import train_eeg_knn

eeg_model = train_eeg_knn(X_features, y_all)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(eeg_model, "saved_models/eeg_knn_model.pkl")

print("\nEEG model saved at saved_models/eeg_knn_model.pkl")