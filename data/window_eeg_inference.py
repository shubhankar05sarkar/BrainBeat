import mne
import numpy as np

def get_eeg_windows(edf_path, window_sec=5, fs=256):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()

    window_samples = window_sec * fs
    windows = []

    for start in range(0, data.shape[1] - window_samples, window_samples):
        end = start + window_samples
        windows.append(data[:, start:end])

    return np.array(windows)
