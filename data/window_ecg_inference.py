import wfdb
import numpy as np

def get_ecg_windows(record_path, window_sec=10):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    fs = record.fs

    window_samples = window_sec * fs
    windows = []

    for start in range(0, len(signal) - window_samples, window_samples):
        end = start + window_samples
        windows.append(signal[start:end])

    return np.array(windows), fs
