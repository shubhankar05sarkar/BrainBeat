import numpy as np
from scipy.signal import welch

def bandpower(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])


def extract_features_from_windows(X, fs=256):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)
    }

    feature_matrix = []

    for window in X:  # window shape: channels × samples
        features = []
        for ch in window:
            for band in bands.values():
                features.append(bandpower(ch, fs, band))
        feature_matrix.append(features)

    return np.array(feature_matrix)