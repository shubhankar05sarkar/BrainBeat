import neurokit2 as nk
import numpy as np

def extract_ecg_features(ecg_clean, rpeaks, fs):
    # Time-domain HRV features
    hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)

    features = [
        hrv["HRV_MeanNN"].values[0],   # Average heartbeat interval
        hrv["HRV_SDNN"].values[0],     # Overall variability
        hrv["HRV_RMSSD"].values[0],    # Short-term variability
        hrv["HRV_pNN50"].values[0]     # Irregular heartbeat %
    ]

    return np.array(features)
