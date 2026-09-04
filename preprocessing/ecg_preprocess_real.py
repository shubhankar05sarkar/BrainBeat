import wfdb
import neurokit2 as nk
import numpy as np

def load_and_preprocess_ecg(record_path):
    # Load ECG using WFDB
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, 0]  # use first lead
    fs = record.fs

    # Clean ECG signal
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=fs)

    # Detect R-peaks
    signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

    return ecg_clean, info, fs
