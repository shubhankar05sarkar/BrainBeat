#RUN THIS FILE TO SEE THE PROJECT
#STEPS:
#.\venv\Scripts\activate ---> streamlit run frontend/app.py

import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import wfdb
import mne

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from inference.eeg_infer import eeg_predict
from inference.ecg_infer import ecg_predict

st.set_page_config(
    page_title="BrainBeat | Multimodal Seizure Detection",
    page_icon="assets/brainbeat_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; }
h1, h2, h3, p, label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)

LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "web_logo.png")
RISK_ICON = os.path.join(BASE_DIR, "..", "assets", "risk_logo.png")
SAFE_ICON = os.path.join(BASE_DIR, "..", "assets", "safe_logo.png")

with st.sidebar:
    c1, c2 = st.columns([1, 4])

    with c1:
        st.image(LOGO_PATH, width=55)

    with c2:
        st.markdown(
            "<div class='brainbeat-title'>BrainBeat</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.info("""
    **Multimodal Seizure Risk Assessment**
    
    • EEG-based neurophysiological analysis  
    • ECG-based cardiac rhythm analysis  
    • Integrated multimodal risk assessment
    """)
    st.caption("Investigational clinical decision-support tool")

st.markdown("""
<div style="text-align:center;">
<h1>Multimodal Seizure Detection</h1>
<p>Integrated Brain and Heart Signal Analysis</p>
</div>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700&display=swap');

.brainbeat-title {
    font-family: 'Inter', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("EEG Input")
    eeg_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

with col2:
    st.subheader("ECG Input")
    ecg_record = st.text_input(
        "ECG record path",
        placeholder="data/ecg/mit_bih/100"
    )

run_btn = st.button(
    "ANALYZE SIGNALS",
    disabled=not (eeg_file and ecg_record.strip())
)

def plot_eeg(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    channel = raw.ch_names[0]
    data, times = raw.get_data(picks=[channel], return_times=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times[:5000], data[0][:5000])
    ax.set_title(f"EEG Signal ({channel})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    return fig


def plot_ecg(record_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal[:3000])
    ax.set_title("ECG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (mV)")
    return fig

if run_btn:
    temp_eeg = "temp_eeg.edf"

    with open(temp_eeg, "wb") as f:
        f.write(eeg_file.getvalue())

    try:
        st.subheader("Signal Visualization")

        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("**EEG Waveform (Representative Channel)**")
            st.pyplot(plot_eeg(temp_eeg))

        with vcol2:
            st.markdown("**ECG Waveform**")
            st.pyplot(plot_ecg(ecg_record.strip()))

        st.divider()

        with st.spinner("Running EEG inference..."):
            eeg_result = eeg_predict(temp_eeg)

        with st.spinner("Running ECG inference..."):
            ecg_result = ecg_predict(ecg_record.strip())

        st.subheader("Signal Analysis Summary")

        r1, r2 = st.columns(2)

        with r1:
            if eeg_result:
                st.warning("EEG: Seizure-like patterns detected")
            else:
                st.success("EEG: No seizure-like patterns detected")

        with r2:
            if ecg_result:
                st.warning("ECG: Cardiac abnormality patterns detected")
            else:
                st.success("ECG: Normal cardiac rhythm")

        st.divider()

        final_decision = eeg_result or ecg_result
        st.subheader("Overall Seizure Risk Assessment")

        if final_decision:
            c1, c2 = st.columns([1, 12])

            with c1:
                st.image(RISK_ICON, width=75)

            with c2:
                st.error(
                    "**ELEVATED SEIZURE RISK**  \n"
                    "Multimodal analysis indicates potential ictal or pre-ictal patterns. "
                    "Clinical review is recommended."
                )
        else:
            c1, c2 = st.columns([1, 12])

            with c1:
                st.image(SAFE_ICON, width=75)

            with c2:
                st.success(
                    "**LOW SEIZURE RISK**  \n"
                    "Both EEG and ECG are consistent with non-ictal baseline activity."
                )

    except Exception as e:
        st.error(f"Inference error: {str(e)}")

    finally:
        if os.path.exists(temp_eeg):
            os.remove(temp_eeg)

st.divider()
st.caption("BrainBeat | Academic Project | Decision-support only")