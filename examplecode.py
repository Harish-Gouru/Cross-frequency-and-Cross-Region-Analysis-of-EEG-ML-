import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import mne
import tempfile
import os

# =================================================
# Page Configuration
# =================================================
st.set_page_config(page_title="EEG Analysis Tool", layout="wide")

st.title("🧠 EEG Cross-Frequency & Cross-Region Analysis Tool")
st.write(
    "This tool performs Phase–Amplitude (PAC), Phase–Phase (PPC), and "
    "Amplitude–Amplitude (AAC) coupling analysis on EEG data."
)

# =================================================
# Sidebar – Data Source
# =================================================
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Select EEG Source",
    [
        "Simulated EEG",
        "Upload EEG Dataset (≤ 200 MB)",
        "Large EEG Dataset (Local Path)"
    ]
)

# =================================================
# Sidebar – Analysis Type
# =================================================
st.sidebar.header("Analysis Type")

analysis_type = st.sidebar.selectbox(
    "Select Coupling Analysis",
    [
        "Phase–Amplitude Coupling (PAC)",
        "Phase–Phase Coupling (PPC)",
        "Amplitude–Amplitude Coupling (AAC)"
    ]
)

# =================================================
# Frequency Bands (Fixed for Comodulogram)
# =================================================
phase_freqs = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13)
}

amp_freqs = {
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}

# =================================================
# Bandpass Filter
# =================================================
def bandpass_filter(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

# =================================================
# EEG DATA LOADING
# =================================================
if data_source == "Simulated EEG":
    fs = 250
    t = np.linspace(0, 10, fs * 10)

    eeg_signal = (
        np.sin(2*np.pi*2*t) +
        np.sin(2*np.pi*6*t) +
        np.sin(2*np.pi*10*t) +
        np.sin(2*np.pi*20*t) +
        np.sin(2*np.pi*35*t)
    )

    st.success("Using simulated EEG signal (fs = 250 Hz)")

elif data_source == "Upload EEG Dataset (≤ 200 MB)":
    uploaded_file = st.sidebar.file_uploader("Upload EEG EDF file", type=["edf"])
    if uploaded_file is None:
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

else:
    file_path = st.sidebar.text_input(
        "Enter full path to EEG EDF file",
        placeholder="e.g. C:\\EEG_Data\\subject01.edf"
    )
    if not file_path or not os.path.exists(file_path):
        st.stop()

    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

# =================================================
# PREPROCESSING
# =================================================
if data_source != "Simulated EEG":
    raw.pick_types(eeg=True)
    raw.filter(0.5, 45, verbose=False)
    raw.notch_filter(50, verbose=False)

    fs = int(raw.info["sfreq"])
    data = raw.get_data()

    channel = st.sidebar.selectbox("Select EEG Channel", raw.ch_names)
    eeg_signal = data[raw.ch_names.index(channel)]
    t = np.arange(len(eeg_signal)) / fs

    st.success(f"EEG Loaded | fs = {fs} Hz | Channel = {channel}")

# =================================================
# RAW EEG SIGNAL
# =================================================
st.subheader("Raw EEG Signal")

fig_raw, ax_raw = plt.subplots()
ax_raw.plot(t, eeg_signal)
ax_raw.set_xlabel("Time (s)")
ax_raw.set_ylabel("Amplitude")
st.pyplot(fig_raw)

# =================================================
# PAC COMODULOGRAM
# =================================================
if analysis_type == "Phase–Amplitude Coupling (PAC)":
    st.subheader("PAC Comodulogram")

    n_bins = 18
    comod = np.zeros((len(phase_freqs), len(amp_freqs)))

    for i, (p_name, (pl, ph)) in enumerate(phase_freqs.items()):
        phase_signal = bandpass_filter(eeg_signal, pl, ph, fs)
        phase = np.angle(hilbert(phase_signal))

        for j, (a_name, (al, ah)) in enumerate(amp_freqs.items()):
            amp_signal = bandpass_filter(eeg_signal, al, ah, fs)
            amp_env = np.abs(hilbert(amp_signal))

            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            amp_means = np.zeros(n_bins)

            for k in range(n_bins):
                idx = np.where((phase >= bins[k]) & (phase < bins[k+1]))[0]
                if len(idx) > 0:
                    amp_means[k] = np.mean(amp_env[idx])

            amp_means /= np.sum(amp_means)
            uniform = np.ones(n_bins) / n_bins
            comod[i, j] = np.sum(
                amp_means * np.log((amp_means + 1e-10) / uniform)
            )

    fig, ax = plt.subplots()
    im = ax.imshow(comod, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(amp_freqs)))
    ax.set_yticks(range(len(phase_freqs)))
    ax.set_xticklabels(list(amp_freqs.keys()))
    ax.set_yticklabels(list(phase_freqs.keys()))
    ax.set_xlabel("Amplitude Frequency")
    ax.set_ylabel("Phase Frequency")
    ax.set_title("PAC Comodulogram (Modulation Index)")
    fig.colorbar(im, ax=ax, label="MI")
    st.pyplot(fig)

    st.caption(
        "Higher Modulation Index (MI) values indicate stronger "
        "phase–amplitude coupling between frequency bands."
    )

# =================================================
# PPC
# =================================================
elif analysis_type == "Phase–Phase Coupling (PPC)":
    st.subheader("Phase–Phase Coupling")

    p1 = bandpass_filter(eeg_signal, 4, 8, fs)
    p2 = bandpass_filter(eeg_signal, 8, 13, fs)

    phase_diff = np.angle(hilbert(p1)) - np.angle(hilbert(p2))
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    fig, ax = plt.subplots()
    ax.hist(phase_diff, bins=50)
    ax.set_xlabel("Phase Difference (rad)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase–Phase Coupling (PLV = {plv:.3f})")
    st.pyplot(fig)

    st.caption(
        "Phase–Phase Coupling (PPC) quantifies synchronization between "
        "the phases of two frequency bands using Phase Locking Value (PLV)."
    )

# =================================================
# AAC
# =================================================
else:
    st.subheader("Amplitude–Amplitude Coupling")

    a1 = np.abs(hilbert(bandpass_filter(eeg_signal, 8, 13, fs)))
    a2 = np.abs(hilbert(bandpass_filter(eeg_signal, 13, 30, fs)))

    corr = np.corrcoef(a1, a2)[0, 1]

    fig, ax = plt.subplots()
    ax.scatter(a1[:2000], a2[:2000], s=2)
    ax.set_xlabel("Alpha Envelope")
    ax.set_ylabel("Beta Envelope")
    ax.set_title(f"AAC (Correlation = {corr:.3f})")
    st.pyplot(fig)

    st.caption(
        "Amplitude–Amplitude Coupling (AAC) measures correlation between "
        "power envelopes of two frequency bands."
    )

# =================================================
# Footer
# =================================================
st.markdown("---")
st.markdown(
    "**Final-Year Project | EEG Cross-Frequency Coupling Analysis Tool**"
)
