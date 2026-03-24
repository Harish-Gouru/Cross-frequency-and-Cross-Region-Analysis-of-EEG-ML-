import streamlit as st
import mne
import pandas as pd
import numpy as np
import plotly.express as px
from tensorpac import Pac
import matplotlib.pyplot as plt
import io
import tempfile
import os

# --- 1. PAGE CONFIG & SIMPLE UI ---
st.set_page_config(page_title="NeuroPulse Analytics", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; color: #212529; }
    
    /* Small, Simple Download Buttons */
    .stDownloadButton > button {
        font-size: 11px !important;
        padding: 2px 10px !important;
        background-color: transparent !important;
        color: #007bff !important;
        border: 1px solid #007bff !important;
        border-radius: 4px !important;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
    }
    
    /* Clean Headings without underlines */
    h1 { color: #0056b3; font-weight: 700; border: none; padding-bottom: 0px; font-size: 22px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_resource
def load_data(uploaded_file):
    if uploaded_file is None:
        info = mne.create_info(ch_names=[f'EEG {i+1:03}' for i in range(64)], sfreq=256, ch_types='eeg')
        return mne.io.RawArray(np.random.randn(64, 5000) * 1e-6, info)
    
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        return raw
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🧠 NeuroPulse")
    uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])
    raw_data = load_data(uploaded_file)
    st.divider()
    nav = st.radio("Analysis Suite", ["📊 Power Spectrum", "🔗 Connectivity Map", "🌀 PAC Coupling"])

# --- 4. TOP METRICS ---
c1, c2, c3 = st.columns(3)
c1.metric("Channels", len(raw_data.ch_names))
c2.metric("Sampling Rate", f"{int(raw_data.info['sfreq'])} Hz")
c3.metric("Recording Time", f"{round(raw_data.times[-1], 2)}s")

# --- MODULES ---

if nav == "📊 Power Spectrum":
    col_t, col_btn = st.columns([0.88, 0.12])
    
    # RANGES ADDED BACK TO SIDEBAR AND CALCULATION
    with st.sidebar:
        st.subheader("Filter Range Settings")
        f_low = st.slider("Low Range (Hz)", 0.1, 20.0, 1.0)
        f_high = st.slider("High Range (Hz)", 20.0, 100.0, 50.0)

    col_t.title(f"Power Spectral Density ({f_low}Hz - {f_high}Hz)")
    
    with st.spinner("Calculating..."):
        psd_raw = raw_data.copy().filter(f_low, f_high, verbose=False)
        psd_obj = psd_raw.compute_psd(fmin=f_low, fmax=f_high)
        fig = psd_obj.plot(average=True, spatial_colors=True, show=False)
        fig.patch.set_facecolor('white')
    
    with col_btn:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥", data=buf.getvalue(), file_name="psd_filtered.png")
    
    st.pyplot(fig)

elif nav == "🔗 Connectivity Map":
    col_t, col_btn = st.columns([0.88, 0.12])
    
    with st.sidebar:
        # RANGES ADDED TO WAVE SELECTION
        bands = {
            "Delta (0.5–4Hz)": (0.5, 4), 
            "Theta (4–8Hz)": (4, 8), 
            "Alpha (8–12Hz)": (8, 12), 
            "Beta (13–30Hz)": (13, 30)
        }
        sel_name = st.selectbox("Wave Selection", list(bands.keys()), index=2)
        l_f, h_f = bands[sel_name]
        num_ch = st.slider("Nodes", 2, len(raw_data.ch_names), len(raw_data.ch_names))

    col_t.title(f"Functional Connectivity: {sel_name}")
    
    ch_list = raw_data.ch_names[:num_ch]
    filt = raw_data.copy().filter(l_f, h_f, verbose=False).pick_channels(ch_list)
    corr = filt.to_data_frame().drop(columns=['time']).corr()
    
    # INCREASED HEIGHT FOR CLARITY (800px instead of default)
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', template="plotly_white", height=800)
    
    with col_btn:
        buf = io.StringIO()
        fig.write_html(buf)
        st.download_button("📥", data=buf.getvalue(), file_name="connectivity.html")
    
    st.plotly_chart(fig, use_container_width=True)

elif nav == "🌀 PAC Coupling":
    col_t, col_btn = st.columns([0.88, 0.12])
    col_t.title("Phase-Amplitude Coupling")
    
    target = st.sidebar.selectbox("Electrode", raw_data.ch_names)
    p = Pac(idpac=(1, 0, 0), f_pha=[2, 12], f_amp=[30, 80])
    data = raw_data.get_data(picks=[target])
    phases = p.filterfit(raw_data.info['sfreq'], data)
    
    plt.figure(figsize=(12, 7))
    p.comodulogram(phases.mean(0), title=f"PAC Analysis: {target} (Alpha Phase mod. Gamma Amp.)", cmap='viridis')
    
    with col_btn:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.download_button("📥", data=buf.getvalue(), file_name="pac.png")
    
    st.pyplot(plt.gcf())
    plt.close()

st.divider()
st.caption(" NIT ROURKELA | HARISH GOURU ")