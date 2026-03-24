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

# --- 1. PAGE SETUP & NEON STYLING ---
st.set_page_config(page_title="NeuroPulse Analytics Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stMetric { background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #58a6ff; box-shadow: 0 4px 20px rgba(88, 166, 255, 0.2); }
    [data-testid="stSidebar"] { background-image: linear-gradient(#0d1117, #161b22); border-right: 1px solid #30363d; }
    h1 { background: -webkit-linear-gradient(#79c0ff, #1f6feb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; }
    .stButton>button { background: linear-gradient(90deg, #238636, #2ea043); color: white; border: none; border-radius: 8px; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 15px rgba(46, 160, 67, 0.6); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE: DATA LOADER ---
@st.cache_resource
def load_data(uploaded_file):
    if uploaded_file is None:
        # Dummy data for immediate visual check
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
        st.error(f"⚠️ Load Failure: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧠 NeuroPulse</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Clinical Data (.edf)", type=["edf"])
    raw_data = load_data(uploaded_file)
    
    st.divider()
    nav = st.radio("Switch Module", ["📊 Power Spectrum", "🔗 Connectivity Map", "🌀 PAC Coupling"])
    
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
    st.divider()
    st.caption("M.Tech Production Build v6.0")

# --- 4. TOP METRIC CARDS ---
c1, c2, c3 = st.columns(3)
c1.metric("Channels", len(raw_data.ch_names), "Live")
c2.metric("Sampling Rate", f"{int(raw_data.info['sfreq'])} Hz")
c3.metric("Rec. Length", f"{round(raw_data.times[-1], 2)}s")

# --- MODULE 1: PSD (RECTIFIED) ---
if nav == "📊 Power Spectrum":
    st.title("Frequency Power Analysis")
    with st.spinner("Processing Spectral Curves..."):
        # Fixed 2-step process for new MNE versions
        psd_obj = raw_data.compute_psd(fmax=50)
        fig = psd_obj.plot(average=True, spatial_colors=True, show=False)
        st.pyplot(fig)

# --- MODULE 2: CONNECTIVITY (BEAUTIFIED) ---
elif nav == "🔗 Connectivity Map":
    st.title("Neural Network Synchronization")
    with st.sidebar:
        sel_band = st.selectbox("Frequency Interest", list(bands.keys()), index=2)
        l_f, h_f = bands[sel_band]
        num_ch = st.slider("Node Count", 2, len(raw_data.ch_names), 16)

    col_t, col_btn = st.columns([0.8, 0.2])
    with col_t:
        st.write(f"Displaying interactions for the **{sel_band}** band.")
    
    # Logic
    ch_list = raw_data.ch_names[:num_ch]
    filt = raw_data.copy().filter(l_f, h_f, verbose=False).pick_channels(ch_list)
    corr = filt.to_data_frame().drop(columns=['time']).corr()

    # Using 'Viridis' for a high-contrast colorful look
    fig = px.imshow(corr, color_continuous_scale='Viridis', aspect="auto")
    
    with col_btn:
        buf = io.StringIO()
        fig.write_html(buf)
        st.download_button("📥 Export Map", data=buf.getvalue(), file_name="network_sync.html")

    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 3: PAC (STRICTLY RECTIFIED) ---
elif nav == "🌀 PAC Coupling":
    st.title("Phase-Amplitude Coupling")
    target = st.sidebar.selectbox("Signal Source", raw_data.ch_names)
    
    with st.spinner("Calculating Coupling..."):
        # RECTIFIED: Removed the 'show' parameter to fix your error
        p = Pac(idpac=(1, 0, 0), f_pha=[2, 12], f_amp=[30, 80])
        data = raw_data.get_data(picks=[target])
        phases = p.filterfit(raw_data.info['sfreq'], data)
        
        plt.figure(figsize=(10, 6), facecolor='#0b0e14')
        # This call is now standard and safe
        p.comodulogram(phases.mean(0), title=f"Coupling Analysis: {target}", cmap='inferno')
        
        # We manually use Streamlit to show the figure
        st.pyplot(plt.gcf())
        plt.close()

st.divider()
st.markdown("<p style='text-align: center; color: #8b949e;'>Final Academic Submission Build | System Ready</p>", unsafe_allow_html=True)