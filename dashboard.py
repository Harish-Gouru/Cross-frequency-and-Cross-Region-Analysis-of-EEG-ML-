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

# --- 1. PAGE CONFIG & CLEAN LIGHT THEME ---
st.set_page_config(page_title="NeuroPulse Analytics", layout="wide", page_icon="🧠")

# Professional Light Theme CSS
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f8f9fa; color: #212529; }
    
    /* Clean Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling - Soft Blue */
    [data-testid="stSidebar"] {
        background-color: #e9ecef;
        border-right: 1px solid #dee2e6;
    }
    
    /* Professional Titles */
    h1 { color: #0056b3; font-weight: 700; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
    h2, h3 { color: #343a40; }
    
    /* Buttons */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE ---
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

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🧠 NeuroPulse")
    st.subheader("Data Management")
    uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])
    raw_data = load_data(uploaded_file)
    
    st.divider()
    nav = st.radio("Analysis Suite", ["📊 Power Spectrum", "🔗 Connectivity Map", "🌀 PAC Coupling"])
    
    st.divider()
    st.info("Status: System Ready")
    st.caption("Institutional Release v6.5")

# --- 4. TOP METRICS ---
c1, c2, c3 = st.columns(3)
with c1: st.metric("Channels", len(raw_data.ch_names))
with c2: st.metric("Sampling Rate", f"{int(raw_data.info['sfreq'])} Hz")
with c3: st.metric("Recording Time", f"{round(raw_data.times[-1], 2)}s")

# --- MODULE 1: PSD (RECTIFIED) ---
if nav == "📊 Power Spectrum":
    st.header("Power Spectral Density")
    with st.spinner("Analyzing frequencies..."):
        # RECTIFIED: 2-step process to avoid AttributeError
        psd_obj = raw_data.compute_psd(fmax=50)
        # We use a white background for the plot to match the UI
        fig = psd_obj.plot(average=True, spatial_colors=True, show=False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)

# --- MODULE 2: CONNECTIVITY ---
elif nav == "🔗 Connectivity Map":
    st.header("Functional Connectivity")
    
    with st.sidebar:
        bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
        sel_band = st.selectbox("Frequency Band", list(bands.keys()), index=2)
        l_f, h_f = bands[sel_band]
        num_ch = st.slider("Electrodes", 2, len(raw_data.ch_names), 16)

    # Process
    ch_list = raw_data.ch_names[:num_ch]
    filt = raw_data.copy().filter(l_f, h_f, verbose=False).pick_channels(ch_list)
    corr = filt.to_data_frame().drop(columns=['time']).corr()

    # Colorful but professional 'RdBu_r' (Standard for research)
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', 
                    title=f"Correlation Matrix ({sel_band})",
                    template="plotly_white")
    
    # Export button in a clean spot
    buf = io.StringIO()
    fig.write_html(buf)
    st.download_button("📥 Export Map as HTML", data=buf.getvalue(), file_name="connectivity.html")
    
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 3: PAC (RECTIFIED) ---
elif nav == "🌀 PAC Coupling":
    st.header("Phase-Amplitude Coupling")
    target = st.sidebar.selectbox("Analysis Electrode", raw_data.ch_names)
    
    with st.spinner("Computing PAC..."):
        # RECTIFIED: Removed 'show' to fix TypeError
        p = Pac(idpac=(1, 0, 0), f_pha=[2, 12], f_amp=[30, 80])
        data = raw_data.get_data(picks=[target])
        phases = p.filterfit(raw_data.info['sfreq'], data)
        
        plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor('white')
        # Professional 'viridis' color map
        p.comodulogram(phases.mean(0), title=f"Coupling Analysis: {target}", cmap='viridis')
        st.pyplot(plt.gcf())
        plt.close()

st.divider()
st.markdown("<p style='text-align: center; color: #6c757d;'>Final Production Build | Professional Light Edition</p>", unsafe_allow_html=True)