import streamlit as st
import mne
import pandas as pd
import numpy as np
import plotly.express as px
from tensorpac import Pac
import matplotlib.pyplot as plt
import tempfile
import os

# --- 1. PAGE SETUP & STYLING ---
st.set_page_config(page_title="NeuroPulse Analytics Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stMetric { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 15px; 
        border: 2px solid #30363d; 
    }
    [data-testid="stSidebar"] { 
        background-image: linear-gradient(#0d1117, #161b22); 
        border-right: 2px solid #30363d; 
    }
    .block-container {
        padding-top: 2rem;
        border-left: 1px solid #30363d;
    }
    h1 { background: -webkit-linear-gradient(#79c0ff, #1f6feb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_resource
def load_data(uploaded_file):
    if uploaded_file is None:
        info = mne.create_info(ch_names=[f'EEG {i+1:03}' for i in range(16)], sfreq=256, ch_types='eeg')
        data = np.random.randn(16, 5000) * 1e-5
        return mne.io.RawArray(data, info)
    
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.pick_types(eeg=True) 
        return raw
    except Exception as e:
        st.error(f"❌ EDF Load Error: {e}")
        return None
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
    st.divider()

if raw_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Channels", len(raw_data.ch_names))
    c2.metric("Sampling Rate", f"{int(raw_data.info['sfreq'])} Hz")
    c3.metric("Duration", f"{round(raw_data.times[-1], 2)}s")

    if nav == "📊 Power Spectrum":
        st.title("Frequency Power Analysis")
        with st.sidebar:
            fmin, fmax = st.slider("Frequency Range (Hz)", 0.5, 100.0, (1.0, 50.0))
        
        with st.spinner("Calculating Spectral Density..."):
            sfreq = raw_data.info['sfreq']
            n_fft = min(2048, raw_data.n_times)
            spectrum = raw_data.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks='eeg', n_fft=n_fft)
            
            psds, freqs = spectrum.get_data(return_freqs=True)
            if psds.size == 0:
                st.warning("⚠️ Could not compute PSD.")
            else:
                avg_psd = 10 * np.log10(psds.mean(axis=0)) 
                df_psd = pd.DataFrame({'Freq': freqs, 'Power': avg_psd})
                fig = px.line(df_psd, x='Freq', y='Power', labels={'Freq': 'Frequency (Hz)', 'Power': 'Power (dB)'})
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color="#30363d", width=2))])
                st.plotly_chart(fig, use_container_width=True)

    elif nav == "🔗 Connectivity Map":
        st.title("Neural Network Synchronization")
        bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 80)}
        with st.sidebar:
            sel_band = st.selectbox("Frequency Band", list(bands.keys()), index=2)
            num_ch = st.slider("Node Count", 2, min(len(raw_data.ch_names), 64), 16)
        
        l_f, h_f = bands[sel_band]
        ch_names = raw_data.ch_names[:num_ch]
        data_filt = raw_data.copy().pick_channels(ch_names).filter(l_f, h_f, verbose=False).get_data()
        corr = np.corrcoef(data_filt)
        fig = px.imshow(corr, x=ch_names, y=ch_names, color_continuous_scale='Viridis', height=800)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
                         shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color="#30363d", width=2))])
        st.plotly_chart(fig, use_container_width=True)

    elif nav == "🌀 PAC Coupling":
        st.title("Advanced PAC Dashboard")
        with st.spinner("Analyzing Phase-Amplitude Coupling..."):
            sf = raw_data.info['sfreq']
            data_raw = raw_data.get_data(picks=[0], stop=int(sf * 5))
            data = data_raw.reshape(1, -1) 
            
            if data.shape[-1] < sf:
                st.warning("⚠️ Signal segment too short for PAC.")
            else:
                f_pha = np.arange(2, 18, 1)
                f_amp = np.arange(60, 200, 2)
                p = Pac(idpac=(2, 0, 0), f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert')
                phases = p.filterfit(sf, data)
                pac_val = phases.mean(0)
                clean_phases = np.angle(phases).flatten()
                clean_phases = clean_phases[np.isfinite(clean_phases)]

                plt.style.use('dark_background')
                fig = plt.figure(figsize=(16, 10))
                fig.patch.set_facecolor('#0b0e14')
                gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[1.5, 1])

                ax_heat = fig.add_subplot(gs[0, 0])
                im = ax_heat.imshow(pac_val, aspect='auto', origin='lower', cmap='jet', extent=[f_pha[0], f_pha[-1], f_amp[0], f_amp[-1]])
                plt.colorbar(im, ax=ax_heat)
                ax_heat.set_title("PAC Comodulogram")

                ax_polar = fig.add_subplot(gs[0, 1], projection='polar')
                if len(clean_phases) > 0:
                    theta = np.linspace(0.0, 2 * np.pi, 18, endpoint=False)
                    radii, _ = np.histogram(clean_phases, bins=18)
                    ax_polar.bar(theta, radii, width=0.3, color='#ff5555', alpha=0.7)
                ax_polar.set_title("Phase Distribution")

                ax_time = fig.add_subplot(gs[1, :])
                ax_time.plot(data[0], color='#58a6ff', lw=1)
                ax_time.set_title("Analyzed Signal Segment")

                plt.tight_layout()
                st.markdown('<div style="border: 2px solid #30363d; border-radius: 10px; padding: 10px;">', unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.caption(" Harish.G | M.Tech v6.0")