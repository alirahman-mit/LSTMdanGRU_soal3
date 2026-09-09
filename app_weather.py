import streamlit as st
import numpy as np
import pandas as pd
from app import (
    get_weather_model, get_scaler,
    render_weather_app
)

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WeatherLSTM · Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main area background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f17 0%, #131320 60%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Sidebar override ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14142a 0%, #0e0e1e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Sidebar brand banner ── */
.sb-brand {
    background: linear-gradient(135deg, #6C63FF 0%, #FF4B8B 100%);
    padding: 28px 24px 22px;
    margin: 0 -1rem 0;
    border-radius: 0 0 16px 16px;
    margin-bottom: 28px;
}
.sb-brand-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.sb-brand-sub {
    font-size: 12px;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.5px;
    margin: 0;
    text-transform: uppercase;
}

/* ── Sidebar section labels ── */
.sb-section-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(255,255,255,0.3);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 12px;
    padding: 0 4px;
}

/* ── Version cards ── */
.ver-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1.5px solid transparent;
}
.ver-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.ver-card-v1 {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(108,99,255,0.04));
    border-color: rgba(108,99,255,0.3);
}
.ver-card-v2 {
    background: linear-gradient(135deg, rgba(255,75,139,0.12), rgba(255,75,139,0.04));
    border-color: rgba(255,75,139,0.3);
}
.ver-card-active-v1 {
    background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(108,99,255,0.10));
    border-color: #6C63FF;
    box-shadow: 0 0 18px rgba(108,99,255,0.25);
}
.ver-card-active-v2 {
    background: linear-gradient(135deg, rgba(255,75,139,0.25), rgba(255,75,139,0.10));
    border-color: #FF4B8B;
    box-shadow: 0 0 18px rgba(255,75,139,0.25);
}
.ver-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 8px;
    border-radius: 20px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.badge-v1 { background: rgba(108,99,255,0.25); color: #a29bfe; }
.badge-v2 { background: rgba(255,75,139,0.25); color: #ff85ab; }
.ver-card-title { font-size: 15px; font-weight: 700; color: #f0f0f0; margin: 0 0 6px; }
.ver-card-text  { font-size: 12.5px; color: rgba(255,255,255,0.5); line-height: 1.6; margin: 0; }

/* ── Stat pill ── */
.stat-row { display: flex; gap: 8px; margin-top: 12px; }
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
}
.stat-pill-val { font-size: 16px; font-weight: 700; color: #fff; }
.stat-pill-key { font-size: 10px; color: rgba(255,255,255,0.35); letter-spacing: 0.5px; margin-top: 2px; }

/* ── Divider ── */
.sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 24px 0;
}

/* ── Sidebar footer ── */
.sb-footer { font-size: 11px; color: rgba(255,255,255,0.2); text-align: center; padding-top: 8px; line-height: 1.6; }

/* ── Main content text overrides ── */
h1, h2, h3 { color: #f0f0f0 !important; }
p, label, .stMarkdown { color: rgba(255,255,255,0.75) !important; }

/* ── Page hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(255,75,139,0.08) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(108,99,255,0.08);
    filter: blur(40px);
}
.hero-tag {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    color: #a29bfe;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -1px;
    margin: 0 0 10px;
    line-height: 1.2;
}
.hero-title span { color: #a29bfe; }
.hero-desc {
    font-size: 15px;
    color: rgba(255,255,255,0.55) !important;
    max-width: 560px;
    line-height: 1.7;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    # Brand Banner
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">🌡️ WeatherLSTM</div>
        <div class="sb-brand-sub">Deep Learning · Time Series</div>
    </div>
    """, unsafe_allow_html=True)

    # Model Selector
    st.markdown('<div class="sb-section-label">Model Version</div>', unsafe_allow_html=True)
    version = st.radio("", ["V1", "V2"], horizontal=True, label_visibility="collapsed")

    # Dynamic Card
    if version == "V1":
        st.markdown("""
        <div class="ver-card ver-card-active-v1">
            <span class="ver-badge badge-v1">● Active</span>
            <div class="ver-card-title">LSTM Baseline</div>
            <div class="ver-card-text">Menggunakan 24 langkah waktu untuk menangkap pola cuaca jangka pendek dengan presisi tinggi.</div>
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-pill-val">24</div>
                    <div class="stat-pill-key">Seq Len</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val">LSTM</div>
                    <div class="stat-pill-key">Arch</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val">V1</div>
                    <div class="stat-pill-key">Version</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ver-card ver-card-active-v2">
            <span class="ver-badge badge-v2">● Active</span>
            <div class="ver-card-title">LSTM Extended</div>
            <div class="ver-card-text">Menggunakan 48 langkah waktu — memori lebih panjang untuk konteks cuaca yang lebih kaya.</div>
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-pill-val">48</div>
                    <div class="stat-pill-key">Seq Len</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val">LSTM</div>
                    <div class="stat-pill-key">Arch</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val">V2</div>
                    <div class="stat-pill-key">Version</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Inactive card (the other version, shown as reference)
    if version == "V1":
        st.markdown("""
        <div class="ver-card ver-card-v2" style="opacity:0.45;">
            <span class="ver-badge badge-v2">V2</span>
            <div class="ver-card-title">LSTM Extended</div>
            <div class="ver-card-text">Sequence 48 · Konteks lebih luas.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ver-card ver-card-v1" style="opacity:0.45;">
            <span class="ver-badge badge-v1">V1</span>
            <div class="ver-card-title">LSTM Baseline</div>
            <div class="ver-card-text">Sequence 24 · Responsif & cepat.</div>
        </div>
        """, unsafe_allow_html=True)

    # Divider + footer
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-footer">
        Tugas Week 3 · Big Data<br>
        LSTM &amp; GRU · Deep Learning
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MAIN — HERO BANNER
# ─────────────────────────────────────────
seq = 24 if version == "V1" else 48
accent = "#a29bfe" if version == "V1" else "#ff85ab"

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">🌤 Temperature Forecasting</div>
    <div class="hero-title">Prediksi Suhu Masa Depan<br>dengan <span>LSTM {version}</span></div>
    <div class="hero-desc">
        Model membaca pola dari <b>{seq} data historis</b> dan secara otomatis memprediksi suhu di langkah berikutnya menggunakan arsitektur <em>Long Short-Term Memory</em>.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MAIN — RENDER LOGIC (dari app.py)
# ─────────────────────────────────────────
render_weather_app(version)
