import streamlit as st
import numpy as np
import tensorflow as tf
from utils import load_tf_model, load_weather_scaler, preprocess_weather_input, get_imdb_word_index, preprocess_imdb_text

# Caching models to avoid reloading on every interaction
@st.cache_resource
def get_weather_model(version):
    if version == 'V1':
        return load_tf_model('models/weather_lstm_v1_seq24.h5', 'weather')
    else:
        return load_tf_model('models/weather_lstm_v2_seq48.h5', 'weather')

@st.cache_resource
def get_sentiment_model(version):
    if version == 'V1':
        return load_tf_model('models/sentiment_gru_v1_seq100.h5', 'sentiment')
    else:
        return load_tf_model('models/sentiment_gru_v2_seq200.h5', 'sentiment')

@st.cache_resource
def get_scaler():
    return load_weather_scaler('models/weather_scaler.pkl')

@st.cache_resource
def get_word_index():
    return get_imdb_word_index()

def render_weather_app(version):
    seq_length = 24 if version == 'V1' else 48
    accent_hex = "#6C63FF" if version == "V1" else "#FF4B8B"
    
    # ── Input Section ──────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <div style="font-size:13px; font-weight:700; color:{accent_hex}; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
            01 — Input Data
        </div>
        <div style="font-size:24px; font-weight:800; color:#f0f0f0; letter-spacing:-0.5px; margin-bottom:8px;">
            Masukkan Data Historis
        </div>
        <div style="font-size:14px; color:rgba(255,255,255,0.45); line-height:1.6;">
            Masukkan tepat <b style="color:rgba(255,255,255,0.75);">{seq_length} angka</b> yang mewakili suhu berurutan, dipisahkan dengan koma.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_guide = st.columns([3, 1], gap="large")
    
    with col_input:
        default_input = ", ".join([str(round(20 + np.random.randn(), 2)) for _ in range(seq_length)])
        user_input = st.text_area(
            "input_area",
            value=default_input,
            height=160,
            label_visibility="collapsed",
            placeholder=f"Contoh: 20.5, 21.3, 19.8, ... ({seq_length} angka)"
        )
        predict_btn = st.button(
            f"  ⚡  Jalankan Prediksi  ({version})",
            type="primary",
            use_container_width=True
        )

    with col_guide:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 20px;
            height: 100%;
        ">
            <div style="font-size:11px; font-weight:700; color:{accent_hex}; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:14px;">Petunjuk</div>
            <div style="font-size:13px; color:rgba(255,255,255,0.55); line-height:1.8;">
                ✦ Gunakan angka desimal (contoh: <code style="color:#a29bfe;">19.5</code>)<br>
                ✦ Pisahkan setiap angka dengan <code style="color:#a29bfe;">,</code><br>
                ✦ Total harus tepat <b style="color:#fff;">{seq_length}</b> angka<br>
                ✦ Satuan: <b style="color:#fff;">°C</b> (Celsius)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Predict Logic ──────────────────────────────────────
    if predict_btn:
        try:
            input_list = [float(x.strip()) for x in user_input.split(',')]
        except ValueError:
            st.error("⚠️ Format tidak valid — pastikan semua nilai adalah angka dan dipisahkan koma.")
            return

        if len(input_list) != seq_length:
            st.error(f"⚠️ Jumlah data salah — diperlukan **{seq_length}**, Anda memasukkan **{len(input_list)}**.")
            return

        with st.spinner("Model sedang menganalisis pola suhu..."):
            try:
                model = get_weather_model(version)
                scaler = get_scaler()
                processed_input = preprocess_weather_input(input_list, scaler, seq_length)
                pred_scaled = model.predict(processed_input)
                pred_actual = scaler.inverse_transform(pred_scaled)[0][0]
            except Exception:
                st.error("Terjadi kesalahan teknis. Pastikan file model tersedia.")
                return

        # ── Result ────────────────────────────────────────────
        st.markdown(f"""
        <div style="margin: 12px 0 8px;">
            <div style="font-size:13px; font-weight:700; color:{accent_hex}; letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
                02 — Hasil Prediksi
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_result, col_chart = st.columns([1, 2], gap="large")

        with col_result:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(255,75,139,0.08));
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 20px;
                padding: 36px 28px;
                text-align: center;
            ">
                <div style="font-size:12px; font-weight:600; color:rgba(255,255,255,0.4); letter-spacing:2px; text-transform:uppercase; margin-bottom:16px;">Predicted Temperature</div>
                <div style="font-size:56px; font-weight:800; color:#ffffff; letter-spacing:-2px; line-height:1.1;">{pred_actual:.1f}</div>
                <div style="font-size:22px; font-weight:600; color:rgba(255,255,255,0.5); margin-bottom:24px;">°C</div>
                <div style="height:1px; background:rgba(255,255,255,0.08); margin-bottom:20px;"></div>
                <div style="display:flex; justify-content:center; gap:24px;">
                    <div>
                        <div style="font-size:18px; font-weight:700; color:{accent_hex};">{version}</div>
                        <div style="font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Version</div>
                    </div>
                    <div>
                        <div style="font-size:18px; font-weight:700; color:{accent_hex};">{seq_length}</div>
                        <div style="font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Seq Len</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            import pandas as pd
            st.markdown("""
            <div style="font-size:13px; font-weight:600; color:rgba(255,255,255,0.6); margin-bottom:12px;">
                📈 Tren Historis + Hasil Prediksi
            </div>
            """, unsafe_allow_html=True)
            chart_data = pd.DataFrame({"Suhu (°C)": input_list + [pred_actual]})
            st.line_chart(chart_data, height=260, use_container_width=True)
            st.caption(f"✦ Titik terakhir pada grafik = prediksi model LSTM {version}")



def render_sentiment_app(version):
    
    seq_length = 100 if version == 'V1' else 200
    
    st.title("🎬 IMDB Sentiment Analysis")
    st.markdown("### Gated Recurrent Unit (GRU)")
    
    # Model Info Card
    st.info(f"""
    **MODEL**: GRU  
    **VERSION**: {version}  
    **SEQUENCE LENGTH**: {seq_length}  
    
    *Description:*  
    {
    'Baseline sentiment classification menggunakan sequence 100 kata.' if version == 'V1' 
    else 'Sequence diperpanjang menjadi 200 kata untuk mempertahankan konteks review yang lebih banyak, yang meningkatkan Accuracy.'
    }
    """)
    
    user_input = st.text_area("Masukkan review film (dalam bahasa Inggris)...", height=150, placeholder="This movie was absolutely fantastic, I loved every second of it...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a movie review.")
            return
            
        with st.spinner("Analyzing sentiment..."):
            model = get_sentiment_model(version)
            word_index = get_word_index()
            
            # Preprocess
            processed_input = preprocess_imdb_text(user_input, word_index, vocab_size=1000, seq_length=seq_length)
            
            # Predict
            pred_prob = model.predict(processed_input)[0][0]
            
            sentiment = "Positive" if pred_prob >= 0.5 else "Negative"
            confidence = pred_prob if sentiment == "Positive" else 1 - pred_prob
            
            color = "#2e7d32" if sentiment == "Positive" else "#c62828"
            icon = "👍" if sentiment == "Positive" else "👎"
            
        st.success("Analysis Complete!")
        
        # Result Card
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background: {color}; color: white; text-align: center;">
            <h2 style="margin:0; font-size: 1.5rem;">{icon} {sentiment}</h2>
            <h1 style="margin:10px 0; font-size: 2.5rem;">{(confidence*100):.1f}% Confidence</h1>
            <p style="margin:0; opacity: 0.9;">Probability Score: {pred_prob:.4f}</p>
            <p style="margin:5px 0 0 0; opacity: 0.7; font-size: 0.8rem;">GRU Model • Version {version} • Sequence {seq_length}</p>
        </div>
        """, unsafe_allow_html=True)
