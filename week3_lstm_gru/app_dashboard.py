import streamlit as st
from app import render_weather_app, render_sentiment_app

st.set_page_config(page_title="Week 3 Dashboard", page_icon="⚙️", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.markdown("Week 3: LSTM & GRU")

app_choice = st.sidebar.selectbox("Pilih Model:", ["Weather Temperature (LSTM)", "IMDB Sentiment (GRU)"])

if app_choice == "Weather Temperature (LSTM)":
    version_choice = st.sidebar.radio("Pilih Versi:", ["V1 (Seq 24)", "V2 (Seq 48)"])
    if version_choice == "V1 (Seq 24)":
        render_weather_app("V1")
    else:
        render_weather_app("V2")

elif app_choice == "IMDB Sentiment (GRU)":
    version_choice = st.sidebar.radio("Pilih Versi:", ["V1 (Seq 100)", "V2 (Seq 200)"])
    if version_choice == "V1 (Seq 100)":
        render_sentiment_app("V1")
    else:
        render_sentiment_app("V2")

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini digunakan untuk menavigasi model secara lokal. Saat deployment, masing-masing model dapat menggunakan entrypoint yang terpisah.")
