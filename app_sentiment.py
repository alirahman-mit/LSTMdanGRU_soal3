import streamlit as st
from app import render_sentiment_app

st.set_page_config(page_title="IMDB Sentiment (GRU)", page_icon="🎬", layout="centered")

version_choice = st.radio("Pilih Versi Model:", ["V1 (Seq 100)", "V2 (Seq 200)"], horizontal=True)

if version_choice == "V1 (Seq 100)":
    render_sentiment_app(version="V1")
else:
    render_sentiment_app(version="V2")
