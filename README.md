# Week 3 Big Data - LSTM & GRU Models

## 1. Judul Project
Week 3: Time Series Forecasting and Sentiment Analysis using LSTM & GRU

## 2. Tujuan
Memenuhi tugas Week 3 Big Data dengan mengimplementasikan minimal dua model Deep Learning (LSTM & GRU) yang masing-masing memiliki dua versi (V1 dan V2) dengan perbedaan parameter sequence length yang nyata untuk analisis perbandingan kinerja. Proyek ini disiapkan untuk di-deploy di Streamlit Cloud dengan 4 link yang berbeda.

## 3. Dataset
- **Weather Temperature Forecasting**: Jena Climate Dataset (`jena_climate_2009_2016.csv`), menggunakan 50.000 data observasi pertama. Fitur utama adalah `T (degC)`.
- **IMDB Sentiment Analysis**: Keras IMDB Movie Reviews Dataset (`tf.keras.datasets.imdb`), dibatasi dengan 1.000 kata paling sering muncul.

## 4. Model yang Digunakan
1. **Model 1**: Long Short-Term Memory (LSTM) untuk memprediksi nilai suhu (Time Series Forecasting).
2. **Model 2**: Gated Recurrent Unit (GRU) untuk mengklasifikasi sentimen review film (Natural Language Processing).

## 5. Arsitektur
- **LSTM (Weather)**: 
  - `Input(shape=(seq_length, 1))`
  - `LSTM(32, activation='relu')`
  - `Dense(1)`
- **GRU (Sentiment)**:
  - `Input(shape=(seq_length,))`
  - `Embedding(input_dim=1000, output_dim=64)`
  - `GRU(32)`
  - `Dense(1, activation='sigmoid')`

## 6. Preprocessing
- **Weather**: Data dibagi tanpa diacak (sequential) menjadi 70% train, 15% val, 15% test. Fitur di-scale menggunakan `StandardScaler` (hanya di-fit pada data training). Di-reshape menggunakan sliding window approach.
- **Sentiment**: Data digabungkan lalu diacak menggunakan `train_test_split` (70% train, 15% val, 15% test, random_state=42). Sequence disamakan panjangnya menggunakan `pad_sequences` (padding='pre', truncating='pre').

## 7. Versioning
Setiap model memiliki dua versi untuk menguji pengaruh context/sequence length:
- **Weather LSTM V1**: Sequence Length 24
- **Weather LSTM V2**: Sequence Length 48
- **Sentiment GRU V1**: Sequence Length 100
- **Sentiment GRU V2**: Sequence Length 200

## 8. Hasil Evaluasi

### Weather Temperature (LSTM)
| Model | Sequence | Val Loss | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| LSTM V1 | 24 | 0.0011 | 0.0202 | 0.0287 |
| LSTM V2 | 48 | 0.0017 | 0.0256 | 0.0347 |

*Catatan: Sesuai eksperimen, performa LSTM Seq48 justru sedikit menurun dibandingkan Seq24.*

### IMDB Sentiment (GRU)
| Model | Sequence | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| GRU V1 | 100 | 0.8232 | 0.7831 | 0.8999 | 0.8375 |
| GRU V2 | 200 | 0.8612 | 0.8754 | 0.8462 | 0.8605 |

## 9. Cara Menjalankan Lokal

1. Buka terminal di folder project `week3_lstm_gru`.
2. Buat Virtual Environment (opsional): `python -m venv venv`
3. Install requirements: `pip install -r requirements.txt`
4. Jalankan script training untuk menghasilkan `.h5` model: `python train_models.py`
5. Jalankan Streamlit Dashboard: `streamlit run app_dashboard.py`

## 10. Cara Deployment (Streamlit Cloud)

Untuk memenuhi persyaratan tugas dengan **4 link berbeda**, ikuti langkah berikut di Streamlit Cloud:
1. Push repository ini ke GitHub.
2. Di Streamlit Cloud (share.streamlit.io), buat **New App**.
3. Pilih repository GitHub ini.
4. Pada isian **Main file path**, masukkan entry point untuk versi yang diinginkan:
   - Untuk **Weather (V1 & V2)**, gunakan: `week3_lstm_gru/app_weather.py`
   - Untuk **Sentiment (V1 & V2)**, gunakan: `week3_lstm_gru/app_sentiment.py`
5. Dengan ini Anda akan memiliki 2 link terpisah (satu untuk Weather, satu untuk Sentiment), dan di dalam masing-masing aplikasi Anda dapat memilih versi V1 atau V2 secara langsung.

## 11. Struktur Folder
```text
week3_lstm_gru/
├── app.py (Main Logic)
├── app_dashboard.py (Local Dashboard)
├── app_weather.py (Deployment Entrypoint untuk Weather V1 & V2)
├── app_sentiment.py (Deployment Entrypoint untuk Sentiment V1 & V2)
├── train_models.py
├── utils.py
├── requirements.txt
├── README.md
├── VERSIONING.md
└── models/
    ├── weather_lstm_v1_seq24.h5
    ├── weather_lstm_v2_seq48.h5
    ├── sentiment_gru_v1_seq100.h5
    ├── sentiment_gru_v2_seq200.h5
    └── weather_scaler.pkl
```

## 12. Link Deployment
- [LINK DEPLOYMENT WEATHER V1]
- [LINK DEPLOYMENT WEATHER V2]
- [LINK DEPLOYMENT SENTIMENT V1]
- [LINK DEPLOYMENT SENTIMENT V2]

## 13. Perbedaan V1 dan V2
Perbedaan utama pada setiap versi adalah panjang window waktu (sequence length) yang digunakan untuk input model. 
- Pada **Weather LSTM**, peningkatan sequence dari 24 ke 48 justru mengurangi akurasi karena terlalu banyak noise atau vanishing gradient di history yang jauh. 
- Pada **Sentiment GRU**, peningkatan sequence dari 100 ke 200 meningkatkan akurasi karena model mampu mempertahankan lebih banyak konteks review kata demi kata sebelum membuat keputusan akhir.
