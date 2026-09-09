import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Setup directories
os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

# ==========================================
# 1. WEATHER TEMPERATURE FORECASTING (LSTM)
# ==========================================
print("=== Starting Weather Temperature Forecasting (LSTM) ===")

# Download Jena Climate Dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

# Load first 50,000 rows
df = pd.read_csv(csv_path)
df = df[:50000]

# Extract feature
temp_data = df['T (degC)'].values.reshape(-1, 1)

# Split data (70% train, 15% val, 15% test) sequentially
n = len(temp_data)
train_df = temp_data[0:int(n*0.7)]
val_df = temp_data[int(n*0.7):int(n*0.85)]
test_df = temp_data[int(n*0.85):]

# Fit Scaler on training data ONLY
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

# Save the scaler
with open('models/weather_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved weather_scaler.pkl")

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Model Building function
def build_lstm_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, 1)),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train and evaluate version
def train_and_eval_weather(seq_length, version_name):
    print(f"--- Training Weather LSTM {version_name} (Seq {seq_length}) ---")
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    model = build_lstm_model(seq_length)
    history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_val, y_val), verbose=1)
    
    # Save model
    model_path = f'models/weather_lstm_{version_name.lower()}_seq{seq_length}.h5'
    model.save(model_path)
    print(f"Saved {model_path}")
    
    # Evaluate
    val_loss = history.history['val_loss'][-1]
    
    # Test predictions
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    y_test_actual = scaler.inverse_transform(y_test)
    
    mae = mean_absolute_error(y_test_actual, preds)
    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    
    print(f"Result {version_name}: Val Loss: {val_loss:.4f}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}\n")

train_and_eval_weather(24, 'V1')
train_and_eval_weather(48, 'V2')


# ==========================================
# 2. IMDB SENTIMENT ANALYSIS (GRU)
# ==========================================
print("=== Starting IMDB Sentiment Analysis (GRU) ===")

vocab_size = 1000

# Load IMDB dataset
(X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Merge and split (70% train, 15% val, 15% test)
X_all = np.concatenate((X_train_imdb, X_test_imdb))
y_all = np.concatenate((y_train_imdb, y_test_imdb))

X_train_split, X_temp, y_train_split, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Building function
def build_gru_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length,)),
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate version
def train_and_eval_sentiment(seq_length, version_name):
    print(f"--- Training Sentiment GRU {version_name} (Seq {seq_length}) ---")
    
    # Pad sequences
    X_tr = tf.keras.preprocessing.sequence.pad_sequences(X_train_split, maxlen=seq_length, padding='pre', truncating='pre')
    X_v = tf.keras.preprocessing.sequence.pad_sequences(X_val_split, maxlen=seq_length, padding='pre', truncating='pre')
    X_te = tf.keras.preprocessing.sequence.pad_sequences(X_test_split, maxlen=seq_length, padding='pre', truncating='pre')
    
    model = build_gru_model(seq_length)
    history = model.fit(X_tr, y_train_split, epochs=3, batch_size=128, validation_data=(X_v, y_val_split), verbose=1)
    
    # Save model
    model_path = f'models/sentiment_gru_{version_name.lower()}_seq{seq_length}.h5'
    model.save(model_path)
    print(f"Saved {model_path}")
    
    # Evaluate
    preds_prob = model.predict(X_te)
    preds = (preds_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test_split, preds)
    prec = precision_score(y_test_split, preds)
    rec = recall_score(y_test_split, preds)
    f1 = f1_score(y_test_split, preds)
    
    print(f"Result {version_name}: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}\n")

train_and_eval_sentiment(100, 'V1')
train_and_eval_sentiment(200, 'V2')

print("All training complete!")
