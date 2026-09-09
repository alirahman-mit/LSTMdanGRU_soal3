import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

class MockModel:
    def __init__(self, model_type):
        self.model_type = model_type
        
    def predict(self, input_data):
        if self.model_type == 'weather':
            # return shape (1, 1) for weather prediction
            return np.array([[0.5]])
        else:
            # return shape (1, 1) for sentiment prediction
            return np.array([[0.8]])

def load_weather_scaler(scaler_path='models/weather_scaler.pkl'):
    if not os.path.exists(scaler_path):
        # Return a mock scaler if not found
        class MockScaler:
            def transform(self, x): return x
            def inverse_transform(self, x): return x
        return MockScaler()
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_tf_model(model_path, model_type):
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Using MockModel for UI demonstration.")
        return MockModel(model_type)
        
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Warning: Failed to load {model_path}. Using MockModel for UI demonstration. Error: {e}")
        return MockModel(model_type)

def preprocess_weather_input(input_list, scaler, expected_seq_length):
    if len(input_list) != expected_seq_length:
        raise ValueError(f"Expected {expected_seq_length} values, got {len(input_list)}")
    
    # Reshape and scale
    input_array = np.array(input_list).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    
    # Reshape for LSTM input: (batch_size, sequence_length, features)
    final_input = scaled_input.reshape(1, expected_seq_length, 1)
    return final_input

def get_imdb_word_index():
    try:
        word_index = tf.keras.datasets.imdb.get_word_index()
        return word_index
    except Exception:
        # Return empty dict if offline/unavailable
        return {}

def preprocess_imdb_text(text, word_index, vocab_size=1000, seq_length=100):
    words = text.lower().split()
    
    # Convert words to integers
    encoded = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3
            if idx < vocab_size:
                encoded.append(idx)
            else:
                encoded.append(2) # OOV
        else:
            encoded.append(2) # OOV
            
    # Add start token at the beginning
    final_encoded = [1] + encoded
    
    # Pad sequences
    padded = tf.keras.preprocessing.sequence.pad_sequences([final_encoded], maxlen=seq_length, padding='pre', truncating='pre')
    return padded
