# core_logic/model_inference.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading .pkl models (ARIMA, ML, Scaler)
import json    # For loading .json parameters
import yfinance as yf
from tensorflow.keras.models import load_model  # For loading .h5 (LSTM)
import os

# --- Import from your teammates ---
# This assumes Person 2's file is core_logic/analysis.py
try:
    from core_logic.analysis import get_spread_parameters
except ImportError:
    st.error("FATAL ERROR: Could not find 'analysis.py'. Make sure Person 2's file is in 'core_logic/'.")
    # A simple fallback for a common error
    from analysis import get_spread_parameters 

# --- Define File Paths (from Person 3's script) ---
SAVED_MODELS_DIR = "saved_models"
PARAMS_PATH = os.path.join(SAVED_MODELS_DIR, "model_parameters.json")
ARIMA_PATH = os.path.join(SAVED_MODELS_DIR, "arima_model.pkl")
LSTM_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_model.h5")
LSTM_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_scaler.pkl")
ML_PATH = os.path.join(SAVED_MODELS_DIR, "ml_model.pkl")

# =====================================================================
#  1. MODEL LOADER FUNCTIONS (with Caching)
# =====================================================================
# We cache these so they only load from disk once per session

@st.cache_resource
def load_model_parameters():
    """Loads the saved beta, mean, and std dev."""
    with open(PARAMS_PATH, 'r') as f:
        return json.load(f)

@st.cache_resource
def load_lstm_model():
    """Loads the saved LSTM model."""
    return load_model(LSTM_PATH, compile=False)

@st.cache_resource
def load_lstm_scaler():
    """Loads the saved LSTM scaler. CRITICAL!"""
    return joblib.load(LSTM_SCALER_PATH)

@st.cache_resource
def load_ml_model():
    """Loads the saved Random Forest model."""
    return joblib.load(ML_PATH)

# =====================================================================
#  2. LIVE DATA HELPER FUNCTION
# =====================================================================

@st.cache_data(ttl="15m") # Cache this for 15 minutes
# In core_logic/model_inference.py

@st.cache_data(ttl="15m") # Cache this for 15 minutes
def get_live_spread_data():
    """
    Gets the latest ~100 trading days of data.
    We need this much for the 60-day LSTM lookback.
    """
    print("Fetching live data...") # For debugging
    
    params = load_model_parameters()
    beta = params['beta']
    
    # Get last ~100 trading days
    ko = yf.download('KO', period='100d', interval='1d')
    pep = yf.download('PEP', period='100d', interval='1d')
    
    # --- THIS IS THE FIX ---
    # We will use 'Close' instead of 'Adj Close' as it is more reliable
    live_data = pd.concat([ko['Close'], pep['Close']], axis=1)
    # --- END FIX ---
    
    live_data.columns = ['KO', 'PEP']
    live_data = live_data.dropna()
    
    # Calculate the live spread
    live_data['Spread'] = np.log(live_data['KO']) - beta * np.log(live_data['PEP'])
    
    return live_data['Spread']

# =====================================================================
#  3. YOUR FOUR SIGNAL FUNCTIONS
# =====================================================================
#
# Person 6 will import these functions.
# Each function must return: (Signal_String, Details_String)

def get_zscore_signal():
    """Calculates the signal based on the classic Z-Score."""
    params = load_model_parameters()
    mean = params['mean']
    std_dev = params['std_dev']
    
    live_spread = get_live_spread_data()
    current_spread = live_spread.iloc[-1]
    
    # Calculate the live Z-Score
    z_score = (current_spread - mean) / std_dev
    
    # Define trading logic
    if z_score < -2.0:
        return "BUY", f"Z-Score: {z_score:.2f} (Strong Reversion Buy)"
    elif z_score > 2.0:
        return "SELL", f"Z-Score: {z_score:.2f} (Strong Reversion Sell)"
    else:
        return "HOLD", f"Z-Score: {z_score:.2f} (Near Mean)"

def get_arima_signal():
    """
    Forecasts using the saved ARIMA model.
    NOTE: For ARIMA to be truly 'live', it must be updated with new data.
    This is slow, so we load, update, and predict all at once.
    """
    # We load the model *inside* the function (no cache)
    # because we are going to update it with live data.
    try:
        model = joblib.load(ARIMA_PATH)
    except FileNotFoundError:
        return "ERROR", "ARIMA model file not found."
    
    live_spread = get_live_spread_data()
    current_spread = live_spread.iloc[-1]
    
    # Update the model with the latest data
    # This 'updates' the model's state to the present
    model.update(live_spread) 
    
    # Forecast 1 step ahead from its new 'present'
    forecast = model.predict(n_periods=1)[0]
    
    if forecast > current_spread:
        return "BUY", f"Forecast: {forecast:.4f} (Predicts spread will rise)"
    else:
        return "SELL", f"Forecast: {forecast:.4f} (Predicts spread will fall)"

def get_lstm_signal():
    """Forecasts using the saved LSTM model."""
    model = load_lstm_model()
    scaler = load_lstm_scaler()
    live_spread = get_live_spread_data()
    current_spread = live_spread.iloc[-1]
    
    # This MUST match Person 3's script: n_input=60
    N_LOOKBACK = 60 
    
    # 1. Get the last 60 days
    last_60_days = live_spread.iloc[-N_LOOKBACK:].values.reshape(-1, 1)
    
    # 2. Scale the data
    scaled_data = scaler.transform(last_60_days)
    
    # 3. Reshape for LSTM: [samples, timesteps, features]
    data_for_pred = scaled_data.reshape((1, N_LOOKBACK, 1))
    
    # 4. Predict
    scaled_prediction = model.predict(data_for_pred)[0][0]
    
    # 5. CRITICAL: Un-scale the prediction
    final_prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]
    
    if final_prediction > current_spread:
        return "BUY", f"Forecast: {final_prediction:.4f} (Predicts spread will rise)"
    else:
        return "SELL", f"Forecast: {final_prediction:.4f} (Predicts spread will fall)"

def get_ml_signal():
    """Forecasts using the saved Random Forest model."""
    model = load_ml_model()
    live_spread = get_live_spread_data()
    current_spread = live_spread.iloc[-1]
    
    # This MUST match Person 3's script: n_lags=5
    N_LAGS = 5
    
    # 1. Get the last 5 days (for lag_1...lag_5)
    last_5_days = live_spread.iloc[-(N_LAGS + 1) : -1].values # Get [day-5, day-4, day-3, day-2, day-1]
    
    # 2. Reshape for the model. Order is [lag_1, lag_2, ..., lag_5]
    # We reverse the array to get [day-1, day-2, ... day-5]
    features_for_pred = np.array(last_5_days[::-1]).reshape(1, -1)
    
    # 3. Predict
    prediction = model.predict(features_for_pred)[0]
    
    if prediction > current_spread:
        return "BUY", f"Forecast: {prediction:.4f} (Predicts spread will rise)"
    else:
        return "SELL", f"Forecast: {prediction:.4f} (Predicts spread will fall)"