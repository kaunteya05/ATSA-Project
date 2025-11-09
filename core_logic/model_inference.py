import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf
from tensorflow.keras.models import load_model
import os

# ---- paths ----
SAVED_MODELS_DIR = "saved_models"
PARAMS_PATH = os.path.join(SAVED_MODELS_DIR, "model_parameters.json")
ARIMA_PATH = os.path.join(SAVED_MODELS_DIR, "arima_model.pkl")
LSTM_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_model.h5")
LSTM_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_scaler.pkl")
ML_PATH = os.path.join(SAVED_MODELS_DIR, "ml_model.pkl")


# ---- cached loaders ----
@st.cache_resource
def _load_params():
    with open(PARAMS_PATH, 'r') as f:
        return json.load(f)

@st.cache_resource
def _load_lstm_model():
    return load_model(LSTM_PATH, compile=False)

@st.cache_resource
def _load_lstm_scaler():
    return joblib.load(LSTM_SCALER_PATH)

@st.cache_resource
def _load_ml_model():
    return joblib.load(ML_PATH)


# ---- live spread ----
@st.cache_data(ttl="15m")
def get_live_spread_data():
    params = _load_params()
    beta = params["beta"]

    ko = yf.download("KO",period="100d",interval="1d",auto_adjust=True,progress=False)
    pep = yf.download("PEP",period="100d",interval="1d",auto_adjust=True,progress=False)

    df = pd.concat([ko["Close"],pep["Close"]],axis=1)
    df.columns=["KO","PEP"]
    df.dropna(inplace=True)

    # LOG SPREAD
    df["Spread"] = np.log(df["KO"]) - beta*np.log(df["PEP"])
    return df["Spread"]


# ---- Z-SCORE ----
def get_zscore_signal():
    params = _load_params()
    mu = params["mu"]
    sigma = params["sigma"]

    live = get_live_spread_data()
    curr = live.iloc[-1]

    z = (curr - mu) / sigma

    if z < -2:
        return "BUY", f"ZScore: {z:.4f}"
    elif z > 2:
        return "SELL", f"ZScore: {z:.4f}"
    else:
        return "HOLD", f"ZScore: {z:.4f}"


# ---- ARIMA ----
def get_arima_signal():
    try:
        model = joblib.load(ARIMA_PATH)
    except:
        return "ERROR", "ARIMA model missing"

    live = get_live_spread_data()
    curr = float(live.iloc[-1])

    pred_raw = model.predict(n_periods=1)
    try:
        pred = float(pred_raw[0])
    except:
        pred = float(pred_raw)

    if pred > curr:
        return "BUY", f"ARIMA: {pred:.4f} > {curr:.4f}"
    else:
        return "SELL", f"ARIMA: {pred:.4f} < {curr:.4f}"


# ---- LSTM ----
def get_lstm_signal():
    model = _load_lstm_model()
    scaler = _load_lstm_scaler()
    live = get_live_spread_data()
    curr = live.iloc[-1]

    N_LOOK = 60
    last = live.iloc[-N_LOOK:].values.reshape(-1,1)
    scaled = scaler.transform(last)
    scaled = scaled.reshape((1,N_LOOK,1))

    pred_scaled = model.predict(scaled,verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]

    if pred > curr:
        return "BUY", f"LSTM: {pred:.4f} > {curr:.4f}"
    else:
        return "SELL", f"LSTM: {pred:.4f} < {curr:.4f}"


# ---- RANDOM FOREST ----
def get_ml_signal():
    model = _load_ml_model()
    live = get_live_spread_data()
    curr = live.iloc[-1]

    arr = live.iloc[-6:-1].values[::-1].reshape(1,-1)
    p = model.predict(arr)[0]

    if p > curr:
        return "BUY", f"RF: {p:.4f} > {curr:.4f}"
    else:
        return "SELL", f"RF: {p:.4f} < {curr:.4f}"
