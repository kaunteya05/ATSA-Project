# pages/2_🚀_Live_Signal.py

import streamlit as st
import sys
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Trading Signals",
    page_icon="🚀"
)

# --- Path Hack for Imports ---
# This tells Python to look in the main project folder
# to find the 'core_logic' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Your Team's Functions ---
# We import the functions from Person 4 (Inference Lead)
try:
    from core_logic.model_inference import (
        get_zscore_signal,
        get_arima_signal,
        get_lstm_signal,
        get_ml_signal
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"Error: {e}. Could not import from 'core_logic.model_inference'. Check file/function names.")
    IMPORTS_SUCCESSFUL = False

# --- Main Page UI ---
st.title("🚀 Live Trading Signals")
st.write("Choose a model to get the latest live signal.")

# --- DELIVERABLE 1: The Radio Button ---
model_choice = st.radio(
    "Select a Trading Model:",
    ("Z-Score (Pairs)", "ARIMA (Time Series)", "LSTM (Deep Learning)", "ML (Classification)"),
    horizontal=True,
    key="model_choice"
)

# --- DELIVERABLE 2 & 3: Call Function & Show Metric (UPDATED) ---
# This code will only run if the imports from Person 4 worked.
if IMPORTS_SUCCESSFUL:
    
    # NEW: Create placeholders for both return values
    signal = "HOLD" 
    details = "Waiting for model..."

    # We add a spinner so the user knows work is being done.
    # The 'get_live_spread_data()' function takes a moment.
    with st.spinner(f"Running {model_choice} model..."):
        try:
            # NEW: We now unpack a (signal, details) tuple from each function
            if model_choice == "Z-Score (Pairs)":
                signal, details = get_zscore_signal()
                
            elif model_choice == "ARIMA (Time Series)":
                signal, details = get_arima_signal()
                
            elif model_choice == "LSTM (Deep Learning)":
                signal, details = get_lstm_signal()
                
            elif model_choice == "ML (Classification)":
                signal, details = get_ml_signal()
        
        except Exception as e:
            # Catch any unexpected runtime errors
            signal = "ERROR"
            details = f"A runtime error occurred in the model: {str(e)}"

    # --- Display the Result (UPDATED) ---
    # We use columns to make it look clean and centered.
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader(f"Signal for: {model_choice}")
        
        # NEW: Handle the "ERROR" signal gracefully
        # Person 4's ARIMA logic might return "ERROR" if the file is missing
        if signal == "ERROR":
            st.error(f"Model Error: {details}")
        
        # --- This is the main display logic ---
        elif signal == "BUY":
            st.metric(label="Current Signal", value="BUY", delta="Positive Forecast", delta_color="normal")
            # NEW: We add the details string below the metric
            st.caption(f"Reason: {details}")
        
        elif signal == "SELL":
            st.metric(label="Current Signal", value="SELL", delta="Negative Forecast", delta_color="inverse")
            # NEW: We add the details string below the metric
            st.caption(f"Reason: {details}")
            
        else: # HOLD
            st.metric(label="Current Signal", value="HOLD", delta="Neutral", delta_color="off")
            # NEW: We add the details string below the metric
            st.caption(f"Reason: {details}")

else:
    st.warning("Model logic could not be imported. Page is disabled.")