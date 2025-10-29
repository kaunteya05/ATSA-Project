"""
This is the main, one-off training script for Person 3.
It imports the data-loading and analysis functions,
trains all three models (ARIMA, LSTM, ML), and saves them to the
'saved_models/' directory.

Run this file once from your terminal:
python model_training.py
"""

import os
import pandas as pd
import numpy as np
import joblib  # For saving .pkl models (ARIMA, ML, Scaler)
import json
from pmdarima import auto_arima # For ARIMA
from sklearn.ensemble import RandomForestRegressor # For "ML Model"
from sklearn.preprocessing import MinMaxScaler # For LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# --- Import from Person 1 and 2 ---
# This assumes Person 2's function `get_spread_parameters`
# 1. Imports and uses Person 1's `load_data()` internally.
# 2. Saves the 'model_parameters.json' file.
# 3. CRITICALLY: Returns the 10-year spread as a pandas.Series.
try:
    # This line is edited to match your project's file name
    from Cointegration_Test import get_spread_parameters
except ImportError as e:
    print(f"FATAL ERROR: Make sure 'cointegration_test.py' exists.")
    print(f"Error details: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- Define File Paths ---
SAVED_MODELS_DIR = "saved_models"
ARIMA_PATH = os.path.join(SAVED_MODELS_DIR, "arima_model.pkl")
LSTM_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_model.h5")
LSTM_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_scaler.pkl") # CRITICAL: Needed for new predictions
ML_PATH = os.path.join(SAVED_MODELS_DIR, "ml_model.pkl")

# --- Helper Functions for Model Preparation ---

def prepare_ml_data(series, n_lags=5):
    """
    Converts a time series into a supervised learning problem
    by using lagged values as features.
    
    X = [lag_1, lag_2, ..., lag_n]
    y = [value_at_t]
    """
    df = pd.DataFrame(series)
    df.columns = ['y']
    
    # Create lagged features
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
        
    df.dropna(inplace=True) # Drop rows with NaN values (at the beginning)
    
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y

def prepare_lstm_data(series, n_input=60):
    """
    Scales and sequences data for an LSTM model.
    """
    # Scale data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    # Create a time series generator
    # It will use `n_input` past days to predict the 1 day ahead
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
    
    return generator, scaler, n_input

# --- Main Training Execution ---

def main():
    """
    Main function to run the entire training pipeline.
    """
    print("--- Person 3: Model Training Script START ---")
    
    # Ensure the 'saved_models' directory exists
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    print(f"Directory '{SAVED_MODELS_DIR}' ensured.")

    # --- Step 1: Get Data from Person 2 ---
    print("Importing data pipeline from Person 1 & 2...")
    try:
        # This function should do P2's job (save JSON) and return the spread
        spread_series = get_spread_parameters()
    except Exception as e:
        print(f"\n--- FATAL ERROR during execution of get_spread_parameters ---")
        print(f"Error details: {e}")
        print("-------------------\n")
        return

    if not isinstance(spread_series, pd.Series) or spread_series.empty:
        print("ERROR: 'get_spread_parameters' did not return a valid pandas Series. Exiting.")
        return
        
    print(f"Successfully loaded 10-year spread data. (Length: {len(spread_series)})")

    # --- Step 2: Train ARIMA Model ---
    print("\nTraining ARIMA model...")
    # auto_arima finds the best (p,d,q) order automatically
    arima_model = auto_arima(spread_series, 
                             seasonal=False,  # We are modeling a (supposedly) stationary spread
                             m=1,             # No seasonality
                             stepwise=True,
                             suppress_warnings=True,
                             trace=False)
    
    print(f"Best ARIMA order found: {arima_model.order}")
    joblib.dump(arima_model, ARIMA_PATH)
    print(f"✅ ARIMA model saved to: {ARIMA_PATH}")

    # --- Step 3: Train LSTM Model ---
    print("\nTraining LSTM model...")
    n_lookback = 60 # Use 60 days of history to predict the next day
    
    lstm_generator, lstm_scaler, n_input = prepare_lstm_data(spread_series, n_input=n_lookback)
    
    # Define the model architecture
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    # Using verbose=0 to keep the terminal clean
    lstm_model.fit(lstm_generator, epochs=50, verbose=0)
    
    # Save the model (.h5) and the scaler (.pkl)
    lstm_model.save(LSTM_PATH)
    joblib.dump(lstm_scaler, LSTM_SCALER_PATH)
    print(f"✅ LSTM model saved to: {LSTM_PATH}")
    print(f"✅ LSTM scaler saved to: {LSTM_SCALER_PATH} (CRITICAL for predictions!)")

    # --- Step 4: Train "ML Model" (Random Forest Regressor) ---
    print("\nTraining ML (Random Forest) model...")
    # We will use 5 lagged days as features to predict the next day
    X_ml, y_ml = prepare_ml_data(spread_series, n_lags=5)
    
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    ml_model.fit(X_ml, y_ml)
    
    joblib.dump(ml_model, ML_PATH)
    print(f"✅ ML model saved to: {ML_PATH}")

    # --- MILESTONE COMPLETE ---
    print("\n-------------------------------------------------")
    print("🎉 MILESTONE COMPLETE! 🎉")
    print("All models trained and saved to 'saved_models/'.")
    print("The 'Quants' team is done. Tell everyone!")
    print("-------------------------------------------------")


if __name__ == "__main__":
    # This block ensures the code only runs when you execute
    # the script directly from the terminal.
    main()