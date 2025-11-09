import os
import pandas as pd
import numpy as np
import joblib
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from core_logic.analysis import get_spread_parameters

SAVED_MODELS_DIR = "saved_models"
ARIMA_PATH = os.path.join(SAVED_MODELS_DIR, "arima_model.pkl")
LSTM_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_model.h5")
LSTM_SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "lstm_scaler.pkl")
ML_PATH = os.path.join(SAVED_MODELS_DIR, "ml_model.pkl")

def prepare_ml_data(series, n_lags=5):
    df = pd.DataFrame(series)
    df.columns = ['y']
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    return df.drop('y', axis=1), df['y']

def prepare_lstm_data(series, n_input=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)
    return generator, scaler, n_input

def main():
    print("--- Model Training START ---")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    spread_series = get_spread_parameters()
    if not isinstance(spread_series, pd.Series) or spread_series.empty:
        print("ERROR: Invalid spread series.")
        return
    
    print(f"Spread length: {len(spread_series)}")

    print("\nTraining ARIMA...")
    arima_model = auto_arima(spread_series, seasonal=False, m=1, stepwise=True, suppress_warnings=True, trace=False)
    joblib.dump(arima_model, ARIMA_PATH)
    print("✅ ARIMA saved")
    
    print("\nTraining RandomForest...")
    X_ml, y_ml = prepare_ml_data(spread_series, n_lags=5)
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    ml_model.fit(X_ml, y_ml)
    joblib.dump(ml_model, ML_PATH)
    print("✅ ML saved")

    print("\nTraining LSTM...")
    n_lookback = 60
    lstm_generator, lstm_scaler, n_input = prepare_lstm_data(spread_series, n_input=n_lookback)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(n_input, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(lstm_generator, epochs=10, verbose=0)       # reduced from 50
    
    lstm_model.save(LSTM_PATH)
    joblib.dump(lstm_scaler, LSTM_SCALER_PATH)
    print("✅ LSTM saved")

    

    print("\n---- TRAINING COMPLETE ----")

if __name__ == "__main__":
    main()
