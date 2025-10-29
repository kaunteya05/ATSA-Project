import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import json
import os

# Import from Person 1's file
try:
    from data_loader import load_data
except ImportError as e:
    print(f"FATAL: Could not import from 'data_loader.py'.")
    print(f"Make sure the file is saved and all libraries are installed.")
    print(f"Error details: {e}")
    exit()

# --- Your Functions (with minor fixes) ---

# Step 1: Stationarity tests (ADF Test for both series)
def check_stationarity(series, name):
    # Added .dropna() to prevent test from crashing
    result = adfuller(series.dropna())
    print(f"\n{name} - Augmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))
    if result[1] < 0.05:
        print("=> Stationary")
    else:
        print("=> Not stationary")

# Step 2: Cointegration test
def test_cointegration(df, col1, col2):
    score, pvalue, _ = coint(df[col1], df[col2])
    print(f"\nEngle-Granger Cointegration Test:")
    print(f"t-statistic: {score}")
    print(f"p-value: {pvalue}")
    if pvalue < 0.05:
        print("=> The series are cointegrated")
        return True
    else:
        print("=> The series are not cointegrated")
        return False

# Step 3: Spread calculation (OLS regression and residuals)
# --- MODIFIED to get all parameters as required by project plan ---
def calculate_parameters_and_spread(df, col1, col2):
    X = df[col2] # e.g., PEP
    y = df[col1] # e.g., KO
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    
    model = sm.OLS(y, X).fit()
    
    # Get Beta (the hedge ratio)
    beta = model.params[col2]
    
    # Calculate the spread
    spread = y - beta * df[col2]
    
    # Get Mu (mean of the spread)
    mu = spread.mean()
    
    # Get Sigma (std dev of the spread)
    sigma = spread.std()

    print("\nOLS Regression Results:")
    print(f"Beta (Hedge Ratio): {beta}")
    print(f"Spread Mean (mu): {mu}")
    print(f"Spread Std Dev (sigma): {sigma}")
    
    return spread, beta, mu, sigma

# --- Main function for Person 3 to import ---
def get_spread_parameters():
    """
    Main function for Person 2.
    1. Loads data.
    2. Runs tests.
    3. Calculates parameters and saves them.
    4. Returns the spread series for Person 3.
    """
    print("--- Person 2: Running Cointegration and Parameter Analysis ---")
    TICKER_1 = 'KO_Close'
    TICKER_2 = 'PEP_Close'
    
    # 1. Load data from data.py
    df = load_data()
    if df is None or df.empty:
        print("Failed to load data. Exiting.")
        return None
        
    # Drop any rows with missing data to align series
    df.dropna(inplace=True)

    # 2. Run Stationarity tests
    check_stationarity(df[TICKER_1], TICKER_1)
    check_stationarity(df[TICKER_2], TICKER_2)

    # 3. Run Cointegration test
    test_cointegration(df, TICKER_1, TICKER_2)

    # 4. Calculate spread and parameters
    spread_series, beta, mu, sigma = calculate_parameters_and_spread(df, TICKER_1, TICKER_2)

    # 5. Save parameters to JSON
    SAVED_MODELS_DIR = "saved_models"
    PARAMS_PATH = os.path.join(SAVED_MODELS_DIR, "model_parameters.json")
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    params = {'beta': beta, 'mu': mu, 'sigma': sigma}
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"\n✅ Parameters saved to: {PARAMS_PATH}")
    print("--- Person 2: Task Complete ---")
    
    # 6. Return the spread series for Person 3
    return spread_series

# This part lets Person 2 run this file directly to test it
if __name__ == "__main__":
    print("Testing cointegration_test.py file...")
    spread = get_spread_parameters()
    if spread is not None:
        print("\nSuccessfully got spread series. First 5 values:")
        print(spread.head())