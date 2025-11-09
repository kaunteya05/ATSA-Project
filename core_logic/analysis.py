import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
import json

# Import KO/PEP data loader
try:
    from data_loader import load_data
except ImportError as e:
    print("FATAL: Could not import from 'data_loader.py'.")
    print("Make sure the file is saved and all libraries are installed.")
    print(f"Error details: {e}")
    raise

# ---------- Helpers ----------
def check_stationarity(series, name):
    """ADF test with safe dropna()."""
    result = adfuller(series.dropna())
    print(f"\n{name} - Augmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value:.3f}")
    print("=> Stationary" if result[1] < 0.05 else "=> Not stationary")

def test_cointegration(df, col1, col2):
    score, pvalue, _ = coint(df[col1], df[col2])
    print("\nEngle-Granger Cointegration Test:")
    print(f"t-statistic: {score}")
    print(f"p-value: {pvalue}")
    if pvalue < 0.05:
        print("=> The series are cointegrated")
        return True
    else:
        print("=> The series are not cointegrated")
        return False

def calculate_parameters_and_spread(df, col1, col2):
    """
    Estimate beta on LOG prices and compute LOG-spread:
        spread = log(col1) - beta * log(col2)
    Returns (spread_series, beta, mu, sigma).
    """
    log_y = np.log(df[col1])  # e.g., KO
    log_x = np.log(df[col2])  # e.g., PEP

    X = sm.add_constant(log_x)
    model = sm.OLS(log_y, X).fit()
    beta = float(model.params[log_x.name])

    spread = log_y - beta * log_x
    mu = float(spread.mean())
    sigma = float(spread.std())

    print("\nOLS (on log prices) Results:")
    print(f"Beta (Hedge Ratio): {beta}")
    print(f"Spread Mean (mu):  {mu}")
    print(f"Spread Std Dev (σ): {sigma}")

    return spread, beta, mu, sigma

# ---------- Main API ----------
def get_spread_parameters():
    """
    1) Loads KO/PEP close
    2) Runs tests
    3) Computes LOG-spread params
    4) Saves {'beta','mu','sigma'} to saved_models/model_parameters.json
    5) Returns spread (pd.Series)
    """
    print("--- Analysis: Cointegration and Parameter Estimation ---")
    COL1 = "KO_Close"
    COL2 = "PEP_Close"

    df = load_data()
    if df is None or df.empty:
        print("Failed to load data. Exiting.")
        return None

    df = df.dropna()

    # Optional: tests on levels (or logs – cointegration is scale-invariant)
    check_stationarity(df[COL1], COL1)
    check_stationarity(df[COL2], COL2)
    test_cointegration(df, COL1, COL2)

    spread_series, beta, mu, sigma = calculate_parameters_and_spread(df, COL1, COL2)

    # Save parameters JSON
    SAVED_MODELS_DIR = "saved_models"
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    PARAMS_PATH = os.path.join(SAVED_MODELS_DIR, "model_parameters.json")

    params = {"beta": beta, "mu": mu, "sigma": sigma}
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=4)

    print(f"\n✅ Parameters saved to: {PARAMS_PATH}")
    print("--- Analysis Complete ---")
    return spread_series

if __name__ == "__main__":
    print("Testing analysis.py...")
    s = get_spread_parameters()
    if s is not None:
        print("\nSpread preview:")
        print(s.head())
