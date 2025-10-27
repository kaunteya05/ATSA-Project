import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data
def load_data():
    """
    Downloads 10 years of 'KO' and 'PEP' stock data, cleans it, and caches the result.
    The function returns a DataFrame of cleaned, 10-year daily Close prices.
    """
    # 1. DOWNLOAD (10 years for KO and PEP)
    tickers = ['KO', 'PEP']
    
    # Download 10 years of daily data
    data = yf.download(
        tickers=tickers,
        period='10y',
        interval='1d',
        group_by='ticker',  # Important: Groups columns by ticker (KO, PEP)
        auto_adjust=True,   # Adjusts prices for splits and dividends
        progress=False
    )
    
    # 2. CLEANING AND TRANSFORMING
    
    # yfinance returns a MultiIndex DataFrame (e.g., ('KO', 'Close'), ('KO', 'Volume'), etc.)
    
    # a. Select only the 'Close' prices for both stocks
    try:
        # Create a new DataFrame with just the Close prices
        close_data = pd.DataFrame({
            'KO_Close': data['KO']['Close'],
            'PEP_Close': data['PEP']['Close']
        })
    except KeyError:
        # Handle cases where yfinance might return a flat structure for a single day, 
        # though rare with period='10y'. We'll stick to the desired columns.
        print("Warning: Data structure unexpected, proceeding with available columns.")
        # If the structure is flat, we'll try to select columns directly, 
        # but the MultiIndex is standard for multiple tickers.
        return None 
    
    # b. Drop rows with any missing values (NaNs)
    # This is a crucial cleaning step to ensure consistent time series analysis
    cleaned_data = close_data.dropna(how='any')
    
    # 3. CACHE (Handled by the @st.cache_data decorator)
    
    return cleaned_data

if __name__ == '__main__':
    # This block is for local testing (won't be used by Streamlit)
    print("--- Testing load_data() locally ---")
    try:
        df = load_data()
        if df is not None:
            print(f"Data loaded successfully. Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nMissing values check (should be 0):")
            print(df.isnull().sum())
    except Exception as e:
        print(f"An error occurred during local test: {e}")