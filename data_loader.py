#import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# CRITICAL CHANGE: ttl=86400 forces a data refresh after 24 hours (86400 seconds).
# The first user to access the app after 24 hours have passed will trigger the update.
#@st.cache_data(ttl=86400) 
def load_data():
    """
    Downloads 'KO' and 'PEP' stock data from Jan 1, 2015, to today's date.
    The data is cached for 24 hours (86400 seconds) to ensure a daily update.
    """
    # 1. DOWNLOAD (Specific date range for KO and PEP)
    tickers = ['KO', 'PEP']
    
    # Define the start date (1st January 2015)
    start_date = '2015-01-01'
    
    # Define the end date as today
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Note: For financial data, yfinance may return up to the last *market* close, not the current minute.
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )
    
    # 2. CLEANING AND TRANSFORMING
    
    # Select only the 'Close' prices for both stocks
    try:
        close_data = pd.DataFrame({
            'KO_Close': data['KO']['Close'],
            'PEP_Close': data['PEP']['Close']
        })
    except KeyError:
        # In case the expected MultiIndex structure isn't returned
        print("Warning: Data structure unexpected.")
        return None 
    
    # Drop rows with any missing values (NaNs)
    cleaned_data = close_data.dropna(how='any')
    
    return cleaned_data

if __name__ == '__main__':
    # Local testing block
    print("--- Testing load_data() locally ---")
    df = load_data()
    if df is not None:
        print(f"Data loaded successfully. Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(df)