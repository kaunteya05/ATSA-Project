import streamlit as st

st.set_page_config(page_title="ATSA Project", layout="wide")

st.title("Applied Time Series Analysis Project")

import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

st.title("ATSA Pairs Trading Project - KO vs PEP")

# Load 10 year KO / PEP close
start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')
df = yf.download(["KO","PEP"],
                 start=start_date,
                 end=end_date,
                 interval="1d",
                 auto_adjust=True,
                 progress=False)

price_df = pd.DataFrame({
    "KO": df["Close"]["KO"],
    "PEP": df["Close"]["PEP"]
}).dropna()

st.subheader("10-Year Price Comparison")
st.line_chart(price_df)

st.markdown("""
### Project Topic: Cointegrated Pairs + Forecast Driven Trading

We analyse 2 stocks from same sector which historically move together  
(e.g. Coca-Cola (KO) vs Pepsi (PEP)).

We model the *spread* between them and generate trading signals using:

- Z-Score Mean Reversion (Pairs Trading)
- ARIMA Time Series Model
- LSTM Deep Learning Forecasting
- Random Forest ML Forecasting

You can view:

| Page | Purpose |
|------|---------|
| 🚀 Live Signal | See today's trading signal from each model |
| 📈 Backtest Comparison | Compare historical profitability of models |
| 🧪 Historical Analysis | Visualize spread, anomalies, mean reversion behavior |
""")

st.info("Use the left sidebar to navigate between pages.")
