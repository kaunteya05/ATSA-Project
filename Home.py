import streamlit as st

st.set_page_config(page_title="ATSA Project", layout="wide")

st.title("Applied Time Series Analysis Project")

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
