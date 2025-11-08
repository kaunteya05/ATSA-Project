import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# 1. Import Logic
# This follows your instruction: "import core_logic.analysis as analysis"
# It assumes you have a folder structure like:
# ├── pages/
# │   └── 3_📈_Historical_Analysis.py
# ├── core_logic/
# │   ├── __init__.py
# │   └── analysis.py  <-- This is the file you provided
# └── data_loader.py
try:
    import core_logic.analysis as analysis
except ImportError:
    st.error(
        "**ImportError:** Could not find the `core_logic.analysis` module. "
        "Please make sure your `analysis.py` file is saved inside a folder "
        "named `core_logic` and that you have an empty `__init__.py` file in that folder."
    )
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Historical Analysis", layout="wide")
st.title("📈 Historical Analysis")
st.markdown("This page analyzes the historical spread of the cointegrated pair and identifies anomalies.")


# --- Data Loading ---
@st.cache_data
def load_spread_data():
    """
    Runs Person 2's analysis function to get the historical spread.
    We use st.cache_data so this expensive analysis (which runs
    tests and loads data) only runs ONCE, not every time the
    slider is changed.
    """
    st.text("Running cointegration analysis (cached)...")
    spread_series = analysis.get_spread_parameters()
    return spread_series

# Load the data
spread_series = load_spread_data()

if spread_series is None:
    st.error("Failed to load spread data. Check the console for logs from 'analysis.py'.")
    st.stop()


# --- Z-Score Calculation & DataFrame ---
# Person 2's script gives us the spread. This page must calculate the Z-Score.
# Z-Score = (Value - Mean) / StdDev
mu = spread_series.mean()
sigma = spread_series.std()

# Create a DataFrame for analysis and plotting
data = pd.DataFrame({
    'Spread': spread_series,
    'Z_Score': (spread_series - mu) / sigma
})
data.index.name = 'Date'
data.reset_index(inplace=True)


# --- 1. Add the st.slider for the Z-Score ---
st.header("Anomaly Threshold Control")
z_score_threshold = st.slider(
    "Select Z-Score Threshold",
    min_value=1.0,
    max_value=4.0,
    value=2.5,  # A common default
    step=0.1,
    help="Any spread value with a Z-Score (standard deviations from the mean) "
         "outside this $\\pm$ range will be flagged as an anomaly."
)
st.markdown(f"**Current Threshold:** $\\pm {z_score_threshold} \\sigma$")


# --- 2. Build the Plotly chart and st.dataframe for the anomalies ---

# Add the 'Anomaly' flag based on the slider's value
data['Anomaly'] = np.abs(data['Z_Score']) > z_score_threshold
anomaly_data = data[data['Anomaly']]

st.header("Historical Spread with Anomalies")

# Create the Plotly chart
fig = px.line(
    data,
    x='Date',
    y='Spread',
    title='Historical Spread (Spread = KO - $\\beta$ * PEP)'
)

# Add horizontal lines for the mean and the Z-Score thresholds
fig.add_hline(y=mu, line_dash="dash", line_color="gray", annotation_text="Mean (Mu)")
fig.add_hline(
    y=mu + (z_score_threshold * sigma),
    line_dash="dot",
    line_color="red",
    annotation_text=f"+{z_score_threshold} $\\sigma$"
)
fig.add_hline(
    y=mu - (z_score_threshold * sigma),
    line_dash="dot",
    line_color="red",
    annotation_text=f"-{z_score_threshold} $\\sigma$"
)

# Add red markers for the detected anomalies
if not anomaly_data.empty:
    fig.add_scatter(
        x=anomaly_data['Date'],
        y=anomaly_data['Spread'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='Red', size=8, symbol='circle-open')
    )

st.plotly_chart(fig, use_container_width=True)

# Build the st.dataframe for the anomalies
st.subheader("Detected Anomalies")
if anomaly_data.empty:
    st.info("No anomalies detected for the current Z-Score threshold.")
else:
    st.dataframe(
        anomaly_data[['Date', 'Spread', 'Z_Score']].style.format({'Z_Score': '{:.2f}'}),
        use_container_width=True
    )