# pages/2_🚀_Live_Signal.py

import streamlit as st
import sys, os

# Ensure we can import from core_logic when Streamlit runs from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import live signal functions
try:
    from core_logic.model_inference import (
        get_zscore_signal,
        get_arima_signal,
        get_lstm_signal,
        get_ml_signal,
    )
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERR = str(e)

# ---------------- UI ----------------
st.set_page_config(page_title="Live Trading Signals", page_icon="🚀", layout="wide")
st.title("🚀 Live Trading Signals — KO vs PEP (Spread)")

st.write(
    "Signals are generated on the **log-spread**:  \n"
    r"$\text{spread} = \log(\mathrm{KO}) - \beta \cdot \log(\mathrm{PEP})$"
)

model_choice = st.radio(
    "Select a Trading Model:",
    ("Z-Score (Pairs)", "ARIMA (Time Series)", "LSTM (Deep Learning)", "ML (Classification)"),
    horizontal=True,
    key="model_choice_live",
)

if not IMPORTS_OK:
    st.error(f"Could not import live model functions: {IMPORT_ERR}")
    st.stop()

# Map of callable per model
MODEL_FN = {
    "Z-Score (Pairs)": get_zscore_signal,
    "ARIMA (Time Series)": get_arima_signal,
    "LSTM (Deep Learning)": get_lstm_signal,
    "ML (Classification)": get_ml_signal,
}

# Helper: convert spread signal -> explicit pair actions
def explain_actions(signal: str):
    """
    For spread = log(KO) - beta*log(PEP):
    BUY  -> expect spread ↑  -> Long KO / Short PEP
    SELL -> expect spread ↓  -> Short KO / Long PEP
    HOLD -> no position
    """
    if signal == "BUY":
        return "BUY (Long KO / Short PEP)"
    if signal == "SELL":
        return "SELL (Short KO / Long PEP)"
    return "HOLD (Square off / No position)"

# ---------------- Run & Display ----------------
st.subheader(f"Signal for: {model_choice}")

with st.spinner(f"Running {model_choice}..."):
    try:
        signal, details = MODEL_FN[model_choice]()  # each returns (signal, details)
    except Exception as e:
        signal, details = "ERROR", f"A runtime error occurred in the model: {e}"

# Normalize details into string
details = "" if details is None else str(details)

if signal == "ERROR":
    st.error(f"Model Error: {details}")
else:
    pretty_value = explain_actions(signal)

    if signal == "BUY":
        st.metric(label="Current Signal", value=pretty_value, delta="Positive Forecast", delta_color="normal")
        st.caption(f"Reason: {details}")
    elif signal == "SELL":
        st.metric(label="Current Signal", value=pretty_value, delta="Negative Forecast", delta_color="inverse")
        st.caption(f"Reason: {details}")
    else:
        st.metric(label="Current Signal", value=pretty_value, delta="Neutral", delta_color="off")
        st.caption(f"Reason: {details}")

# Helpful note for viva/exam
st.info(
    "Interpretation: A **BUY** signal means the spread is expected to **increase** "
    "(go **long KO**, **short PEP**). A **SELL** signal means the spread is expected to **decrease** "
    "(go **short KO**, **long PEP**). **HOLD** means no trade / square off."
)
