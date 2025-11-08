import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import sqrt
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import json, os

from core_logic.analysis import get_spread_parameters

ARIMA_PATH  = "saved_models/arima_model.pkl"
LSTM_PATH   = "saved_models/lstm_model.h5"
SCALER_PATH = "saved_models/lstm_scaler.pkl"
ML_PATH     = "saved_models/ml_model.pkl"

st.set_page_config(page_title="Backtest Comparison", layout="wide")
st.title("📊 Backtest Comparison — ARIMA vs LSTM vs Random Forest (on log-spread)")

st.markdown("""
Walk-forward 1-step-ahead predictions on historical **log-spread** (KO vs PEP).
""")

# SIDEBAR SETTINGS
with st.sidebar:
    st.header("⚙️ Backtest Settings")
    lookback_days = st.number_input("Window (days from end)", 60, 2000, 200, 10)   # FASTER now
    tc_bps = st.number_input("Transaction cost (bps per position change)", 0, 50, 2, 1)
    run_btn = st.button("Run Backtests")

tc_float = tc_bps / 10000.0

# UTILS
def sharpe(returns):
    if len(returns)<2: return 0.0
    mu, sd = returns.mean(), returns.std() + 1e-12
    return float((mu/sd)*sqrt(252))

def max_dd(cum):
    roll = cum.cummax()
    return float((cum-roll).min())

def walk_forward(spread,preds,tc):
    idx = spread.index.intersection(preds.index)
    spread = spread.loc[idx]; preds = preds.loc[idx]
    ret_next = spread.diff().shift(-1)
    sig = np.sign(preds-spread).replace(0,method="ffill")
    trades=(sig!=sig.shift(1)).astype(int); trades.iloc[0]=0
    step=(sig*ret_next - trades*tc).iloc[:-1].fillna(0.0)
    cum=step.cumsum()
    hit=(np.sign(ret_next.loc[step.index])==sig.loc[step.index]).mean()
    return step,cum,{"Total P&L":float(cum.iloc[-1]),"Trades":int(trades.sum()),"Hit Rate":float(hit),"Sharpe":sharpe(step),"Max Drawdown":max_dd(cum)}

def chart(cum):
    fig=go.Figure()
    for k,v in cum.items(): fig.add_trace(go.Scatter(x=v.index,y=v.values,mode="lines",name=k))
    fig.update_layout(title="Cumulative P&L (walk-forward, 1-step ahead)",xaxis_title="Date",yaxis_title="P&L")
    return fig

@st.cache_data
def load_spread(): return get_spread_parameters()

spread_all=load_spread()
if spread_all is None or spread_all.empty: st.error("Could not compute spread."); st.stop()
if lookback_days > len(spread_all): lookback_days=len(spread_all)
spread=spread_all.iloc[-lookback_days:].copy()

@st.cache_resource
def load_models():
    d={}
    try:d["arima"]=joblib.load(ARIMA_PATH)
    except Exception as e:d["arima_err"]=str(e)
    try:
        d["lstm"]=load_model(LSTM_PATH,compile=False)
        d["scaler"]=joblib.load(SCALER_PATH)
    except Exception as e:d["lstm_err"]=str(e)
    try:d["rf"]=joblib.load(ML_PATH)
    except Exception as e:d["rf_err"]=str(e)
    return d

models=load_models()

if run_btn:
    results=[]; cum_curves={}

    # ---- ARIMA (NO UPDATE = FAST FIX) ----
    if "arima" in models:
        arima=models["arima"]
        preds=[]
        idx=spread.index
        for t in range(len(spread)-1):
            try:p=float(arima.predict(n_periods=1)[0])
            except:p=float(spread.iloc[t])
            preds.append((idx[t],p))
    # light update every 10 steps
            if t % 10 == 0:  
                try: arima.update(spread.iloc[[t]])
                except: pass

        preds=pd.Series([p for _,p in preds], index=[i for i,_ in preds])
        step,cum,summ=walk_forward(spread,preds,tc_float)
        cum_curves["ARIMA"]=cum; summ["Model"]="ARIMA"; results.append(summ)

    # ---- LSTM (REDUCED LOOKBACK = FASTER) ----
    if "lstm" in models and "scaler" in models:
        lstm,scaler=models["lstm"],models["scaler"]
        look=30
        s_full=spread_all.iloc[-(lookback_days+look+1):]
        preds=[]
        for t in range(look,len(s_full)-1):
            win=s_full.iloc[t-look:t].values.reshape(-1,1)
            win_s=scaler.transform(win)
            x=win_s.reshape(1,look,1)
            p_s=lstm.predict(x,verbose=0)[0][0]
            p=float(scaler.inverse_transform([[p_s]])[0][0])
            preds.append((s_full.index[t],p))
        preds=pd.Series([p for _,p in preds], index=[i for i,_ in preds])
        preds=preds.loc[preds.index.intersection(spread.index)]
        step,cum,summ=walk_forward(spread,preds,tc_float)
        cum_curves["LSTM"]=cum; summ["Model"]="LSTM"; results.append(summ)

    # ---- RF ----
    if "rf" in models:
        rf=models["rf"]; n=5
        preds=[]; s=spread_all
        start=len(s)-lookback_days
        for i in range(start,len(s)-1):
            if i<n: continue
            lags=s.iloc[i-n:i].values[::-1].reshape(1,-1)
            try:p=float(rf.predict(lags)[0])
            except:p=float(s.iloc[i])
            preds.append((s.index[i],p))
        preds=pd.Series([p for _,p in preds], index=[i for i,_ in preds])
        preds=preds.loc[preds.index.intersection(spread.index)]
        step,cum,summ=walk_forward(spread,preds,tc_float)
        cum_curves["Random Forest"]=cum; summ["Model"]="Random Forest"; results.append(summ)

    st.subheader("Cumulative P&L")
    st.plotly_chart(chart(cum_curves), use_container_width=True)

    st.subheader("Performance Summary")
    if len(results)>0:
        res=pd.DataFrame(results)[["Model","Total P&L","Trades","Hit Rate","Sharpe","Max Drawdown"]]
        st.dataframe(res.style.format({"Total P&L":"{:.4f}","Hit Rate":"{:.1%}","Sharpe":"{:.2f}","Max Drawdown":"{:.4f}"}))
    else:
        st.warning("No performance results computed.")
else:
    st.info("Set parameters on left + click Run Backtests.")
