import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Configuration
st.set_page_config(page_title="Multi-Asset Sharpe Ratio Simulator", layout="wide")

ASSET_MAP = {
    "Gold": "GC=F",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "S&P500": "^GSPC",
    "US30": "^DJI",
    "US10": "IEF",
    "USD": "CASH_USD",
    "EUR": "EURUSD=X",
    "World_Emerging": "EEM"
}

@st.cache_data(ttl=3600)
def load_data(start_date, end_date):
    tickers = [ticker for ticker in ASSET_MAP.values() if ticker != "CASH_USD"]
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
    data = data.ffill().dropna()
    returns = data.pct_change().dropna()
    returns["CASH_USD"] = 0.0
    return returns

# Header
st.title("📊 Multi-Asset Portfolio & Sharpe Ratio Simulator")
st.markdown("Simulate customized asset allocations, risk metrics, and historical growth trajectories.")

# Sidebar Controls
st.sidebar.header("1. Date Range & Risk-Free Rate")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-01-01"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25) / 100

st.sidebar.header("2. Asset Allocations (%)")
allocations = {}
for asset in ASSET_MAP.keys():
    default_val = 20 if asset in ["S&P500", "Gold"] else (10 if asset in ["BTC", "US10", "World_Emerging", "USD", "US30"] else 0)
    allocations[asset] = st.sidebar.slider(f"{asset}", 0, 100, default_val, 5)

total_alloc = sum(allocations.values())

# Warning for Allocation Sum
if total_alloc != 100:
    st.sidebar.warning(f"Total Allocation: **{total_alloc}%**. Weights will be auto-normalized to 100%.")
else:
    st.sidebar.success("Total Allocation: **100%**")

# Data Fetching & Processing
returns_df = load_data(start_date, end_date)

weights = {asset: w / (total_alloc if total_alloc > 0 else 1) for asset, w in allocations.items()}
weight_vector = np.array([weights[asset] for asset in ASSET_MAP.keys()])
selected_tickers = [ASSET_MAP[asset] for asset in ASSET_MAP.keys()]

portfolio_returns = returns_df[selected_tickers].dot(weight_vector)

# Calculations (252 Trading Days)
ann_return = portfolio_returns.mean() * 252
ann_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0

cumulative_growth = (1 + portfolio_returns).cumprod() * 10000
max_drawdown = ((cumulative_growth.cummax() - cumulative_growth) / cumulative_growth.cummax()).max()

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Annualized Return", f"{ann_return:.2%}")
col2.metric("Annualized Volatility", f"{ann_volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"-{max_drawdown:.2%}")
col5.metric("Ending Value ($10k)", f"${cumulative_growth.iloc[-1]:,.2f}")

st.markdown("---")

# Chart Layout
col_chart, col_weights = st.columns([3, 1])

with col_chart:
    st.subheader("Historical Portfolio Growth ($10,000 Initial)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_growth.index, y=cumulative_growth, mode='lines', name='Simulated Portfolio', line=dict(color='#00D1B2', width=2)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)", margin=dict(l=20, r=20, t=20, b=20), height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_weights:
    st.subheader("Asset Weight Mix")
    active_weights = {k: v for k, v in weights.items() if v > 0}
    fig_pie = go.Figure(data=[go.Pie(labels=list(active_weights.keys()), values=list(active_weights.values()), hole=.4)])
    fig_pie.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350, showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)
