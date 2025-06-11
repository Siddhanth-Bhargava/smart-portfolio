import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
st.set_page_config(page_title="Smart Portfolio", layout="wide")
from portfolio.data import fetch_prices
from portfolio.metrics import compute_returns
from portfolio.optimizer import optimal_portfolio
from portfolio.optimize_sgd import sgd_optimize
from datetime import date

# Welcome message
st.title("📈 Smart Portfolio Optimizer")
st.markdown("""
Welcome! This app helps you build a custom investment portfolio from top US companies.
Just pick your stocks, choose a time period, and let us do the math!
""")

st.sidebar.header("Build Your Portfolio")
st.sidebar.markdown("**Select the companies you want to invest in:**")

@st.cache_data
def get_sp500_tickers():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

sp500_tickers = get_sp500_tickers()
selected = st.sidebar.multiselect(
    "Stocks (S&P 500)", sp500_tickers, default=["AAPL", "MSFT", "TSLA"], key="sp500_multiselect",
    help="Pick one or more companies to include in your portfolio."
)
tickers = selected

st.sidebar.markdown("**Choose the time period for your investment analysis:**")
start_date, end_date = st.sidebar.date_input(
    "Date range", [date(2021,1,1), date(2022,1,1)],
    help="Pick the start and end dates for the historical data."
)

st.sidebar.markdown("**Select your investment strategy:**")
algo = st.sidebar.radio(
    "Strategy",
    ["Balanced (Efficient Frontier)", "Growth (SGD)"],
    help="Balanced is for risk/reward, Growth is for higher potential returns."
)

if st.sidebar.button("Run"):
    prices = fetch_prices(tickers, start_date, end_date)
    st.write(f"Price data shape: {prices.shape if prices is not None else 'None'}")
    if prices is not None and not prices.empty:
        st.write("Preview of price data:")
        st.write(prices.head())
    if prices is None or prices.empty or prices.shape[0] < 2 or prices.shape[1] < 1 or prices.isnull().values.any():
        st.error("Price data is invalid or insufficient. Please check your tickers and date range.")
    else:
        st.subheader("Price Chart")
        st.line_chart(prices)
        rets = compute_returns(prices)
        st.subheader("Correlation Matrix")
        st.write(rets.corr())
        if algo == "Balanced (Efficient Frontier)":
            st.subheader("Recommended Portfolio Weights")
            st.markdown("These weights show how much of your money would go into each stock for the best balance of risk and reward.")
            weights = optimal_portfolio(prices)
            st.write(weights)
        else:
            lr = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, help="Controls how fast the optimizer learns.")
            epochs = st.sidebar.slider("Epochs", 100, 2000, 500, step=100, help="Number of training cycles.")
            λ = st.sidebar.slider("Risk aversion λ", 0.0, 1.0, 0.1, step=0.05, help="Higher λ means less risk.")
            try:
                w_sgd, losses = sgd_optimize(prices, lr=lr, epochs=epochs, λ=λ,
                                             batch_size=int(len(prices)/10), verbose=False)
                st.subheader("SGD-Optimized Weights")
                st.markdown("These weights are calculated using a growth-focused strategy.")
                st.write({t: round(w,4) for t, w in zip(tickers, w_sgd)})
                st.subheader("How the Optimizer Improved Your Portfolio")
                st.line_chart(losses)
            except ValueError as e:
                st.error(f"SGD Optimization failed: {e}")

# Add a FAQ or explainer section
with st.expander("What do these results mean?"):
    st.markdown("""
    - **Price Chart:** Shows how your selected stocks performed over time.
    - **Correlation Matrix:** Tells you how similar the stocks' movements are.
    - **Portfolio Weights:** How much of your money would go into each stock for the best balance of risk and reward.
    - **SGD Loss Convergence:** Shows how the optimizer improved your portfolio over time.
    """)
