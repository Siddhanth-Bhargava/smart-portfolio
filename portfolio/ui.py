import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from portfolio.data import fetch_prices
from portfolio.metrics import compute_returns
from portfolio.optimizer import optimal_portfolio
from portfolio.optimize_sgd import sgd_optimize
from datetime import date

st.set_page_config(page_title="Smart Portfolio", layout="wide")

st.title("📈 Smart Portfolio Optimizer")
st.write("Welcome to your portfolio optimization dashboard!")

st.info("This is a placeholder UI. Replace this with your actual app code.")

# Sidebar controls
tickers = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA")
tickers = [t.strip() for t in tickers.split(",")]

start_date, end_date = st.sidebar.date_input(
    "Date range", [date(2021,1,1), date(2022,1,1)]
)

algo = st.sidebar.radio("Optimizer", ["Efficient Frontier", "SGD"])

if st.sidebar.button("Run"):
    prices = fetch_prices(tickers, start_date, end_date)
    if prices is None or prices.empty or prices.shape[0] < 2 or prices.shape[1] < 1 or prices.isnull().values.any():
        st.error("Price data is invalid or insufficient. Please check your tickers and date range.")
    else:
        st.subheader("Price Chart")
        st.line_chart(prices)

        rets = compute_returns(prices)
        st.subheader("Correlation Matrix")
        st.write(rets.corr())

        if algo == "Efficient Frontier":
            st.subheader("Analytical (MPT) Weights")
            weights = optimal_portfolio(prices)
            st.write(weights)
        else:
            lr = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01)
            epochs = st.sidebar.slider("Epochs", 100, 2000, 500, step=100)
            λ = st.sidebar.slider("Risk aversion λ", 0.0, 1.0, 0.1, step=0.05)
            try:
                w_sgd, losses = sgd_optimize(prices, lr=lr, epochs=epochs, λ=λ,
                                             batch_size=int(len(prices)/10), verbose=False)
                st.subheader("SGD-Optimized Weights")
                st.write({t: round(w,4) for t, w in zip(tickers, w_sgd)})

                st.subheader("Loss Convergence")
                st.line_chart(losses)
            except ValueError as e:
                st.error(f"SGD Optimization failed: {e}")
