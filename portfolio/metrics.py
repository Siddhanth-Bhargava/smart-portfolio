import numpy as np
import pandas as pd

def compute_returns(price_df):
    """Percent‐change returns, drop the NaN row."""
    return price_df.pct_change().dropna()

def portfolio_stats(weights, returns_df):
    """
    Given weight vector and returns DataFrame:
      - annualized return
      - annualized volatility
      - Sharpe ratio (assume rf=0)
    """
    mu = returns_df.mean() * 252       # expected annual return per asset
    Sigma = returns_df.cov() * 252     # annual covariance matrix
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(weights @ Sigma.values @ weights)
    sharpe = port_return / port_vol
    return port_return, port_vol, sharpe

prices = pd.DataFrame({"A":[100,110,105], "B":[200,210,220]})
rets = compute_returns(prices)
portfolio_stats([0.5,0.5], rets)
