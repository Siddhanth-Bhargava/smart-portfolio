from pypfopt import EfficientFrontier, risk_models, expected_returns

def optimal_portfolio(price_df):
    """
    Uses PyPortfolioOpt to maximize Sharpe ratio.
    Returns cleaned weights dict.
    """
    mu = expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    return ef.clean_weights()