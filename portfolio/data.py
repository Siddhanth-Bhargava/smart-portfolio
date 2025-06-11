import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_prices(tickers, start_date, end_date, batch_size=50):
    """
    Download adjusted close prices for given tickers & date range.
    Handles batching and missing tickers gracefully.
    Returns a DataFrame indexed by date, columns are tickers.
    """
    all_prices = []
    valid_tickers = []
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_date, progress=False)
            # Handle both single and multi-ticker cases
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif ('Adj Close', batch[0]) in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            # If only one ticker, make it a DataFrame
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(batch[0])
            all_prices.append(prices)
            valid_tickers.extend(prices.columns.tolist())
        except Exception as e:
            print(f"Error fetching batch {batch}: {e}")
    if all_prices:
        result = pd.concat(all_prices, axis=1)
        # Drop columns with all NaNs (invalid tickers)
        result = result.dropna(axis=1, how='all')
        print(f"Fetched data for {len(result.columns)} valid tickers.")
        return result
    else:
        print("No data fetched.")
        return pd.DataFrame()

def calculate_returns(prices):
    """
    Calculate daily returns from price data.
    """
    returns = prices.pct_change().dropna()
    return returns

def get_sample_data():
    """
    Get sample stock data for testing (latest year).
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    prices = fetch_prices(tickers, start_date, end_date)
    returns = calculate_returns(prices)
    
    return prices, returns

if __name__ == "__main__":
    print("🚀 Smart Portfolio - Data Module")
    print("=" * 40)
    
    # Test the data fetching functionality
    prices, returns = get_sample_data()
    
    if not prices.empty:
        print(f"\n📊 Price Data Shape: {prices.shape}")
        print("\n💰 Latest Prices:")
        print(prices.tail())
        
        print(f"\n📈 Returns Data Shape: {returns.shape}")
        print("\n📊 Average Daily Returns:")
        print(returns.mean())
        
        print("\n✅ Data module working correctly!")
    else:
        print("❌ Failed to fetch data")
