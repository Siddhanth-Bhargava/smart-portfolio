import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_prices(tickers, start_date, end_date):
    """
    Download adjusted close prices for given tickers & date range.
    Returns a DataFrame indexed by date.
    """
    try:
        print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
        data = yf.download(tickers, start=start_date, end=end_date)
        
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Data shape: {data.shape}")
        
        # Extract adjusted close prices
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif ('Adj Close', tickers[0]) in data.columns:
            # Multi-level columns case
            prices = data['Adj Close']
        else:
            # Fallback to Close prices if Adj Close not available
            prices = data['Close']
        
        print(f"Successfully downloaded data for {len(prices)} trading days")
        return prices.dropna()
    except Exception as e:
        print(f"Error fetching data: {e}")
        print(f"Available data columns: {data.columns.tolist() if 'data' in locals() else 'No data'}")
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
