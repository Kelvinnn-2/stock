import yfinance as yf
import pandas as pd
import pytz
from datetime import time

def is_market_hours(dt):
    """
    Check if time is during market hours
    
    Args:
        dt: Datetime object with timezone info
    
    Returns:
        bool: True if within market hours, False otherwise
    """
    # Convert to Eastern Time (US market hours)
    eastern = pytz.timezone('US/Eastern')
    dt_eastern = dt.astimezone(eastern)
    
    # Check if it's a weekday
    if dt_eastern.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Standard market hours: 9:30 AM to 4:00 PM Eastern
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    # Check if current time is within market hours
    current_time = dt_eastern.time()
    return market_open <= current_time <= market_close

def filter_market_hours(df):
    """
    Filter out non-market hours data from DataFrame
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame: Filtered DataFrame containing only market hours data
    """
    if df.empty:
        return df
    
    # Make sure index is datetime with timezone info
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Filter only market hours data
    df_filtered = df[df.index.map(is_market_hours)]
    return df_filtered

def fetch_stock_data(symbol, period, interval, remove_after_hours=True):
    """
    Fetch stock data from Yahoo Finance API
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL)
        period: Time period to fetch (e.g., '1y', '5d')
        interval: Data interval (e.g., '1d', '1h', '15m')
        remove_after_hours: Whether to filter out after-hours trading data
    
    Returns:
        DataFrame: Stock price data with OHLC and volume
    """
    # Fetch stock data
    ticker_data = yf.Ticker(symbol)
    df = ticker_data.history(period=period, interval=interval)
    
    if df.empty:
        return df
    
    # Filter out non-trading periods (zero volume)
    df = df[df["Volume"] > 0]
    
    # Remove after-hours data if requested
    if remove_after_hours and interval in ['5m', '15m', '30m', '1h']:
        df = filter_market_hours(df)
    
    return df