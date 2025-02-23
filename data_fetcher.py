import yfinance as yf
import pandas as pd
from datetime import datetime, time
import pytz

def is_market_hours(timestamp, tz='America/New_York'):
    """
    Check if the given timestamp falls within regular market hours (9:30 AM - 4:00 PM ET).
    
    Args:
        timestamp: Timestamp to check
        tz: Timezone string (default: 'America/New_York')
        
    Returns:
        bool: True if timestamp is within market hours, False otherwise
    """
    ny_tz = pytz.timezone(tz)
    ts_ny = timestamp.astimezone(ny_tz)
    
    market_start = time(9, 30)
    market_end = time(16, 0)
    
    # Check if it's a weekday and within market hours
    return (ts_ny.time() >= market_start and 
            ts_ny.time() <= market_end and 
            ts_ny.weekday() < 5)

def fetch_stock_data(symbol: str, period: str, interval: str, remove_after_hours: bool = True) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance API.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        period: Time period to fetch (e.g., '1y', '6mo', '1d')
        interval: Data interval (e.g., '1d', '1h', '15m')
        remove_after_hours: Whether to remove after-hours trading data
        
    Returns:
        pd.DataFrame: DataFrame containing stock data with columns:
            - Datetime index
            - Open, High, Low, Close prices
            - Volume
            - Optional technical indicators
    
    Raises:
        ValueError: If invalid symbol or parameters provided
        Exception: For other API or processing errors
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch the data
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
            
        # Convert index to datetime if it isn't already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Remove after-hours data if requested and using intraday data
        if remove_after_hours and interval in ['1m', '5m', '15m', '30m', '1h']:
            df = df[df.index.map(is_market_hours)]
            
        # Forward fill any missing values
        df = df.ffill()

        df = df.ffill()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        return df
        
    except ValueError as ve:
        raise ValueError(f"Error fetching data for {symbol}: {str(ve)}")
    except Exception as e:
        raise Exception(f"Unexpected error fetching data for {symbol}: {str(e)}")

def get_available_timeframes():
    """
    Returns a dictionary of available timeframes and their corresponding period/interval pairs.
    
    Returns:
        dict: Dictionary mapping timeframe names to (period, interval) tuples
    """
    return {
        "1mo": ("5y", "1mo"),
        "1wk": ("5y", "1wk"),
        "1d": ("1y", "1d"),
        "4h": ("90d", "1h"),
        "1h": ("30d", "30m"),
        "30m": ("15d", "15m"),
        "15m": ("7d", "5m")
    }