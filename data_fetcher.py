import yfinance as yf
import pandas as pd
from datetime import time
import pytz

def is_market_hours(timestamp, tz='America/New_York'):
    """
    Check if the given timestamp falls within regular market hours (9:30 AM - 4:00 PM ET).
    Returns True if it's a weekday and within market hours, False otherwise.
    """
    ny_tz = pytz.timezone(tz)
    ts_ny = timestamp.astimezone(ny_tz)
    
    market_start = time(9, 30)
    market_end = time(16, 0)
    
    return (ts_ny.time() >= market_start and 
            ts_ny.time() <= market_end and 
            ts_ny.weekday() < 5)

def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Given 1-hour OHLC data, resample it into 4-hour bars.
    We aggregate columns as follows:
      - Open  = first of the 4h window
      - High  = max of the 4h window
      - Low   = min of the 4h window
      - Close = last of the 4h window
      - Volume= sum of the 4h window
    """
    if df_1h.empty:
        return df_1h
    
    # Ensure the index is a DateTimeIndex with tz. If not, localize or convert.
    # For example, if your df_1h index is naive, you can do:
    #   df_1h.index = df_1h.index.tz_localize('UTC')
    # if needed. But often yfinance returns a tz-aware index.

    df_4h = df_1h.resample('4h').agg({
        'Open':  'first',
        'High':  'max',
        'Low':   'min',
        'Close': 'last',
        'Volume':'sum'
    }).dropna(subset=['Open','High','Low','Close'])  # ensure we drop any empty bars

    return df_4h

def fetch_stock_data(symbol: str, period: str, interval: str, remove_after_hours: bool = True) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    If the user requests "4h", we actually fetch 1-hour data (for ~180 days),
    then resample to 4-hour bars.
    
    Returns an empty DataFrame if no data or columns missing.
    """
    # Decide actual fetch interval/period
    actual_interval = interval
    actual_period   = period
    
    if interval == '4h':
        # We'll fetch 1-hour data for ~180 days, then resample to 4h
        actual_interval = '1h'
        actual_period   = '180d'  # about 6 months
    elif interval == '1h':
        # you might override period to e.g. 180d
        actual_period   = '180d'
    elif interval == '30m':
        actual_period   = '60d'
    elif interval == '15m':
        actual_period   = '30d'
    
    try:
        ticker = yf.Ticker(symbol)
        df_raw = ticker.history(period=actual_period, interval=actual_interval)
        
        if df_raw.empty:
            return pd.DataFrame()
        
        # Ensure index is DateTime
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index = pd.to_datetime(df_raw.index)
        
        # Remove after-hours data if requested
        intraday_list = ['1m','2m','5m','15m','30m','60m','90m','1h','4h']
        if remove_after_hours and actual_interval in intraday_list:
            df_raw = df_raw[df_raw.index.map(is_market_hours)]
        
        # Forward-fill missing values
        df_raw = df_raw.ffill()
        
        # If user requested 4h, resample the 1h data
        if interval == '4h':
            df_resampled = _resample_to_4h(df_raw)
        else:
            df_resampled = df_raw
        
        required = ['Open','High','Low','Close','Volume']
        if not all(c in df_resampled.columns for c in required):
            return pd.DataFrame()
        
        return df_resampled
    
    except Exception as e:
        # Return empty DataFrame if an unexpected error
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_available_timeframes():
    """
    Dictionary of timeframe -> (period, interval).
    We'll pass e.g. '5y' for '4h', but inside fetch_stock_data, we override to 1h+resample.
    """
    return {
        "1mo": ("5y", "1mo"),
        "1wk": ("5y", "1wk"),
        "1d":  ("1y", "1d"),
        "4h":  ("5y", "4h"),   # we'll fetch 1h & resample to 4h
        "1h":  ("5y", "1h"),   # we override to 180d
        "30m": ("5y", "30m"),  # override to 60d
        "15m": ("5y", "5m")    # override to 30d
    }
