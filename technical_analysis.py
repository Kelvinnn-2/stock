import pandas as pd
from typing import Dict, List, Optional, Union, Literal

def calculate_ma_technical_rating(df: pd.DataFrame) -> Dict[str, Union[str, List[str]]]:
    """
    Calculate technical indicators and generate trading signals
    
    Args:
        df: DataFrame with OHLC price data and moving averages
    
    Returns:
        dict: Rating information including signal, confidence level, and details
    """
    # Get the most recent values
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 2 else None
    
    # Check if we have enough data for MA200
    has_long_term_data = not pd.isna(latest['MA200'])
    
    # Initialize rating details
    rating: Literal["BUY", "HOLD", "SELL", "NEUTRAL"] = "NEUTRAL"
    confidence: str = "Medium"
    details: List[str] = []
    emoji: str = "🔍"
    
    # Strong Buy Signal (Bullish Momentum) - Case 1
    if has_long_term_data and latest['MA10'] > latest['MA20'] > latest['MA60'] > latest['MA200']:
        rating = "BUY"
        confidence = "Strong"
        details.append("Strong upward trend with confirmation across all moving averages")
        emoji = "✅"
    
    # Weak Buy Signal (Early Uptrend) - Case 2
    elif prev is not None and prev['MA10'] <= prev['MA20'] and latest['MA10'] > latest['MA20']:
        rating = "BUY"
        confidence = "Weak"
        details.append("Short-term uptrend starting (MA10 crossing above MA20)")
        emoji = "⚠️"
    
    # Hold Signal (Sideways Movement) - Case 3
    elif latest['MA10'] and latest['MA20'] and abs(latest['MA10'] - latest['MA20'])/latest['MA20'] < 0.01:
        rating = "HOLD"
        confidence = "Medium"
        details.append("Moving averages are flat with no clear trend direction")
        emoji = "⏳"
    
    # Weak Sell Signal (Early Downtrend) - Case 4
    elif prev is not None and prev['MA10'] >= prev['MA20'] and latest['MA10'] < latest['MA20']:
        rating = "SELL"
        confidence = "Weak"
        details.append("Short-term trend weakening (MA10 crossing below MA20)")
        emoji = "⚠️"
    
    # Strong Sell Signal (Bearish Breakdown) - Case 5
    elif has_long_term_data and latest['MA10'] < latest['MA20'] < latest['MA60'] < latest['MA200']:
        rating = "SELL"
        confidence = "Strong"
        details.append("Strong downtrend confirmed across all moving averages")
        emoji = "❌"
    
    # Reversal Buy Signal (Oversold Condition) - Case 6
    elif prev is not None and prev['MA10'] < prev['MA20'] and latest['MA10'] < latest['MA20'] and \
         (latest['MA10'] - prev['MA10']) > 0 and (latest['MA10'] - latest['MA20'])/latest['MA20'] > -0.01:
        rating = "BUY"
        confidence = "Reversal Opportunity"
        details.append("MA10 turning upward, potential for reversal")
        emoji = "🔄"
    
    # Overbought Sell Signal (Potential Reversal) - Case 7
    elif has_long_term_data and latest['MA10'] > latest['MA20'] > latest['MA60'] > latest['MA200'] and \
         (latest['MA10'] - latest['MA20'])/latest['MA20'] > 0.05:
        rating = "SELL"
        confidence = "Profit Taking"
        details.append("Stock is overbought; consider locking in profits")
        emoji = "❗"
    
    # Default case is handled by the initial values
    else:
        details.append("Not enough clear signals")
    
    # Additional indicator checks
    if 'MA10' in df.columns and 'MA20' in df.columns:
        recent_trend = df['MA10'].iloc[-5:].diff().mean()
        if recent_trend > 0 and rating == "BUY":
            details.append("Short-term momentum is positive")
        elif recent_trend < 0 and rating == "SELL":
            details.append("Short-term momentum is negative")
    
    return {
        'rating': rating,
        'confidence': confidence,
        'details': details,
        'emoji': emoji
    }

def calculate_pricema_rating(df: pd.DataFrame) -> Dict[str, Union[str, List[str]]]:
    """Calculate price-based moving average rating."""
    if df.empty or len(df) < 2:
        return {
            "rating": "NEUTRAL",
            "emoji": "🔍",
            "confidence": "Low",
            "details": ["Not enough data for analysis"]
        }

    latest = df.iloc[-1]
    ma60: Optional[float] = latest.get('MA60', None)
    ma200: Optional[float] = latest.get('MA200', None)

    if ma60 is None or ma200 is None:
        return {
            "rating": "NEUTRAL",
            "emoji": "🔍",
            "confidence": "Low",
            "details": ["Missing moving averages"]
        }

    current_price: float = latest['Close']
    pct_from_ma60: float = ((current_price - ma60) / ma60) * 100
    pct_from_ma200: float = ((current_price - ma200) / ma200) * 100

    rating: Literal["BUY", "HOLD", "SELL", "NEUTRAL"] = "NEUTRAL"
    emoji: str = "🔍"
    confidence: str = "Medium" 
    details: List[str] = []

    if current_price > ma60 and pct_from_ma60 > 2 and current_price > ma200:
        rating, emoji, confidence = "BUY", "✅", "Strong"
        details = [
            f"Price is {pct_from_ma60:.1f}% above MA60, indicating strong momentum",
            f"Confirmed uptrend as price is above MA200"
        ]
    elif current_price > ma60 and current_price < ma200:
        rating, emoji, confidence = "BUY", "⚠️", "Weak"
        details = [
            f"Price is {pct_from_ma60:.1f}% above MA60, but below MA200",
            "Trend needs confirmation"
        ]
    elif abs(pct_from_ma60) < 2:
        rating, emoji, confidence = "HOLD", "⏳", "Neutral"
        details = [
            f"Price is within {abs(pct_from_ma60):.1f}% of MA60, indicating sideways movement"
        ]
    elif current_price < ma60 and current_price > ma200:
        rating, emoji, confidence = "SELL", "⚠️", "Weak"
        details = [
            f"Price is {abs(pct_from_ma60):.1f}% below MA60, showing weakness",
            "Still above MA200 support"
        ]
    elif current_price < ma60 and current_price < ma200:
        rating, emoji, confidence = "SELL", "❌", "Strong"
        details = [
            f"Price is {abs(pct_from_ma60):.1f}% below MA60 and also below MA200",
            "Confirmed downtrend"
        ]

    # Add momentum context only if enough data is available
    if len(df) >= 5:
        details.append(get_momentum_context(df))

    return {
        "rating": rating,
        "emoji": emoji,
        "confidence": confidence,
        "details": details
    }

def get_momentum_context(df: pd.DataFrame) -> str:
    """Evaluate recent price momentum with safeguards against data length issues."""
    if df.empty or len(df) < 5:
        return "Not enough data for momentum analysis"

    five_day_change: float = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100

    if five_day_change > 5:
        return f"Strong upward momentum with {five_day_change:.1f}% gain in last 5 days"
    elif five_day_change > 2:
        return f"Moderate upward momentum with {five_day_change:.1f}% gain in last 5 days"
    elif five_day_change < -5:
        return f"Strong downward momentum with {abs(five_day_change):.1f}% loss in last 5 days"
    elif five_day_change < -2:
        return f"Moderate downward momentum with {abs(five_day_change):.1f}% loss in last 5 days"
    else:
        return f"Neutral momentum with {five_day_change:.1f}% change in last 5 days"