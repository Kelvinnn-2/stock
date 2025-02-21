import pandas as pd

def calculate_ma_technical_rating(df):
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
    rating = "NEUTRAL"
    confidence = "Medium"
    details = []
    
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
    
    # Default case
    else:
        emoji = "🔍"
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


#V2 Price>ma 
from typing import Dict, List, Tuple

def calculate_pricema_rating(df: pd.DataFrame) -> Dict:
    """
    Calculate technical rating based on price relationships with moving averages.
    
    Args:
        df: DataFrame with price data and MAs (MA60, MA200 required)
        
    Returns:
        Dict containing rating details:
            - rating: String rating classification
            - emoji: Visual indicator
            - confidence: Rating confidence level
            - details: List of supporting reasons
    """
    # Get latest values
    current_price = df['Close'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    
    # Calculate percentage differences
    pct_from_ma60 = ((current_price - ma60) / ma60) * 100
    pct_from_ma200 = ((current_price - ma200) / ma200) * 100
    
    # Initialize rating components
    rating = ""
    emoji = ""
    confidence = ""
    details = []
    
    # Strong Buy Conditions
    if current_price > ma60 and pct_from_ma60 > 2 and current_price > ma200:
        rating = "BUY"
        emoji = "✅"
        confidence = "Strong"
        details = [
            f"Price is {pct_from_ma60:.1f}% above 60-day MA, showing strong momentum",
            f"Confirmed uptrend with price above both 60-day and 200-day MAs",
            "Volume and price action support bullish momentum"
        ]
    
    # Weak Buy Conditions
    elif current_price > ma60 and current_price < ma200:
        rating = "BUY"
        emoji = "⚠️"
        confidence = "Weak"
        details = [
            f"Price is {pct_from_ma60:.1f}% above 60-day MA, showing potential momentum",
            "Still below 200-day MA, waiting for long-term trend confirmation",
            "Monitor for increasing volume to confirm trend"
        ]
    
    # Hold Conditions
    elif abs(pct_from_ma60) < 2:
        rating = "HOLD"
        emoji = "⏳"
        confidence = "Neutral"
        details = [
            f"Price is close to 60-day MA (within {abs(pct_from_ma60):.1f}%)",
            "No clear trend direction established",
            "Wait for clearer directional signals"
        ]
    
    # Weak Sell Conditions
    elif current_price < ma60 and current_price > ma200:
        rating = "SELL"
        emoji = "⚠️"
        confidence = "Weak"
        details = [
            f"Price is {abs(pct_from_ma60):.1f}% below 60-day MA, showing weakness",
            "Still above 200-day MA support level",
            "Monitor for potential further weakness"
        ]
    
    # Strong Sell Conditions
    elif current_price < ma60 and current_price < ma200:
        rating = "SELL"
        emoji = "❌"
        confidence = "Strong"
        details = [
            f"Price is below both 60-day and 200-day MAs",
            f"Currently {abs(pct_from_ma60):.1f}% below 60-day MA",
            "Confirmed downtrend with bearish momentum"
        ]
    
    # Reversal Buy Opportunity
    elif current_price > ma200 * 0.98 and current_price < ma200 * 1.02 and current_price < ma60:
        rating = "BUY"
        emoji = "🔄"
        confidence = "Reversal Opportunity"
        details = [
            "Price testing 200-day MA support level",
            "Potential oversold bounce opportunity",
            "Watch for volume confirmation of reversal"
        ]
    
    # Overbought Sell Signal
    elif pct_from_ma60 > 10 and current_price > ma200:
        rating = "SELL"
        emoji = "❗"
        confidence = "Profit Taking"
        details = [
            f"Price extended {pct_from_ma60:.1f}% above 60-day MA",
            "Overbought conditions suggest profit taking",
            "Consider reducing position size"
        ]
    
    # Add momentum context
    details.append(get_momentum_context(df))
    
    return {
        "rating": rating,
        "emoji": emoji,
        "confidence": confidence,
        "details": details
    }

def get_momentum_context(df: pd.DataFrame) -> str:
    """
    Get additional context about price momentum.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        String describing recent price momentum
    """
    # Calculate 5-day price change
    five_day_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
    
    if five_day_change > 5:
        return f"Strong upward momentum with {five_day_change:.1f}% gain over last 5 days"
    elif five_day_change > 2:
        return f"Moderate upward momentum with {five_day_change:.1f}% gain over last 5 days"
    elif five_day_change < -5:
        return f"Strong downward momentum with {abs(five_day_change):.1f}% loss over last 5 days"
    elif five_day_change < -2:
        return f"Moderate downward momentum with {abs(five_day_change):.1f}% loss over last 5 days"
    else:
        return f"Neutral momentum with {abs(five_day_change):.1f}% change over last 5 days"