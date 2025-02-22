import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given dataset.
    
    Parameters:
    data (pd.Series): Series of closing prices
    window (int): Lookback period for RSI calculation (default is 14)
    
    Returns:
    pd.Series: RSI values
    """
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_signal(rsi: float) -> Dict[str, any]:
    """
    Returns a dictionary containing the RSI evaluation with an emoji, rating,
    confidence level, and a list of details based on the RSI value.
    """
    if rsi > 80:
        return {
            'emoji': '❗',
            'rating': 'SELL (Profit Taking)',
            'confidence': 'Strong',
            'details': [
                'RSI indicates extremely high levels, suggesting profit taking.'
            ]
        }
    elif 56 <= rsi <= 70:
        return {
            'emoji': '⚠️',
            'rating': 'BUY with Caution (Momentum is building)',
            'confidence': 'Medium',
            'details': [
                'RSI suggests that upward momentum is building, but caution is advised.'
            ]
        }
    elif 40 <= rsi <= 44:
        return {
            'emoji': '✅',
            'rating': 'BUY (Strong upward trend)',
            'confidence': 'High',
            'details': [
                'RSI indicates a strong upward trend in the market.'
            ]
        }
    elif 45 <= rsi <= 55:
        return {
            'emoji': '⏳',
            'rating': 'HOLD (No strong trend)',
            'confidence': 'Neutral',
            'details': [
                'RSI is in a neutral range, suggesting no clear trend.'
            ]
        }
    elif 35 <= rsi < 40:
        return {
            'emoji': '⚠️',
            'rating': 'SELL with Caution (Bearish momentum forming)',
            'confidence': 'Medium',
            'details': [
                'RSI suggests that bearish momentum may be starting to form.'
            ]
        }
    elif 30 <= rsi < 35:
        return {
            'emoji': '❌',
            'rating': 'SELL (Downtrend confirmed)',
            'confidence': 'Strong',
            'details': [
                'RSI confirms a downtrend in the market.'
            ]
        }
    elif 20 < rsi < 30:
        return {
            'emoji': '🔄',
            'rating': 'BUY (Reversal Opportunity)',
            'confidence': 'Strong',
            'details': [
                'RSI indicates oversold conditions, offering a potential reversal opportunity.'
            ]
        }
    else:
        return {
            'emoji': '❓',
            'rating': 'No Clear Signal',
            'confidence': 'N/A',
            'details': [
                'RSI value does not fall into any predefined category.'
            ]
        }
