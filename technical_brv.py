# technical_brv.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    A typical RSI (Wilder's) implementation in Python.
    """
    delta = series.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_function(
    close_series: pd.Series,
    sensitivity: float,
    rsi_period: int,
    rsi_base: float
) -> pd.Series:
    """
    Replicates the Pine Script logic:
        rsi_modified = sensitivity * (rsi(close, rsiPeriod) - rsiBase)
        then clamps values between 0 and 20.
    """
    raw_rsi = compute_rsi(close_series, rsi_period)
    rsi_modified = sensitivity * (raw_rsi - rsi_base)
    
    # Clamp to [0, 20]
    rsi_modified = np.where(rsi_modified > 20, 20, rsi_modified)
    rsi_modified = np.where(rsi_modified < 0, 0, rsi_modified)
    
    return pd.Series(rsi_modified, index=close_series.index)

def calculate_brv_rating(
    df: pd.DataFrame,
    RSIBaseBanker: float = 50,
    RSIPeriodBanker: int = 50,
    RSIBaseHotMoney: float = 30,
    RSIPeriodHotMoney: int = 40,
    SensitivityBanker: float = 1.5,
    SensitivityHotMoney: float = 0.7
) -> pd.DataFrame:
    """
    Creates three columns in the DataFrame that mimic the Pine Script output:
    - 'Retailer': constant 20
    - 'HotMoney': rsi_function(...) for "Hot Money"
    - 'Banker':   rsi_function(...) for "Banker"
    """
    df['Retailer'] = 20  # Constant from your original script
    
    df['HotMoney'] = rsi_function(
        df['Close'], 
        SensitivityHotMoney, 
        RSIPeriodHotMoney, 
        RSIBaseHotMoney
    )
    
    df['Banker'] = rsi_function(
        df['Close'], 
        SensitivityBanker, 
        RSIPeriodBanker, 
        RSIBaseBanker
    )
    
    return df

def interpret_brv_signals(banker: float, hot_money: float, retailer: float) -> dict:
    """
    Returns a dictionary containing emoji, rating, confidence, and a list of details
    about the BRV (Banker-Retailer Volume) signal.
    """
    # Default values
    rating = "NEUTRAL"
    confidence = "Medium"
    details = []
    emoji = "🔍"
    
    # Case 1: Strong BUY
    if banker > 15:
        rating = "BUY"
        confidence = "Strong"
        details.append("Banker control is high.")
        emoji = "✅"
    
    # Case 2: Moderate BUY
    elif banker > 10:
        rating = "BUY"
        confidence = "Moderate"
        details.append("Banker involvement is moderate.")
        emoji = "⚠️"
    
    # Case 3: Speculative BUY
    elif banker>5 and hot_money > 10:
        rating = "BUY"
        confidence = "Speculative"
        details.append("Hot money is flowing in.")
        emoji = "⚠️"
    
    # Case 4: Weak SELL
    elif banker < 5 and hot_money < 5:
        rating = "SELL"
        confidence = "Weak"
        details.append("Very low banker/hot money interest.")
        emoji = "⚠️"
    
    # Default: HOLD (Uncertain)
    else:
        rating = "HOLD"
        confidence = "Uncertain"
        details.append("No strong signals from Banker/Hot Money.")
        emoji = "⏳"
    
    return {
        'emoji': emoji,
        'rating': rating,
        'confidence': confidence,
        'details': details
    }

def display_brv_stacked_chart(df: pd.DataFrame):
    df['Banker_segment'] = df['Banker']
    df['HotMoney_segment'] = df['HotMoney']
    df['Retailer_segment'] = 20 - (df['Banker'] + df['HotMoney'])
    df['Retailer_segment'] = df['Retailer_segment'].clip(lower=0)
    
    fig = go.Figure()
    
    # Include hours/minutes so each interval is unique
    x_cats = df.index.strftime('%Y-%m-%d %H:%M')
    
    fig.add_trace(go.Bar(x=x_cats, y=df['Banker_segment'],   name='Banker',    marker_color='red'))
    fig.add_trace(go.Bar(x=x_cats, y=df['HotMoney_segment'], name='HotMoney',  marker_color='yellow'))
    fig.add_trace(go.Bar(x=x_cats, y=df['Retailer_segment'], name='Retailer',  marker_color='green'))
    
    fig.update_layout(
        barmode='stack',
        yaxis=dict(range=[0, 20]),
        xaxis=dict(type='category'),  # still a categorical axis
        title='Banker-Retailer Volume (Stacked)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Remove temporary columns
    df.drop(['Banker_segment', 'HotMoney_segment', 'Retailer_segment'], axis=1, inplace=True)

