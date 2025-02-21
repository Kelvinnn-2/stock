import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz

# Function to check if time is during market hours
def is_market_hours(dt):
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

# Function to filter out non-market hours
def filter_market_hours(df):
    if df.empty:
        return df
    
    # Make sure index is datetime with timezone info
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Filter only market hours data
    df_filtered = df[df.index.map(is_market_hours)]
    return df_filtered

# Function to calculate technical indicators and rating
def calculate_technical_rating(df):
    # Calculate Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
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

# Streamlit App
def main():
    st.title("Stock Price Technical Analysis")
    st.subheader("Enter a stock symbol and select a timeframe")
    
    # User input for stock symbol
    symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()
    
    # Timeframe selection
    timeframes = {
        "1mo": ("5y", "1d"), 
        "1d": ("1y", "1d"), 
        "4h": ("90d", "1h"), 
        "1h": ("30d", "30m"),
        "30m": ("15d", "15m"),
        "15m": ("7d", "5m")
    }
    timeframe = st.selectbox("Select timeframe", list(timeframes.keys()))
    
    # Determine the correct period and interval
    period, interval = timeframes[timeframe]
    
    # Option to remove after-hours trading
    remove_after_hours = st.checkbox("Remove after-hours trading", value=True)
    
    # Option to show/hide moving averages
    show_mas = st.checkbox("Show Moving Averages", value=True)
    
    if st.button("Get Data"):
        with st.spinner(f"Fetching {symbol} data..."):
            try:
                # Fetch stock data
                ticker_data = yf.Ticker(symbol)
                df = ticker_data.history(period=period, interval=interval)
                
                if df.empty:
                    st.error("No data found. Please check the stock symbol or timeframe.")
                    return
                
                # Filter out non-trading periods (zero volume)
                df = df[df["Volume"] > 0]
                
                # Remove after-hours data if requested
                if remove_after_hours and interval in ['5m', '15m', '30m', '1h']:
                    df = filter_market_hours(df)
                
                # Calculate technical indicators
                df['MA10'] = df['Close'].rolling(window=10).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA60'] = df['Close'].rolling(window=60).mean()
                df['MA200'] = df['Close'].rolling(window=200).mean()
                
                # Get technical rating
                rating_data = calculate_technical_rating(df)
                
                # Create Candlestick Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol,
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    line=dict(width=1.5)  # Make candlesticks more visible
                ))
                
                # Add Moving Averages if requested
                if show_mas:
                    ma_colors = {
                        'MA10': '#1f77b4',  # Blue
                        'MA20': '#ff7f0e',  # Orange
                        'MA60': '#2ca02c',  # Green
                        'MA200': '#d62728'  # Red
                    }
                    
                    for ma, color in ma_colors.items():
                        if ma in df.columns and not df[ma].isnull().all():
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[ma],
                                name=ma,
                                line=dict(color=color, width=1.5),
                                connectgaps=True
                            ))
                
                # Add volume as a bar chart at the bottom
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker=dict(color='rgba(100, 100, 100, 0.5)'),
                    yaxis='y2'
                ))
                
                # Determine suitable range buttons based on selected timeframe
                if interval in ['5m', '15m', '30m']:
                    range_buttons = [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=5, label="5d", step="day", stepmode="backward"),
                        dict(step="all")
                    ]
                elif interval == '1h':
                    range_buttons = [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=5, label="5d", step="day", stepmode="backward"),
                        dict(count=10, label="10d", step="day", stepmode="backward"),
                        dict(step="all")
                    ]
                else:
                    range_buttons = [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                
                # Enhance the layout
                fig.update_layout(
                    title={
                        'text': f"Technical Analysis for {symbol}",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=True,
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=range_buttons
                        ),
                        showline=True, 
                        linewidth=1.2, 
                        linecolor='black',
                        type='date'  # Ensure proper date handling
                    ),
                    yaxis=dict(
                        fixedrange=False,
                        showgrid=True,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinecolor='black',
                        title="Price"
                    ),
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        visible=False
                    ),
                    autosize=True,
                    height=600,
                    hovermode="x unified",
                    plot_bgcolor="#f5f5f5",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Display MA Rating
                st.subheader(f"MA Rating: {rating_data['emoji']} {rating_data['rating']} ({rating_data['confidence']})")
                
                for detail in rating_data['details']:
                    st.write(f"• {detail}")
                
                # Create a box with explanations
                with st.expander("Technical Rating Explanation"):
                    st.markdown("""
                    **Rating System Explanation:**
                    
                    - **✅ BUY (Strong)** - Strong upward trend with confirmation from multiple moving averages
                    - **⚠️ BUY (Weak)** - Short-term uptrend starting, momentum is building
                    - **⏳ HOLD** - No clear trend direction, wait for clearer signals
                    - **⚠️ SELL (Weak)** - Short-term trend weakening, caution advised
                    - **❌ SELL (Strong)** - Downtrend confirmed across multiple timeframes
                    - **🔄 BUY (Reversal Opportunity)** - Potential for a bounce from oversold conditions
                    - **❗ SELL (Profit Taking)** - Stock is overbought, consider locking in profits
                    
                    *This technical analysis is for informational purposes only and should not be considered financial advice.*
                    """)
                
                # Moving average comparison table
                if show_mas:
                    st.subheader("Moving Average Analysis")
                    ma_df = pd.DataFrame({
                        'Indicator': ['MA10', 'MA20', 'MA60', 'MA200'],
                        'Value': [
                            f"${df['MA10'].iloc[-1]:.2f}" if not pd.isna(df['MA10'].iloc[-1]) else "N/A",
                            f"${df['MA20'].iloc[-1]:.2f}" if not pd.isna(df['MA20'].iloc[-1]) else "N/A", 
                            f"${df['MA60'].iloc[-1]:.2f}" if not pd.isna(df['MA60'].iloc[-1]) else "N/A",
                            f"${df['MA200'].iloc[-1]:.2f}" if not pd.isna(df['MA200'].iloc[-1]) else "N/A"
                        ],
                        'Relation to Price': [
                            f"Price is {'above' if df['Close'].iloc[-1] > df['MA10'].iloc[-1] else 'below'} MA10" if not pd.isna(df['MA10'].iloc[-1]) else "N/A",
                            f"Price is {'above' if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else 'below'} MA20" if not pd.isna(df['MA20'].iloc[-1]) else "N/A",
                            f"Price is {'above' if df['Close'].iloc[-1] > df['MA60'].iloc[-1] else 'below'} MA60" if not pd.isna(df['MA60'].iloc[-1]) else "N/A",
                            f"Price is {'above' if df['Close'].iloc[-1] > df['MA200'].iloc[-1] else 'below'} MA200" if not pd.isna(df['MA200'].iloc[-1]) else "N/A"
                        ]
                    })
                    st.table(ma_df)
                
                # Display price stats
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("Current", f"${df['Close'].iloc[-1]:.2f}", 
                              f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f} ({(df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100:.2f}%)")
                with stats_cols[1]:
                    st.metric("High", f"${df['High'].max():.2f}")
                with stats_cols[2]:
                    st.metric("Low", f"${df['Low'].min():.2f}")
                with stats_cols[3]:
                    st.metric("Volume", f"{df['Volume'].sum():,}")
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table with expandable details
                with st.expander("View Raw Data"):
                    st.dataframe(df.reset_index())
                
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.info("If you're seeing timezone errors, this is likely due to yfinance API limitations. Try a different timeframe.")

if __name__ == "__main__":
    main()