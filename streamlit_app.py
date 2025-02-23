# main.py

import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data, get_available_timeframes
from technical_analysis import calculate_ma_technical_rating, calculate_pricema_rating
from visualisation import create_candlestick_chart
from rsi import calculate_rsi, rsi_signal

# Import your BRV logic
from technical_brv import (
    calculate_brv_rating,
    interpret_brv_signals,
    display_brv_stacked_chart
)

def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist."""
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'last_timeframe' not in st.session_state:
        st.session_state.last_timeframe = None

def display_technical_analysis(df: pd.DataFrame, ma_rating: dict, price_ma_rating: dict):
    """Display technical analysis information (without RSI in the columns)."""
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"MA Rating: {ma_rating['emoji']} {ma_rating['rating']} ({ma_rating['confidence']})")
        for detail in ma_rating['details']:
            st.write(f"• {detail}")
    with col2:
        st.subheader(f"Price/MA Rating: {price_ma_rating['emoji']} {price_ma_rating['rating']} ({price_ma_rating['confidence']})")
        for detail in price_ma_rating['details']:
            st.write(f"• {detail}")

def display_moving_averages(df: pd.DataFrame):
    """Display moving average analysis table."""
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

def display_rsi_analysis(df: pd.DataFrame):
    """Display RSI rating in a style similar to the MA rating."""
    rsi_value = df['RSI'].iloc[-1]
    rsi_rating = rsi_signal(rsi_value)
    st.subheader(f"RSI Rating: {rsi_rating['emoji']} {rsi_rating['rating']} ({rsi_rating['confidence']})")
    for detail in rsi_rating['details']:
        st.write(f"• {detail}")

def display_brv_analysis(df: pd.DataFrame):
    """Displays the Banker-Retailer Volume analysis in Streamlit."""
    st.subheader("Banker-Retailer Volume (BRV)")
    if not {'Retailer', 'HotMoney', 'Banker'}.issubset(df.columns):
        st.warning("BRV columns not found. Make sure you ran 'calculate_brv_rating(df)' first.")
        return

    # Get the latest values
    latest_banker   = df['Banker'].iloc[-1]
    latest_hotmoney = df['HotMoney'].iloc[-1]
    latest_retailer = df['Retailer'].iloc[-1]

    # Interpret signals as a dictionary
    brv_dict = interpret_brv_signals(latest_banker, latest_hotmoney, latest_retailer)

    # Display the rating
    st.subheader(f"BRV Rating: {brv_dict['emoji']} {brv_dict['rating']} ({brv_dict['confidence']})")
    for detail in brv_dict['details']:
        st.write(f"• {detail}")

    # Show numeric values
    st.write(f"**Banker**: {latest_banker:.2f} | **Hot Money**: {latest_hotmoney:.2f} | **Retailer**: {latest_retailer:.2f}")
    
    # Display the stacked bar chart
    display_brv_stacked_chart(df)

def display_price_statistics(df: pd.DataFrame):
    """Display price statistics including 52-week high/low, etc."""
    if len(df) >= 252:
        high_52w = df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
        low_52w = df['Low'].rolling(window=252, min_periods=1).min().iloc[-1]
    else:
        high_52w = df['High'].max()
        low_52w = df['Low'].min()

    stats_cols1 = st.columns(4)
    with stats_cols1[0]:
        st.metric("Current", f"${df['Close'].iloc[-1]:.2f}", 
                  f"{(df['Close'].iloc[-1] - df['Close'].iloc[-2]):.2f} ({(df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100:.2f}%)")
    with stats_cols1[1]:
        st.metric("52-Week High", f"${high_52w:.2f}")
    with stats_cols1[2]:
        st.metric("52-Week Low", f"${low_52w:.2f}")
    with stats_cols1[3]:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standard technical indicators for the dataset."""
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def main():
    st.set_page_config(
        page_title="Stock Price Technical Analysis",
        page_icon="📈",
        layout="wide"
    )
    
    initialize_session_state()
    
    # -- SIDEBAR (Nav Bar) Inputs --
    st.sidebar.title("Stock Price Technical Analysis")
    st.sidebar.subheader("Configuration")

    symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()
    
    timeframes = get_available_timeframes()
    timeframe = st.sidebar.selectbox("Select timeframe", list(timeframes.keys()))
    
    show_mas = st.sidebar.checkbox("Show Moving Averages", value=True)
    remove_after_hours = st.sidebar.checkbox("Remove after-hours trading", value=True)
    
    # "Get Data" button in the sidebar
    fetch_button = st.sidebar.button("Get Data")
    
    # -- MAIN PAGE TITLE --
    st.title("Stock Price Technical Analysis")
    
    if fetch_button or symbol != st.session_state.last_symbol or timeframe != st.session_state.last_timeframe:
        # Update session state
        st.session_state.last_symbol = symbol
        st.session_state.last_timeframe = timeframe
        
        period, interval = timeframes[timeframe]
        
        with st.spinner(f"Fetching {symbol} data..."):
            try:
                df = fetch_stock_data(symbol, period, interval, remove_after_hours)
                
                if df.empty:
                    st.error("No data found. Please check the stock symbol or timeframe.")
                    return
                
                # 1. Calculate standard indicators
                df = calculate_technical_indicators(df)
                
                # 2. MA & Price/MA rating
                ma_rating = calculate_ma_technical_rating(df)
                price_ma_rating = calculate_pricema_rating(df)
                
                # 3. Banker/HotMoney/Retailer columns
                df = calculate_brv_rating(df)
                
                # 4. Display price stats
                display_price_statistics(df)
                
                # 5. Candlestick chart
                fig = create_candlestick_chart(df, symbol, show_mas, interval)
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Show raw data in an expander
                with st.expander("View Raw Data"):
                    st.dataframe(df.reset_index())
                
                # 7. Display MA & Price/MA rating
                display_technical_analysis(df, ma_rating, price_ma_rating)
                
                # 8. Optionally show MAs
                if show_mas:
                    display_moving_averages(df)
                
                # 9. RSI rating
                display_rsi_analysis(df)
                
                # 10. BRV analysis
                display_brv_analysis(df)
                
            except ValueError as ve:
                st.error(f"Error: {str(ve)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("If you're seeing timezone errors, this is likely due to yfinance API limitations. Try a different timeframe.")

if __name__ == "__main__":
    main()
