import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data, get_available_timeframes
from technical_analysis import calculate_ma_technical_rating, calculate_pricema_rating
from visualisation import create_candlestick_chart
from rsi import calculate_rsi, rsi_signal

def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist"""
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'last_timeframe' not in st.session_state:
        st.session_state.last_timeframe = None

def display_technical_analysis(df: pd.DataFrame, ma_rating: dict, price_ma_rating: dict):
    """Display technical analysis information (without RSI in the columns)"""
    # Move Technical Rating Explanation above the ratings
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

    # Create two columns for the ratings (MA and Price/MA)
    col1, col2 = st.columns(2)
    
    # MA Rating in first column
    with col1:
        st.subheader(f"MA Rating: {ma_rating['emoji']} {ma_rating['rating']} ({ma_rating['confidence']})")
        for detail in ma_rating['details']:
            st.write(f"• {detail}")
    
    # Price/MA Rating in second column
    with col2:
        st.subheader(f"Price/MA Rating: {price_ma_rating['emoji']} {price_ma_rating['rating']} ({price_ma_rating['confidence']})")
        for detail in price_ma_rating['details']:
            st.write(f"• {detail}")

def display_moving_averages(df: pd.DataFrame):
    """Display moving average analysis table"""
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
    
    


def display_price_statistics(df: pd.DataFrame):
    """Display price statistics including 52-week high/low with fallback to min/max if insufficient data"""
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
    """Calculate technical indicators for the dataset"""
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Stock Price Technical Analysis",
        page_icon="📈",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("Stock Price Technical Analysis")
    st.subheader("Enter a stock symbol and select a timeframe")
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()
    with col2:
        timeframes = get_available_timeframes()
        timeframe = st.selectbox("Select timeframe", list(timeframes.keys()))
    
    # Additional options
    col3, col4 = st.columns(2)
    with col3:
        show_mas = st.checkbox("Show Moving Averages", value=True)
        st.markdown("Applicable for US stock only:")
        remove_after_hours = st.checkbox("Remove after-hours trading", value=True)
    
    if (st.button("Get Data") or 
        symbol != st.session_state.last_symbol or 
        timeframe != st.session_state.last_timeframe):
        
        st.session_state.last_symbol = symbol
        st.session_state.last_timeframe = timeframe
        
        period, interval = timeframes[timeframe]
        
        with st.spinner(f"Fetching {symbol} data..."):
            try:
                # Fetch and process data
                df = fetch_stock_data(symbol, period, interval, remove_after_hours)
                
                if df.empty:
                    st.error("No data found. Please check the stock symbol or timeframe.")
                    return
                
                # Calculate indicators and get analysis
                df = calculate_technical_indicators(df)
                ma_rating = calculate_ma_technical_rating(df)
                price_ma_rating = calculate_pricema_rating(df)
                
                # Display price statistics
                display_price_statistics(df)
                
                # Create and display chart
                fig = create_candlestick_chart(df, symbol, show_mas, interval)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                with st.expander("View Raw Data"):
                    st.dataframe(df.reset_index())
                
                # Display analysis components (MA & Price/MA ratings)
                display_technical_analysis(df, ma_rating, price_ma_rating)
                
                # Display moving averages (if checked)
                if show_mas:
                    display_moving_averages(df)
                
                # Now display RSI as a separate table below the MAs
                display_rsi_analysis(df)
                
            except ValueError as ve:
                st.error(f"Error: {str(ve)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("If you're seeing timezone errors, this is likely due to yfinance API limitations. Try a different timeframe.")

if __name__ == "__main__":
    main()
