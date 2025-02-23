import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data, get_available_timeframes
from technical_analysis import calculate_ma_technical_rating, calculate_pricema_rating
from visualisation import create_candlestick_chart
from rsi import calculate_rsi, rsi_signal
from technical_brv import calculate_brv_rating, interpret_brv_signals, display_brv_stacked_chart
from ml_knn import KnnClassifier  

def initialize_session_state():
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'last_timeframe' not in st.session_state:
        st.session_state.last_timeframe = None

def display_technical_analysis(df: pd.DataFrame, ma_rating: dict, price_ma_rating: dict):
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
    rsi_value = df['RSI'].iloc[-1]
    rsi_rating = rsi_signal(rsi_value)
    st.subheader(f"RSI Rating: {rsi_rating['emoji']} {rsi_rating['rating']} ({rsi_rating['confidence']})")
    for detail in rsi_rating['details']:
        st.write(f"• {detail}")

def display_brv_analysis(df: pd.DataFrame):
    st.subheader("Banker-Retailer Volume (BRV)")
    if not {'Retailer','HotMoney','Banker'}.issubset(df.columns):
        st.warning("BRV columns not found. Make sure you ran 'calculate_brv_rating(df)' first.")
        return
    latest_banker   = df['Banker'].iloc[-1]
    latest_hotmoney = df['HotMoney'].iloc[-1]
    latest_retailer = df['Retailer'].iloc[-1]
    brv_dict = interpret_brv_signals(latest_banker, latest_hotmoney, latest_retailer)
    st.subheader(f"BRV Rating: {brv_dict['emoji']} {brv_dict['rating']} ({brv_dict['confidence']})")
    for detail in brv_dict['details']:
        st.write(f"• {detail}")
    st.write(f"**Banker**: {latest_banker:.2f} | **Hot Money**: {latest_hotmoney:.2f} | **Retailer**: {latest_retailer:.2f}")
    display_brv_stacked_chart(df)

def display_price_statistics(df: pd.DataFrame):
    if len(df) >= 252:
        high_52w = df['High'].rolling(252).max().iloc[-1]
        low_52w  = df['Low'].rolling(252).min().iloc[-1]
    else:
        high_52w = df['High'].max()
        low_52w  = df['Low'].max()
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
    df['MA10']  = df['Close'].rolling(10).mean()
    df['MA20']  = df['Close'].rolling(20).mean()
    df['MA60']  = df['Close'].rolling(60).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['RSI']   = calculate_rsi(df['Close'])
    return df

def subset_data_for_display(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Subset of the DataFrame for chart display:
      - 1mo,1wk: all data
      - 1d: last 365 days
      - 4h: last 90 days
      - 1h: last 30 days
      - 30m: last 15 days
      - 15m: last 7 days
    """
    if df.empty:
        return df
    now = df.index[-1]
    if timeframe in ["1mo","1wk"]:
        return df
    elif timeframe == "1d":
        start = now - pd.Timedelta(days=365)
    elif timeframe == "4h":
        start = now - pd.Timedelta(days=90)
    elif timeframe == "1h":
        start = now - pd.Timedelta(days=30)
    elif timeframe == "30m":
        start = now - pd.Timedelta(days=15)
    elif timeframe == "15m":
        start = now - pd.Timedelta(days=7)
    else:
        return df
    return df[df.index >= start]


def run_knn_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs kNN classification on df, displays signals, and returns df with added columns:
    'knn_prediction' and 'knn_signal'.

    The function uses the kNN prediction sum combined with the current RSI to classify the signal into one of seven categories:
      - ✅ BUY (Strong)
      - ⚠️ BUY (Weak)
      - ⏳ HOLD
      - ⚠️ SELL (Weak)
      - ❌ SELL (Strong)
      - 🔄 BUY (Reversal Opportunity)
      - ❗ SELL (Profit Taking)
      
    It then outputs a detailed explanation of the signal.
    """
    try:
        if df.empty:
            return df
        
        # Basic kNN configuration
        knn_indicator = "All"
        short_window  = 14
        long_window   = 28
        base_k        = 252
        vol_filter    = False
        bar_thresh    = 300

        # Instantiate the kNN classifier
        knn = KnnClassifier(
            long_window   = long_window,
            short_window  = short_window,
            base_k        = base_k,
            indicator     = knn_indicator,
            use_vol_filter= vol_filter,
            bar_threshold = bar_thresh
        )
        
        # Fit the classifier on the entire dataset range
        start_date = df.index[0]
        end_date   = df.index[-1]
        knn.fit(df, start_date, end_date)
        
        # Predict signals
        df_knn = knn.predict(df)
        
        # Store results in the main DataFrame
        df['knn_prediction'] = df_knn['knn_prediction']
        df['knn_signal']     = df_knn['knn_signal']
        
        # Get the latest prediction sum and RSI value
        latest_prediction = df['knn_prediction'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50.0
        
        rating_label = ""
        explanation = ""
        
        # Classify final signal into one of 7 categories using both kNN sum and RSI:
        if latest_prediction < 0 and latest_rsi < 30:
            rating_label = "🔄 BUY (Reversal Opportunity)"
            explanation = (
                "Although the kNN prediction sum is negative—indicating historical conditions were bearish—"
                " the current RSI is very low (below 30), which signals oversold conditions. This suggests that the market "
                "may be poised for a bounce, making it a potential reversal opportunity to buy."
            )
        elif latest_prediction > 0 and latest_rsi > 70:
            rating_label = "❗ SELL (Profit Taking)"
            explanation = (
                "Even though the kNN prediction sum is positive—implying bullish historical momentum—the current RSI is very high "
                "(above 70), indicating overbought conditions. This suggests that the market might be due for a correction, so it "
                "could be a good time to sell and lock in profits."
            )
        elif latest_prediction >= 10:
            rating_label = "✅ BUY (Strong)"
            explanation = (
                "The kNN prediction sum is significantly positive (greater than or equal to 10), which strongly indicates that "
                "historical data under similar conditions led to upward price moves. This strong bullish bias suggests a robust buying signal."
            )
        elif latest_prediction >= 2:
            rating_label = "⚠️ BUY (Weak)"
            explanation = (
                "The kNN prediction sum is moderately positive (between 2 and 9), implying a mild bullish bias. This weak buy signal "
                "indicates that conditions are slightly favorable for an upward move, though the momentum is not as strong as a strong buy."
            )
        elif abs(latest_prediction) < 2:
            rating_label = "⏳ HOLD"
            explanation = (
                "The kNN prediction sum is close to zero, suggesting that the historical behavior of similar conditions does not "
                "favor a clear directional move. In this scenario, the market is in a holding pattern with no clear trend."
            )
        elif latest_prediction <= -10:
            rating_label = "❌ SELL (Strong)"
            explanation = (
                "The kNN prediction sum is significantly negative (less than or equal to -10), strongly indicating that historical data "
                "under similar conditions led to downward price movements. This strong bearish bias suggests a robust selling signal."
            )
        else:
            rating_label = "⚠️ SELL (Weak)"
            explanation = (
                "The kNN prediction sum is moderately negative (between -9 and -2), implying a weak bearish bias. This suggests a mild "
                "sell signal, where conditions are slightly unfavorable."
            )
        
        # Display the final classification with detailed explanation
        st.subheader(f"ML kNN-based Signal: {rating_label}")
        st.write("Latest kNN Prediction Sum:", f"{latest_prediction:.2f}")
        st.write("Latest RSI:", f"{latest_rsi:.2f}")
        st.write(explanation)
        
    except Exception as e_knn:
        st.error(f"kNN Error: {str(e_knn)}")
    
    return df

def main():
    st.set_page_config(
        page_title="Stock Price Technical Analysis",
        page_icon="📈",
        layout="wide"
        # initial_sidebar_state="collapsed"
    )
    initialize_session_state()
    
    st.sidebar.title("Stock Price Technical Analysis")
    st.sidebar.subheader("Configuration")
    symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()
    timeframes = get_available_timeframes()
    timeframe  = st.sidebar.selectbox("Select timeframe", list(timeframes.keys()))
    show_mas   = st.sidebar.checkbox("Show Moving Averages", value=True)
    remove_after_hours = st.sidebar.checkbox("Remove after-hours trading", value=True)
    
    if st.sidebar.button("Get Data") or symbol != st.session_state.last_symbol or timeframe != st.session_state.last_timeframe:
        st.session_state.last_symbol    = symbol
        st.session_state.last_timeframe = timeframe
        
        # We'll fetch '5y' for daily or weekly intervals, but intraday is overridden inside fetch_stock_data
        # for example: "4h" => actually "730d" internally, "30m"/"15m" => "60d" etc.
        period, interval = timeframes[timeframe]
        
        st.title("Stock Price Technical Analysis")
        with st.spinner(f"Fetching {symbol} data..."):
            try:
                df = fetch_stock_data(symbol, period, interval, remove_after_hours)
                if df.empty:
                    st.error("No data found. Please check the stock symbol or timeframe.")
                    return
                
                # 1) Compute standard indicators
                df = calculate_technical_indicators(df)
                
                # 2) Compute MA & Price/MA rating
                ma_rating = calculate_ma_technical_rating(df)
                price_ma_rating = calculate_pricema_rating(df)
                
                # 3) Compute Banker-Retailer Volume
                df = calculate_brv_rating(df)
                
                # 4) Display Price Statistics
                display_price_statistics(df)
                
                # 5) Subset Data for Chart Display
                df_display = subset_data_for_display(df, timeframe)
                
                # 6) Create Candlestick Chart from subset
                fig = create_candlestick_chart(df_display, symbol, show_mas, interval)
                st.plotly_chart(fig, use_container_width=True)
                
                
                # 7) Display MA & Price/MA rating
                display_technical_analysis(df, ma_rating, price_ma_rating)
                
                # 8) Optionally display MAs
                if show_mas:
                    display_moving_averages(df)
                
                # 9) RSI rating
                display_rsi_analysis(df)
                
                # 10) kNN-based signals
                run_knn_classification(df)
                
                # 11) BRV analysis
                display_brv_analysis(df)
            
            except ValueError as ve:
                st.error(f"Error: {str(ve)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("If you're seeing timezone errors, this is likely due to yfinance API limitations. Try a different timeframe.")

if __name__ == "__main__":
    main()
