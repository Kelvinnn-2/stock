from ml_knn import KnnClassifier
import streamlit as st
import pandas as pd
from rsi import calculate_rsi, rsi_signal
from technical_brv import  interpret_brv_signals, display_brv_stacked_chart

#######################################
# 1. CSV Loading and Ticker Lookup    #
#######################################
def load_klse_mapping() -> pd.DataFrame:
    return pd.read_csv("KLSE.csv")

def partial_match_malaysia(user_input: str, mapping_df: pd.DataFrame) -> str:
    """
    Partial match 'user_input' in 'Company' first, then 'Ticker'.
    If found, append '.KL' if not present. If no match, fallback => user_input + '.KL'.
    """
    user_input = user_input.strip().upper()
    
    if not {"Ticker","Company"}.issubset(mapping_df.columns):
        # If columns missing, fallback
        return user_input + ".KL"
    
    mapping_df["Ticker"]  = mapping_df["Ticker"].astype(str).str.upper()
    mapping_df["Company"] = mapping_df["Company"].astype(str).str.upper()
    
    def partial_match(col: str) -> str:
        matched = mapping_df[mapping_df[col].str.contains(user_input, na=False)]
        if not matched.empty:
            t = matched.iloc[0]["Ticker"].strip().upper()
            if not t.endswith(".KL"):
                t += ".KL"
            return t
        return ""
    
    # 1) Match in Company
    result = partial_match("Company")
    if result:
        return result
    
    # 2) Match in Ticker
    result = partial_match("Ticker")
    if result:
        return result
    
    # 3) Fallback
    return user_input + ".KL"

#######################################
# 2. Session and UI Setup            #
#######################################
def initialize_session_state():
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'last_timeframe' not in st.session_state:
        st.session_state.last_timeframe = None
    # Default to "Malaysia" if region_mode not set
    if 'region_mode' not in st.session_state:
        st.session_state.region_mode = 'Malaysia'

#######################################
# 3. Display Functions                #
#######################################
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
            f"${df['MA10'].iloc[-1]:.2f}" if 'MA10' in df.columns and not pd.isna(df['MA10'].iloc[-1]) else "N/A",
            f"${df['MA20'].iloc[-1]:.2f}" if 'MA20' in df.columns and not pd.isna(df['MA20'].iloc[-1]) else "N/A",
            f"${df['MA60'].iloc[-1]:.2f}" if 'MA60' in df.columns and not pd.isna(df['MA60'].iloc[-1]) else "N/A",
            f"${df['MA200'].iloc[-1]:.2f}" if 'MA200' in df.columns and not pd.isna(df['MA200'].iloc[-1]) else "N/A"
        ],
        'Relation to Price': [
            f"Price is {'above' if df['Close'].iloc[-1] > df['MA10'].iloc[-1] else 'below'} MA10" if 'MA10' in df.columns else "N/A",
            f"Price is {'above' if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else 'below'} MA20" if 'MA20' in df.columns else "N/A",
            f"Price is {'above' if df['Close'].iloc[-1] > df['MA60'].iloc[-1] else 'below'} MA60" if 'MA60' in df.columns else "N/A",
            f"Price is {'above' if df['Close'].iloc[-1] > df['MA200'].iloc[-1] else 'below'} MA200" if 'MA200' in df.columns else "N/A"
        ]
    })
    st.table(ma_df)

def display_rsi_analysis(df: pd.DataFrame):
    if 'RSI' not in df.columns:
        return
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
    latest_banker = df['Banker'].iloc[-1]
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
        low_52w = df['Low'].rolling(252).min().iloc[-1]
    else:
        high_52w = df['High'].max()
        low_52w = df['Low'].max()
    stats_cols1 = st.columns(4)
    with stats_cols1[0]:
        st.metric("Current", f"${df['Close'].iloc[-1]:.2f}",
                  f"{(df['Close'].iloc[-1]-df['Close'].iloc[-2]):.2f} ({(df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100:.2f}%)")
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

#######################################
# 4. kNN-based Classification         #
#######################################
def run_knn_classification(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return df
        
        knn_indicator = "All"
        short_window  = 14
        long_window   = 28
        base_k        = 252
        vol_filter    = False
        bar_thresh    = 300

        knn = KnnClassifier(
            long_window   = long_window,
            short_window  = short_window,
            base_k        = base_k,
            indicator     = knn_indicator,
            use_vol_filter= vol_filter,
            bar_threshold = bar_thresh
        )
        
        start_date = df.index[0]
        end_date   = df.index[-1]
        knn.fit(df, start_date, end_date)
        
        df_knn = knn.predict(df)
        
        df['knn_prediction'] = df_knn['knn_prediction']
        df['knn_signal']     = df_knn['knn_signal']
        
        latest_prediction = df['knn_prediction'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50.0
        
        if latest_prediction < 0 and latest_rsi < 30:
            rating_label = "🔄 BUY (Reversal Opportunity)"
            explanation = (
                "Although the kNN prediction sum is negative—indicating historical conditions were bearish—"
                "the current RSI is very low (below 30), which signals oversold conditions. This suggests that "
                "the market may be poised for a bounce."
            )
        elif latest_prediction > 0 and latest_rsi > 70:
            rating_label = "❗ SELL (Profit Taking)"
            explanation = (
                "Even though the kNN prediction sum is positive, the RSI is above 70, indicating overbought conditions. "
                "The market may be due for a correction, so it could be a good time to sell and lock in profits."
            )
        elif latest_prediction >= 10:
            rating_label = "✅ BUY (Strong)"
            explanation = "A strongly positive kNN sum (≥10) indicates robust bullish bias."
        elif latest_prediction >= 2:
            rating_label = "⚠️ BUY (Weak)"
            explanation = "A moderately positive kNN sum (2–9) suggests mild bullish conditions."
        elif abs(latest_prediction) < 2:
            rating_label = "⏳ HOLD"
            explanation = "The kNN sum is near zero, implying no clear directional bias."
        elif latest_prediction <= -10:
            rating_label = "❌ SELL (Strong)"
            explanation = "A strongly negative kNN sum (≤-10) indicates robust bearish bias."
        else:
            rating_label = "⚠️ SELL (Weak)"
            explanation = "A moderately negative kNN sum (–9 to –2) suggests mild bearish conditions."
        
        st.subheader(f"ML kNN-based Signal: {rating_label}")
        st.write("Latest kNN Prediction Sum:", f"{latest_prediction:.2f}")
        st.write("Latest RSI:", f"{latest_rsi:.2f}")
        st.write("Explanation:", explanation)
        
    except Exception as e_knn:
        st.error(f"kNN Error: {str(e_knn)}")
    return df