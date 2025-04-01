import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data, get_available_timeframes
from technical_analysis import calculate_ma_technical_rating, calculate_pricema_rating
from visualisation import create_candlestick_chart
from technical_brv import calculate_brv_rating
from data_display import (
    load_klse_mapping, partial_match_malaysia, initialize_session_state, calculate_technical_indicators,
    subset_data_for_display, run_knn_classification, display_price_statistics, display_technical_analysis,
    display_moving_averages, display_rsi_analysis, display_brv_analysis
)
from stock_predictor import train_and_predict_30_steps
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.set_page_config(page_title="Stock Price Technical Analysis", page_icon="📈", layout="wide")
    initialize_session_state()
    
    #----------------------------------
    # Toggle for Malaysia vs. Global
    #----------------------------------
    col1, col2 = st.columns(2)
    with col1:
        malaysia_btn = st.button("🇲🇾 Malaysia")
    with col2:
        global_btn = st.button("🌐 Global")
    
    if malaysia_btn:
        st.session_state.region_mode = "Malaysia"
    if global_btn:
        st.session_state.region_mode = "Global"
    
    st.markdown(f"**Region Mode:** {st.session_state.region_mode}")
    
    #----------------------------------
    # Symbol input
    #----------------------------------
    st.sidebar.title("Stock Price Technical Analysis")
    st.sidebar.subheader("Configuration")
    user_input = st.sidebar.text_input("Enter stock code or name", "velesto")
    
    ticker_symbol = ""
    if st.session_state.region_mode == "Malaysia":
        klse_df = load_klse_mapping()
        ticker_symbol = partial_match_malaysia(user_input, klse_df)
    else:
        ticker_symbol = user_input.strip().upper()
    
    st.sidebar.write("Using Ticker:", ticker_symbol)
    
    #----------------------------------
    # Timeframes, checkboxes, etc.
    #----------------------------------
    timeframes = get_available_timeframes()
    timeframe = st.sidebar.selectbox("Select timeframe", list(timeframes.keys()))
    show_mas = st.sidebar.checkbox("Show Moving Averages", value=True)
    
    if st.session_state.region_mode == "Malaysia":
        remove_after_hours = False
    else:
        remove_after_hours = st.sidebar.checkbox("Remove after-hours trading", value=True)
    
    #----------------------------------
    # On button click, fetch data
    #----------------------------------
    if st.sidebar.button("Get Data") or ticker_symbol != st.session_state.last_symbol or timeframe != st.session_state.last_timeframe:
        st.session_state.last_symbol = ticker_symbol
        st.session_state.last_timeframe = timeframe
        
        full_period = "5y"
        _, interval = timeframes[timeframe]
        
        st.title("Stock Price Technical Analysis")
        with st.spinner(f"Fetching {ticker_symbol} data..."):
            df = fetch_stock_data(ticker_symbol, full_period, interval, remove_after_hours)
            
            if df.empty:
                st.info(f"No data found for symbol {ticker_symbol}. Possibly not recognized by Yahoo.")
                return
            
            # 1) Indicators
            df = calculate_technical_indicators(df)
            
            # 2) MA & Price/MA rating
            ma_rating = calculate_ma_technical_rating(df)
            price_ma_rating = calculate_pricema_rating(df)
            
            # 3) BRV
            df = calculate_brv_rating(df)
            
            # 4) Stats
            display_price_statistics(df)
            
            # 5) Subset data for chart
            df_display = subset_data_for_display(df, timeframe)
            
            # 6) Chart
            fig = create_candlestick_chart(df_display, ticker_symbol, show_mas, interval)
            st.plotly_chart(fig, use_container_width=True)
            
            # 7) Raw data
            with st.expander("View Raw Data"):
                st.dataframe(df.reset_index())
            
            # 8) Tech analysis
            display_technical_analysis(df, ma_rating, price_ma_rating)
            
            # 9) MAs
            if show_mas:
                display_moving_averages(df)
            
            # 10) RSI
            display_rsi_analysis(df)
            
            # 11) kNN
            st.subheader("ML kNN-based Signals")
            run_knn_classification(df)
            
            # 12) BRV
            display_brv_analysis(df)

    #----------------------------------
    # Prediction button
    #----------------------------------
    if st.sidebar.button("Predict Next 30 Steps"):
        with st.spinner(f"Predicting next 30 days for {ticker_symbol}..."):
            try:
                st.info(f"Starting prediction process for {ticker_symbol}")
                
                # Run the prediction
                future_dates, predicted_values, (lower_bound, upper_bound), df_hist = train_and_predict_30_steps(ticker_symbol, timeframe)
                
                st.info(f"Prediction complete. Forecast generated for {len(future_dates)} days.")

                if len(future_dates) == 0:
                    st.error(f"Prediction failed for {ticker_symbol}. Try a different timeframe or check data availability.")
                    return

                # Convert predictions to DataFrame for display
                future_df = pd.DataFrame({
                    "Date": future_dates, 
                    "Predicted Price": predicted_values,
                    "Lower Bound": lower_bound,
                    "Upper Bound": upper_bound
                })
                st.subheader(f"Predicted Stock Prices for {ticker_symbol} (Backtest + 30d Forecast)")
                st.dataframe(future_df)

                #----------------------------------
                # Plot only the last 1 year of historical data + backtest + forecast
                #----------------------------------

                # 1) Compute 1-year cutoff from the last historical date
                if len(df_hist) == 0:
                    st.warning("No historical data in df_hist to plot.")
                    return
                one_year_ago = df_hist.index[-1] - pd.DateOffset(days=365)

                # 2) Slice df_hist to keep only last 1 year
                df_plot = df_hist.loc[df_hist.index >= one_year_ago]

                # 3) Find where future_dates crosses that cutoff
                start_idx = 0
                for i, d in enumerate(future_dates):
                    if d >= one_year_ago:
                        start_idx = i
                        break

                # 4) Slice predictions & dates from start_idx onward
                plot_dates = future_dates[start_idx:]
                plot_pred  = predicted_values[start_idx:]
                plot_low   = lower_bound[start_idx:]
                plot_up    = upper_bound[start_idx:]

                # 5) Plot
                plt.figure(figsize=(12, 6))
                # Last 1 year of real data in blue
                plt.plot(df_plot.index, df_plot["Close"], label="Historical Prices (Last 1y)", color='blue')

                # Overwrite portion of predictions that falls in last 1y, plus future
                plt.plot(plot_dates, plot_pred, color='orange', label="Backtest + Forecast")
                plt.fill_between(plot_dates, plot_low, plot_up, color='orange', alpha=0.2, label="95% CI")

                plt.title(f"{ticker_symbol} (Last 1 Year) + 30d Forecast")
                plt.xlabel("Date")
                plt.ylabel("Stock Price")
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
