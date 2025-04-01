import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import matplotlib.pyplot as plt

from data_fetcher import fetch_stock_data

##############################################
# 0. Dynamic Period Selection
##############################################
def fetch_dynamic_period(symbol, timeframe, remove_after_hours=True):
    """
    Tries multiple periods in descending order to fetch the oldest available data.
    """
    period_candidates = ["max", "10y", "5y", "3y", "1y"]
    for p in period_candidates:
        df_try = fetch_stock_data(symbol, period=p, interval=timeframe, remove_after_hours=remove_after_hours)
        if not df_try.empty:
            print(f"Using period='{p}' for {symbol} with {len(df_try)} rows.")
            return df_try
    print(f"No data found for {symbol} in candidate periods.")
    return pd.DataFrame()

##############################################
# 1. Preprocessing & Feature Engineering
##############################################
def add_technical_indicators(df):
    """
    Adds advanced technical indicators plus Volume and ATR.
    You can expand this function with additional indicators as needed.
    """
    if df.empty:
        print("Error: DataFrame is empty before adding indicators.")
        return df

    # Ensure Volume column exists
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Bollinger Bands
    df['Boll_Middle'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Boll_Upper'] = df['Boll_Middle'] + (2 * rolling_std)
    df['Boll_Lower'] = df['Boll_Middle'] - (2 * rolling_std)

    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Stochastic Oscillator
    df['14-high'] = df['High'].rolling(window=14).max()
    df['14-low']  = df['Low'].rolling(window=14).min()
    df['Stochastic'] = (df['Close'] - df['14-low']) / (df['14-high'] - df['14-low']) * 100
    df.drop(columns=['14-high','14-low'], inplace=True)

    # Log Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # ATR (Average True Range)
    df['H-L']  = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low']  - df['Close'].shift(1))
    df['TR']   = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR']  = df['TR'].rolling(window=14).mean()
    df.drop(columns=['H-L','H-PC','L-PC','TR'], inplace=True)

    df.dropna(inplace=True)
    return df

def prepare_data(df, timeframe, horizon=30):
    """
    Reorders columns (ensuring 'Close' is first), locks the column order,
    then scales and builds sliding-window sequences.
    """
    if df.empty:
        print("Error: DataFrame is empty before scaling.")
        return np.array([]), np.array([]), None

    # Reorder so 'Close' is first
    cols = list(df.columns)
    if 'Close' in cols:
        cols.remove('Close')
        cols.insert(0, 'Close')
    final_cols = cols
    df = df[final_cols].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    lookback = 60 if timeframe == "1d" else 104 if timeframe == "1wk" else 168
    if len(scaled_data) < (lookback + horizon):
        print(f"Error: Not enough historical data. Need {lookback+horizon}, have {len(scaled_data)}")
        return np.array([]), np.array([]), None

    X, y = [], []
    for i in range(lookback, len(scaled_data) - horizon):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i+horizon, 0])  # Predict 'Close' horizon days ahead

    X = np.array(X)
    y = np.array(y)
    scaler.final_cols = final_cols  # store column order for later use
    return X, y, scaler

##############################################
# 2. Model Definition
##############################################
def build_advanced_model(input_shape, dropout=0.05, lr=1e-4, loss='mse'):
    """
    Builds a hybrid CNN-LSTM-GRU model with MSE.
    """
    from tensorflow.keras import layers, models, optimizers
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(dropout),
        layers.LayerNormalization(),
        layers.GRU(128, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    opt = optimizers.AdamW(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    return model

##############################################
# 3. Monte Carlo Dropout Single-Step Prediction
##############################################
@tf.function
def mc_dropout_predict(model, X_input, mc_samples=10):
    """
    Runs 'mc_samples' forward passes with dropout enabled.
    Returns the mean and standard deviation of predictions.
    """
    ta = tf.TensorArray(dtype=tf.float32, size=mc_samples)
    i = tf.constant(0)
    def cond(i, ta):
        return i < mc_samples
    def body(i, ta):
        pred = model(X_input, training=True)
        pred = tf.reshape(pred, (1,))
        ta = ta.write(i, pred)
        return i+1, ta
    i, ta = tf.while_loop(cond, body, [i, ta])
    preds = ta.stack()
    mean_pred = tf.reduce_mean(preds)
    std_pred = tf.math.reduce_std(preds)
    return mean_pred, std_pred

##############################################
# 4. Train + Walk-Forward Forecast with Dynamic Period
##############################################
def train_and_predict_30_steps(symbol, timeframe):
    """
    1) Dynamically fetches data using the oldest available period.
    2) Preprocesses & adds features.
    3) Trains or loads the model.
    4) Performs walk-forward backtesting (last 150 days) and iterative forecasting (next 60 days).
    """
    model_path = f"{symbol}_{timeframe}_enhanced_model.h5"
    # Use dynamic period selection to fetch the oldest possible data.
    df_raw = fetch_dynamic_period(symbol, timeframe, remove_after_hours=True)
    if df_raw.empty:
        print("Error: No data found for", symbol)
        return [], [], ([], []), pd.DataFrame()

    # Add features and preprocess
    df = df_raw.copy()
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    X, y, scaler = prepare_data(df, timeframe, horizon=60)  # horizon=60 for 60-day forecast
    if X.size == 0:
        return [], [], ([], []), df

    # Train/Val/Test split
    n = len(X)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Optionally, you can perform hyperparameter tuning here.
    best_lr, best_dropout = (1e-4, 0.05)

    from tensorflow.keras.models import load_model
    try:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            print(f"✅ Loaded existing model {model_path}")
        else:
            raise ValueError("No saved model found.")
    except:
        model = build_advanced_model((X_train.shape[1], X_train.shape[2]),
                                     dropout=best_dropout, lr=best_lr, loss='mse')
        print(f"🚀 Training final model for {symbol} ...")
        callbacks = []  
        model.fit(X_train, y_train, epochs=100, batch_size=32,
                  validation_data=(X_val, y_val), verbose=1, callbacks=callbacks)
        model.save(model_path)

    # Re-compile the model manually
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=best_lr), loss='mse')
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE:", test_loss)

    # Walk-forward backtest (last 150 days) and forecast (next 60 days)
    last_idx = len(df) - 1
    past_days = 150
    future_days = 60

    predicted_list = [None] * (past_days + future_days)
    lower_list = [None] * (past_days + future_days)
    upper_list = [None] * (past_days + future_days)

    final_cols = scaler.final_cols

    def get_scaled_input(day_idx):
        lookback = X_train.shape[1]
        start = day_idx - lookback
        end = day_idx
        if start < 0:
            return None
        raw_slice = df.loc[df.index[start:end], final_cols]
        if len(raw_slice) < lookback:
            return None
        scaled_slice = scaler.transform(raw_slice)
        return scaled_slice[np.newaxis, :, :]

    start_backtest = last_idx - past_days + 1
    for i in range(start_backtest, last_idx + 1):
        X_input = get_scaled_input(i)
        if X_input is None:
            continue
        mean_pred, std_pred = mc_dropout_predict(model, X_input, mc_samples=10)
        mp, sp = mean_pred.numpy(), std_pred.numpy()
        offset = i - start_backtest
        predicted_list[offset] = mp
        lower_list[offset] = mp - 1.5 * sp
        upper_list[offset] = mp + 1.5 * sp

    # Future iterative forecast
    lookback = X_train.shape[1]
    real_slice = df.loc[df.index[last_idx - lookback + 1:last_idx + 1], final_cols]
    scaled_slice = scaler.transform(real_slice)
    buffer = scaled_slice.copy()

    for fstep in range(future_days):
        X_input = buffer[np.newaxis, :, :]
        mean_pred, std_pred = mc_dropout_predict(model, X_input, mc_samples=10)
        mp, sp = mean_pred.numpy(), std_pred.numpy()
        offset = past_days + fstep
        predicted_list[offset] = mp
        lower_list[offset] = mp - 1.5 * sp
        upper_list[offset] = mp + 1.5 * sp

        new_row = buffer[-1].copy()
        new_row[0] = mp
        buffer = np.roll(buffer, -1, axis=0)
        buffer[-1] = new_row

    predicted_array = np.array(predicted_list, dtype=float)
    lower_array = np.array(lower_list, dtype=float)
    upper_array = np.array(upper_list, dtype=float)

    out_len = past_days + future_days
    zero_pad = np.zeros((out_len, X_train.shape[2] - 1))
    combined_pred = np.hstack([predicted_array.reshape(-1, 1), zero_pad])
    combined_lower = np.hstack([lower_array.reshape(-1, 1), zero_pad])
    combined_upper = np.hstack([upper_array.reshape(-1, 1), zero_pad])

    inv_pred = scaler.inverse_transform(combined_pred)[:, 0]
    inv_lower = scaler.inverse_transform(combined_lower)[:, 0]
    inv_upper = scaler.inverse_transform(combined_upper)[:, 0]

    date_index = df.index[(last_idx - past_days + 1):(last_idx + 1)].tolist()
    last_day = df.index[last_idx]
    for i in range(1, future_days + 1):
        date_index.append(last_day + datetime.timedelta(days=i))

    return date_index, inv_pred, (inv_lower, inv_upper), df
