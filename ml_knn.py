# ml_knn.py

import numpy as np
import pandas as pd

def normalize_series(series: pd.Series, length: int = 20, min_val: float = 0, max_val: float = 100) -> pd.Series:
    """
    Mimics the PineScript 'minimax' logic to scale a rolling window of data
    into a [min_val..max_val] range.
    """
    # Rolling highest/lowest
    roll_max = series.rolling(length).max()
    roll_min = series.rolling(length).min()
    scaled = (series - roll_min) / (roll_max - roll_min + 1e-9)  # avoid division by zero
    return scaled * (max_val - min_val) + min_val

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Basic RSI calculation. 
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Basic CCI calculation.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(length).mean()
    md = (tp - ma).abs().rolling(length).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

def compute_roc(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Rate of Change (ROC).
    """
    return (series - series.shift(length)) / (series.shift(length) + 1e-9) * 100

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Euclidean distance between 2 vectors p1, p2.
    """
    return np.sqrt(np.sum((p1 - p2)**2))

class KnnClassifier:
    """
    A simplified 2D kNN classifier that mimics the PineScript logic:
      - Two features (long-window, short-window).
      - 'direction' = sign of close[1] - close[0] for each bar.
      - For the current bar, we compute distance to all previous bars in the dataset,
        then pick the k nearest neighbors and sum their directions to form a final prediction.
      - If sum > 0 => BUY, sum < 0 => SELL, else HOLD.
    """

    def __init__(
        self,
        long_window: int = 28,
        short_window: int = 14,
        base_k: int = 252,
        indicator: str = "All", 
        use_vol_filter: bool = False,
        bar_threshold: int = 300
    ):
        self.long_window = long_window
        self.short_window = short_window
        self.base_k = base_k
        self.k = int(np.floor(np.sqrt(base_k)))  # same logic as Pine
        self.indicator = indicator  # "RSI", "CCI", "ROC", "Volume", or "All"
        self.use_vol_filter = use_vol_filter
        self.bar_threshold = bar_threshold

        # We will store the "training" data here:
        self.feature1 = []
        self.feature2 = []
        self.directions = []
        # For output
        self.predictions_ = None
        self.signals_ = None

    def _get_label(self, close_series: pd.Series) -> pd.Series:
        """
        For each bar, define label = sign(close[1] - close[0]).
        We'll shift -1 to align with the current bar.
        """
        future_close = close_series.shift(-1)
        label = np.where(
            future_close > close_series,  1,
            np.where(future_close < close_series, -1, 0)
        )
        return pd.Series(label, index=close_series.index)

    def _select_feature(self, df: pd.DataFrame, short: bool = True) -> pd.Series:
        """
        Returns the requested feature (long or short) depending on self.indicator.
        If 'All', we average them as in the PineScript code.
        """
        # RSI
        if short:
            rsi_ = compute_rsi(df['Close'], self.short_window)
            cci_ = compute_cci(df, self.short_window)
            roc_ = compute_roc(df['Close'], self.short_window)
            vol_ = normalize_series(df['Volume'], self.short_window, 0, 99)
        else:
            rsi_ = compute_rsi(df['Close'], self.long_window)
            cci_ = compute_cci(df, self.long_window)
            roc_ = compute_roc(df['Close'], self.long_window)
            vol_ = normalize_series(df['Volume'], self.long_window, 0, 99)

        if self.indicator == "RSI":
            return rsi_
        elif self.indicator == "CCI":
            return cci_
        elif self.indicator == "ROC":
            return roc_
        elif self.indicator == "Volume":
            return vol_
        else:
            # "All" => average
            return (rsi_ + cci_ + roc_ + vol_) / 4

    def fit(self, df: pd.DataFrame, start_date=None, end_date=None):
        """
        Build the 'training set' from the DataFrame rows that fall within [start_date, end_date].
        We'll store feature1, feature2, directions in Python lists.
        """
        # If user specified date range, filter
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        df = df.copy()

        # Compute features
        feat1_series = self._select_feature(df, short=False)  # long
        feat2_series = self._select_feature(df, short=True)   # short
        label_series = self._get_label(df['Close'])

        # We'll fill NAs with 0 or drop them
        feat1_series = feat1_series.ffill().fillna(0)
        feat2_series = feat2_series.ffill().fillna(0)
        label_series = label_series.fillna(0)

        # Convert to lists
        self.feature1 = feat1_series.tolist()
        self.feature2 = feat2_series.tolist()
        self.directions = label_series.astype(int).tolist()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each bar in df, compute:
         1) the features (f1, f2),
         2) distance to all stored points,
         3) pick the k nearest neighbors,
         4) sum directions => prediction,
         5) interpret as +1 (BUY), -1 (SELL), 0 (HOLD).
        We'll also do a simple 'volatility filter' if enabled.
        We'll store the final signals in df['knn_signal'].
        """
        # We'll build the output arrays
        predictions = np.zeros(len(df))
        signals = np.zeros(len(df))

        # Precompute features for df
        feat1_series = self._select_feature(df, short=False).fillna(0)
        feat2_series = self._select_feature(df, short=True).fillna(0)
        feat1_vals = feat1_series.values
        feat2_vals = feat2_series.values

        # Optional: volatility filter => ta.atr(10) > ta.atr(40)
        # We'll do a quick approximation in Python
        # If you want the exact Pine approach, implement a custom ATR function
        # or use e.g. pip install ta / use 'ta.volatility.AverageTrueRange'
        # For demonstration, we'll do a naive approach:
        def rolling_atr(series, window=14):
            # naive ATR approach for demonstration
            # Typically you'd want High/Low/Close for a real ATR
            return series.diff().abs().rolling(window).mean().fillna(0)

        if self.use_vol_filter:
            # We'll approximate using just close data (NOT recommended in real usage)
            atr10 = rolling_atr(df['Close'], 10)
            atr40 = rolling_atr(df['Close'], 40)
            vol_filter_array = (atr10 > atr40).astype(int)  # 1 or 0
        else:
            vol_filter_array = np.ones(len(df), dtype=int)

        # We'll also track a bar threshold for "holding" a position
        bar_counter = 0
        max_bars = self.bar_threshold
        current_signal = 0

        # We'll store the training data in arrays for fast access
        f1_train = np.array(self.feature1)
        f2_train = np.array(self.feature2)
        dir_train = np.array(self.directions)

        # We'll do the classification for each bar
        for i in range(len(df)):
            f1_val = feat1_vals[i]
            f2_val = feat2_vals[i]
            # Compute distances to all stored points
            dists = np.sqrt((f1_val - f1_train)**2 + (f2_val - f2_train)**2)
            # Sort ascending
            idx_sorted = np.argsort(dists)
            # Take the first k
            k_indices = idx_sorted[:self.k]
            # Sum directions
            sum_dir = dir_train[k_indices].sum()
            predictions[i] = sum_dir

            # Convert sum_dir => buy, sell, hold
            # + => buy, - => sell, else hold
            if sum_dir > 0 and vol_filter_array[i] == 1:
                # check bar threshold
                if bar_counter >= max_bars:
                    current_signal = 0
                    bar_counter = 0
                else:
                    current_signal = 1
            elif sum_dir < 0 and vol_filter_array[i] == 1:
                if bar_counter >= max_bars:
                    current_signal = 0
                    bar_counter = 0
                else:
                    current_signal = -1
            else:
                current_signal = 0

            # if changed from previous
            if i > 0 and current_signal != signals[i-1]:
                bar_counter = 0
            else:
                bar_counter += 1

            signals[i] = current_signal

        df['knn_prediction'] = predictions
        df['knn_signal'] = signals
        self.predictions_ = predictions
        self.signals_ = signals
        return df
