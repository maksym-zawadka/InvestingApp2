"""
models/lstm_model.py
────────────────────
Integracja wytrenowanego modelu LSTM z aplikacją Streamlit.

Pliki wymagane w folderze models/:
    model.keras   – wytrenowany model Keras
    scaler.pkl    – RobustScaler fitowany na 13 cechach

Wejście: sekwencja 30 świec × 13 cech
Wyjście: P(wzrost następnej sesji) [0–1, sigmoid]
"""

import os
import pickle
import numpy as np
import pandas as pd

SEQUENCE_LEN = 15

FEATURE_COLS = [
    'Price_Change_1',
    'Price_Change_5',
    'Candle_Body_Ratio',
    'RSI',
    'MACD_Hist',
    'Close_SMA10_Ratio',
    'Close_SMA20_Ratio',
    'Bollinger_Percent_B',
    'ATR_Normalized',
    'Volatility_10',
    'Volume_Ratio',
    'Day_of_Week_Sin',
    'Day_of_Week_Cos',
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    v = df['Volume'].astype(float)

    # Zwroty
    df['Price_Change_1'] = c.pct_change(1)
    df['Price_Change_5'] = c.pct_change(5)

    # Stosunek ciała świecy do całego zakresu (wicks)
    body  = (df['Close'] - df['Open']).abs()
    total = (h - l).replace(0, np.nan)
    df['Candle_Body_Ratio'] = body / total

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD Histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal

    # Close vs SMA
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    df['Close_SMA10_Ratio'] = c / (sma10 + 1e-10)
    df['Close_SMA20_Ratio'] = c / (sma20 + 1e-10)

    # Bollinger %B
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df['Bollinger_Percent_B'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ATR znormalizowany
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['ATR_Normalized'] = atr14 / (c + 1e-10)

    # Zmienność 10-dniowa
    df['Volatility_10'] = c.pct_change().rolling(10).std()

    # Volume Ratio
    vol_sma10 = v.rolling(10).mean()
    df['Volume_Ratio'] = v / (vol_sma10 + 1e-10)

    # Dzień tygodnia (kodowanie cykliczne)
    dow = pd.to_datetime(df.index).dayofweek.astype(float)
    df['Day_of_Week_Sin'] = np.sin(2 * np.pi * dow / 5)
    df['Day_of_Week_Cos'] = np.cos(2 * np.pi * dow / 5)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load():
    base        = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base, "model.keras")
    scaler_path = os.path.join(base, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak pliku: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Brak pliku: {scaler_path}")

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return LSTMModel(model, scaler)


class LSTMModel:
    def __init__(self, model, scaler):
        self.model  = model
        self.scaler = scaler

    def predict(self, df: pd.DataFrame, ticker: str) -> tuple:
        enriched = add_indicators(df)

        if len(enriched) < SEQUENCE_LEN:
            raise ValueError(
                f"Za mało danych: {len(enriched)} świec po obliczeniu wskaźników "
                f"(minimum {SEQUENCE_LEN}). Rozszerz zakres dat do co najmniej 60 dni."
            )

        features        = enriched[FEATURE_COLS].values[-SEQUENCE_LEN:]   # (30, 13)
        features_scaled = self.scaler.transform(features)                  # (30, 13)
        X               = features_scaled[np.newaxis, ...]                 # (1, 30, 13)

        prob = float(self.model.predict(X, verbose=0)[0][0])

        last_close     = float(df["Close"].iloc[-1])
        avg_daily_move = float(df["Close"].pct_change().abs().tail(20).mean())
        direction      = 1 if prob >= 0.5 else -1
        confidence     = abs(prob - 0.5) * 2
        predicted_price = round(last_close * (1 + direction * confidence * avg_daily_move), 2)

        return predicted_price, prob
