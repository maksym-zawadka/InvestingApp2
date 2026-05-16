"""
Test modelu LSTM na danych z Yahoo Finance.
pip install yfinance tensorflow scikit-learn
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "model.keras"
SCALER_PATH = "scaler.pkl"
SEQUENCE_LEN = 30
TICKER = "nvda"
PERIOD = "2y"

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


# ============================================================
# INDICATORS (identyczne jak w treningu)
# ============================================================
def add_indicators(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    v = df['Volume'].astype(float)

    df['Price_Change_1'] = c.pct_change()
    df['Price_Change_5'] = c.pct_change(5)

    hl_range = h - l
    df['Candle_Body_Ratio'] = (c - o) / (hl_range + 1e-10)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_val = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss_val + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - macd_signal

    sma_10 = c.rolling(10).mean()
    df['Close_SMA10_Ratio'] = c / (sma_10 + 1e-10) - 1

    sma_20 = c.rolling(20).mean()
    df['Close_SMA20_Ratio'] = c / (sma_20 + 1e-10) - 1

    bb_std = c.rolling(20).std()
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std
    df['Bollinger_Percent_B'] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR_Normalized'] = atr / (c + 1e-10)

    df['Volatility_10'] = df['Price_Change_1'].rolling(10).std()

    vol_sma = v.rolling(20).mean()
    df['Volume_Ratio'] = v / (vol_sma + 1e-10)


    dow = df['Date'].dt.dayofweek
    df['Day_of_Week_Sin'] = np.sin(2 * np.pi * dow / 5)
    df['Day_of_Week_Cos'] = np.cos(2 * np.pi * dow / 5)

    df['Target'] = (c.shift(-1) > c).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# LOAD DATA
# ============================================================
raw = yf.download(TICKER, period=PERIOD, auto_adjust=True)
raw.columns = raw.columns.droplevel(1) if raw.columns.nlevels > 1 else raw.columns
raw.reset_index(inplace=True)



df = add_indicators(raw)
print(f"Po dodaniu wskaznikow: {len(df)} dni")

print("Laduje model i scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# ============================================================
# PREPARE SEQUENCES
# ============================================================
features = scaler.transform(df[FEATURE_COLS].values)
targets = df['Target'].values

X, y, dates = [], [], []
for i in range(len(features) - SEQUENCE_LEN + 1):
    X.append(features[i:i + SEQUENCE_LEN])
    y.append(targets[i + SEQUENCE_LEN - 1])
    dates.append(df['Date'].iloc[i + SEQUENCE_LEN - 1] if 'Date' in df.columns else i + SEQUENCE_LEN - 1)

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"Sekwencji do predykcji: {len(X)}")

# ============================================================
# PREDICT
# ============================================================
print("\nPredykcja...")
probs = model.predict(X, batch_size=512, verbose=0).flatten()
preds = (probs > 0.5).astype(int)

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*50}")
print(f"  WYNIKI: {TICKER}")
print(f"{'='*50}")

print(f"\nRozkad predykcji:")
print(f"  Wzrost: {(preds == 1).sum()} ({(preds == 1).mean()*100:.1f}%)")
print(f"  Spadek:  {(preds == 0).sum()} ({(preds == 0).mean()*100:.1f}%)")

print(f"\nRozkad rzeczywisty:")
print(f"  Wzrost: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"  Spadek:  {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

print(f"\n--- Classification Report ---")
print(classification_report(y, preds, target_names=['Spadek', 'Wzrost']))

print(f"--- Confusion Matrix ---")
cm = confusion_matrix(y, preds)
print(f"                  Pred:Spadek  Pred:Wzrost")
print(f"  Real:Spadek     {cm[0][0]:>10}  {cm[0][1]:>11}")
print(f"  Real:Wzrost     {cm[1][0]:>10}  {cm[1][1]:>11}")

acc = (preds == y).mean()
auc = roc_auc_score(y, probs)
print(f"\nAccuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")

# ============================================================
# OSTATNIE 10 PREDYKCJI
# ============================================================
print(f"\n--- Ostatnie 10 predykcji ---")
print(f"{'Data':<12} {'Prob':>6} {'Pred':>6} {'Real':>6} {'OK?':>5}")
for i in range(-10, 0):
    d = str(dates[i])[:10] if hasattr(dates[i], 'strftime') else str(dates[i])
    p = probs[i]
    pr = "UP" if preds[i] == 1 else "DOWN"
    re = "UP" if y[i] == 1 else "DOWN"
    ok = "v" if preds[i] == y[i] else "x"
    print(f"{d:<12} {p:>6.3f} {pr:>6} {re:>6} {ok:>5}")

# ============================================================
# PREDYKCJA NA JUTRO
# ============================================================
print(f"\n--- Predykcja na nastepny dzien ---")
last_seq = features[-SEQUENCE_LEN:].reshape(1, SEQUENCE_LEN, -1)
tomorrow_prob = model.predict(last_seq, verbose=0)[0][0]
print(f"Ticker:          {TICKER}")
print(f"Prawdopodobienstwo wzrostu: {tomorrow_prob:.4f}")
print(f"Predykcja:       {'WZROST' if tomorrow_prob > 0.5 else 'SPADEK'}")
print(f"Pewnosc:         {abs(tomorrow_prob - 0.5) * 200:.1f}%")