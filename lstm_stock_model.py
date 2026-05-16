"""
LSTM Stock Price Movement Predictor
Trenowanie na Google Colab z GPU, dane w wielu plikach CSV.
Wynik: model.keras
"""

import os
import gc
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/content/data"          # folder z plikami CSV
MODEL_PATH = "/content/model.keras"
SCALER_PATH = "/content/scaler.pkl"
SEQUENCE_LEN = 30                   # okno czasowe
BATCH_SIZE = 1024
EPOCHS = 50
VALIDATION_SPLIT_RATIO = 0.15
LEARNING_RATE = 1e-3
CHUNK_SIZE = 30
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_LSTM = 0.3
DROPOUT_DENSE = 0.2
EARLY_STOP_PATIENCE = 8
LR_REDUCE_PATIENCE = 4
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6
SCALER_MAX_SAMPLES = 500_000        # max próbek do fitowania scalera
CLASS_WEIGHT_BOOST = 1.1            # mnożnik wagi klasy 0 (spadek) - wyższy = mniej fałszywych "wzrostów"

# ============================================================
# GPU SETUP
# ============================================================
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU dostępne: {[g.name for g in gpus]}")
else:
    print("BRAK GPU - trening będzie wolny!")

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"Mixed precision: compute={policy.compute_dtype}, variable={policy.variable_dtype}")


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=['OpenInt'], errors='ignore', inplace=True)

    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume'].astype(float)

    # SMA
    df['SMA_10'] = c.rolling(10).mean()
    df['SMA_20'] = c.rolling(20).mean()

    # EMA
    df['EMA_12'] = c.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = c.ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility (20-day rolling std of returns)
    returns = c.pct_change()
    df['Volatility'] = returns.rolling(20).std()

    # Price changes
    df['Price_Change'] = returns
    df['Price_Change_5'] = c.pct_change(5)

    # Volume indicators
    df['Volume_SMA_10'] = v.rolling(10).mean()
    df['Volume_Ratio'] = v / (df['Volume_SMA_10'] + 1e-10)
    df['OBV'] = (np.sign(c.diff()) * v).cumsum()

    # High-Low range
    df['HL_Range'] = (h - l) / (c + 1e-10)

    # Close vs SMA
    df['Close_SMA10_Ratio'] = c / (df['SMA_10'] + 1e-10)
    df['Close_SMA20_Ratio'] = c / (df['SMA_20'] + 1e-10)

    # Target: 1 = cena wzrośnie jutro, 0 = spadnie
    df['Target'] = (c.shift(-1) > c).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# FEATURE COLUMNS
# ============================================================
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI', 'Volatility', 'Price_Change', 'Price_Change_5',
    'Volume_SMA_10', 'Volume_Ratio', 'OBV', 'HL_Range',
    'Close_SMA10_Ratio', 'Close_SMA20_Ratio'
]


# ============================================================
# LOAD & PROCESS ALL FILES -> FIT SCALER, COLLECT STATS
# ============================================================
def get_csv_files():
    files = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.lower().endswith('.txt')
    ])
    print(f"Znaleziono {len(files)} plików TXT")
    return files


def process_file(filepath, scaler=None, fit_scaler=False):
    """Wczytaj plik, dodaj wskaźniki, skaluj, podziel na sekwencje."""
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = add_indicators(df)
    if len(df) < SEQUENCE_LEN + 10:
        return None, None, None, None, scaler

    features = df[FEATURE_COLS].values
    targets = df['Target'].values

    if fit_scaler and scaler is None:
        scaler = RobustScaler()
        scaler.fit(features)
    features = scaler.transform(features)

    # Chronologiczny podział: train | val
    split_idx = int(len(features) * (1 - VALIDATION_SPLIT_RATIO))

    def make_sequences(data, labels, start, end):
        X, y = [], []
        for i in range(start, end - SEQUENCE_LEN):
            X.append(data[i:i + SEQUENCE_LEN])
            y.append(labels[i + SEQUENCE_LEN])
        if len(X) == 0:
            return None, None
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    X_train, y_train = make_sequences(features, targets, 0, split_idx)
    X_val, y_val = make_sequences(features, targets, split_idx, len(features))

    return X_train, y_train, X_val, y_val, scaler


# ============================================================
# FAZA 1: Fitowanie scalera na wszystkich danych (partiami)
# ============================================================
print("\n=== FAZA 1: Fitowanie scalera ===")
csv_files = get_csv_files()

# Zbieramy statystyki inkrementalnie zamiast ładować wszystko do RAM
scaler_samples = []

for fpath in csv_files:
    df = pd.read_csv(fpath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_indicators(df)
    if len(df) < SEQUENCE_LEN + 10:
        continue
    scaler_samples.append(df[FEATURE_COLS].values)
    total = sum(len(s) for s in scaler_samples)
    if total >= SCALER_MAX_SAMPLES:
        break

scaler_data = np.concatenate(scaler_samples, axis=0)
scaler = RobustScaler()
scaler.fit(scaler_data)
print(f"Scaler dopasowany na {len(scaler_data)} próbkach")
del scaler_samples, scaler_data
gc.collect()

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler zapisany: {SCALER_PATH}")


# ============================================================
# FAZA 2: Budowanie zbiorów partiami
# ============================================================
print("\n=== FAZA 2: Tworzenie sekwencji (partiami) ===")

TEMP_DIR = "/content/temp_seq"
os.makedirs(TEMP_DIR, exist_ok=True)

CHUNK = CHUNK_SIZE
train_count = 0
val_count = 0
part_idx = 0

for i in range(0, len(csv_files), CHUNK):
    chunk_files = csv_files[i:i + CHUNK]
    print(f"  Przetwarzam pliki {i+1}-{i+len(chunk_files)} / {len(csv_files)}")

    chunk_train_X, chunk_train_y = [], []
    chunk_val_X, chunk_val_y = [], []

    for fpath in chunk_files:
        X_tr, y_tr, X_v, y_v, _ = process_file(fpath, scaler=scaler, fit_scaler=False)
        if X_tr is not None:
            chunk_train_X.append(X_tr)
            chunk_train_y.append(y_tr)
        if X_v is not None:
            chunk_val_X.append(X_v)
            chunk_val_y.append(y_v)

    if chunk_train_X:
        X_tr_chunk = np.concatenate(chunk_train_X, axis=0)
        y_tr_chunk = np.concatenate(chunk_train_y, axis=0)
        np.save(os.path.join(TEMP_DIR, f"train_X_{part_idx}.npy"), X_tr_chunk)
        np.save(os.path.join(TEMP_DIR, f"train_y_{part_idx}.npy"), y_tr_chunk)
        train_count += len(X_tr_chunk)
        del X_tr_chunk, y_tr_chunk

    if chunk_val_X:
        X_v_chunk = np.concatenate(chunk_val_X, axis=0)
        y_v_chunk = np.concatenate(chunk_val_y, axis=0)
        np.save(os.path.join(TEMP_DIR, f"val_X_{part_idx}.npy"), X_v_chunk)
        np.save(os.path.join(TEMP_DIR, f"val_y_{part_idx}.npy"), y_v_chunk)
        val_count += len(X_v_chunk)
        del X_v_chunk, y_v_chunk

    del chunk_train_X, chunk_train_y, chunk_val_X, chunk_val_y
    gc.collect()
    part_idx += 1

print(f"  Zapisano {part_idx} partii na dysk. Train: ~{train_count}, Val: ~{val_count}")

# ============================================================
# CLASS WEIGHTS (liczone z małej próbki, nie ładujemy wszystkiego)
# ============================================================
print("  Obliczam class weights z próbki...")
sample_y = []
for f in sorted(os.listdir(TEMP_DIR)):
    if f.startswith("train_y_"):
        y_part = np.load(os.path.join(TEMP_DIR, f))
        sample_y.append(y_part)
sample_y = np.concatenate(sample_y, axis=0)
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=sample_y)
class_weight_dict = {0: class_weights[0] * CLASS_WEIGHT_BOOST, 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")
print(f"Train target dist: {np.bincount(sample_y)}")
del sample_y
gc.collect()

# Val target dist
val_y_all = np.concatenate([
    np.load(os.path.join(TEMP_DIR, f))
    for f in sorted(os.listdir(TEMP_DIR)) if f.startswith("val_y_")
], axis=0)
print(f"Val target dist:   {np.bincount(val_y_all)}")
del val_y_all
gc.collect()


# ============================================================
# TF DATASET - batch generator (1 chunk w RAM naraz)
# ============================================================
N_FEATURES = len(FEATURE_COLS)

def get_npy_files(prefix):
    x_files = sorted([os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"{prefix}_X_")])
    y_files = sorted([os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"{prefix}_y_")])
    return x_files, y_files

def batch_generator(prefix, shuffle=False):
    """Generator yieldujący gotowe batche z plików .npy - 1 chunk w RAM naraz."""
    x_files, y_files = get_npy_files(prefix)
    def gen():
        indices = list(range(len(x_files)))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            X = np.load(x_files[idx])
            y = np.load(y_files[idx])
            if shuffle:
                perm = np.random.permutation(len(X))
                X, y = X[perm], y[perm]
            for start in range(0, len(X) - BATCH_SIZE + 1, BATCH_SIZE):
                yield X[start:start + BATCH_SIZE], y[start:start + BATCH_SIZE]
    return gen

batch_output_sig = (
    tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LEN, N_FEATURES), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
)

print("  Tworzę tf.data pipeline...")
train_ds = tf.data.Dataset.from_generator(batch_generator("train", shuffle=True), output_signature=batch_output_sig)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(batch_generator("val", shuffle=False), output_signature=batch_output_sig)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

STEPS_PER_EPOCH = train_count // BATCH_SIZE
VAL_STEPS = val_count // BATCH_SIZE
print(f"  Train: ~{train_count} samples ({STEPS_PER_EPOCH} steps), Val: ~{val_count} samples ({VAL_STEPS} steps)")

# ============================================================
# MODEL
# ============================================================
print("\n=== Budowanie modelu LSTM ===")

inputs = tf.keras.Input(shape=(SEQUENCE_LEN, N_FEATURES))

x = tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True,
                          recurrent_dropout=0.0)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(DROPOUT_LSTM)(x)

x = tf.keras.layers.LSTM(LSTM_UNITS_2, return_sequences=False,
                          recurrent_dropout=0.0)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(DROPOUT_LSTM)(x)

x = tf.keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
x = tf.keras.layers.Dropout(DROPOUT_DENSE)(x)

# float32 output dla mixed precision
outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=EARLY_STOP_PATIENCE, mode='max',
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LR, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    ),
]


# ============================================================
# TRENING
# ============================================================
print("\n=== START TRENINGU ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# FINAL EVAL
# ============================================================
print("\n=== Ewaluacja najlepszego modelu ===")
best_model = tf.keras.models.load_model(MODEL_PATH)
results = best_model.evaluate(val_ds, steps=VAL_STEPS, verbose=1)
metric_names = best_model.metrics_names
for name, val in zip(metric_names, results):
    print(f"  {name}: {val:.4f}")

# Cleanup temp files
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"\nModel zapisany: {MODEL_PATH}")
print(f"Scaler zapisany: {SCALER_PATH}")
print("Gotowe!")