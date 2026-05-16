"""
models/xgboost_model.py
───────────────────────
Integracja modelu XGBoost z live pipeline SEC EDGAR + FinBERT.

Plik wymagany w folderze models/:
    model_raportow.pkl  – słownik z kluczami:
        "model"     – XGBClassifier
        "features"  – lista wybranych kolumn po feature selection
        "threshold" – optymalny próg decyzyjny
        "metrics"   – słownik metryk (opcjonalnie)
"""

import os
import joblib
import numpy as np
import pandas as pd


def load():
    base       = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "model_raportow.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak pliku: {model_path}")

    bundle = joblib.load(model_path)
    for key in ("model", "features", "threshold"):
        if key not in bundle:
            raise KeyError(f"Plik pkl nie zawiera klucza '{key}'. "
                           f"Dostępne: {list(bundle.keys())}")

    return XGBoostModel(
        model     = bundle["model"],
        features  = bundle["features"],
        threshold = float(bundle["threshold"]),
        metrics   = bundle.get("metrics", {}),
    )


class XGBoostModel:
    def __init__(self, model, features, threshold, metrics):
        self.model     = model
        self.features  = features
        self.threshold = threshold
        self.metrics   = metrics

    def predict(self, df: pd.DataFrame, ticker: str,
                status_callback=None) -> tuple:
        """
        1. Pobiera najnowszy raport z SEC EDGAR
        2. Przepuszcza przez FinBERT
        3. Buduje wektor cech → XGBoost
        4. Zwraca (predicted_price, prob)
        """
        from models.sec_finbert import fetch_sentiment

        # Live pipeline — może trwać 30–120s przy pierwszym uruchomieniu
        sentiment = fetch_sentiment(ticker, status_callback=status_callback)

        features_vec = self._build_features(sentiment)
        prob         = float(self.model.predict_proba(features_vec)[0][1])
        kupuj        = prob >= self.threshold

        # Orientacyjna cena za ~63 sesje
        last_close   = float(df["Close"].iloc[-1])
        ret_63       = float(df["Close"].pct_change().tail(63).sum())
        direction    = 1 if kupuj else -1
        confidence   = abs(prob - self.threshold) / max(self.threshold, 1 - self.threshold)
        pred_price   = round(last_close * (1 + direction * confidence * abs(ret_63)), 2)

        return pred_price, prob, sentiment   # zwracamy też sentiment do wyświetlenia

    def _build_features(self, sentiment: dict) -> np.ndarray:
        """Buduje wektor cech zgodny z treningiem (brak historii → delta/rolling = 0)."""
        SENTIMENT_COLS = [
            "positive", "negative", "neutral", "net_score",
            "pct_positive", "pct_negative",
        ]
        feat = {}

        for col in SENTIMENT_COLS:
            feat[col] = float(sentiment.get(col, 0.0))

        for col in SENTIMENT_COLS:
            feat[f"delta_{col}"] = 0.0

        feat["rolling4_net_score"]  = feat["net_score"]
        feat["rolling4_negative"]   = feat["negative"]
        feat["net_score_surprise"]  = 0.0
        feat["negative_surprise"]   = 0.0

        filing_type     = str(sentiment.get("filing_type", "10-Q"))
        feat["is_annual"] = 1 if filing_type == "10-K" else 0

        try:
            feat["quarter"] = pd.to_datetime(sentiment.get("filing_date")).quarter
        except Exception:
            feat["quarter"] = 0

        feat["n_chunks"] = float(sentiment.get("n_chunks", 0))

        vec = np.array([[feat.get(f, 0.0) for f in self.features]], dtype=np.float32)
        return vec
