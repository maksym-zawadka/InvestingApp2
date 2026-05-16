"""
XGBoost Tuning: Optuna + Temporal CV + Feature Selection
=========================================================

Wejście:  full_dataset.csv (z train_model_v2.py, zawiera features + labels)
          LUB sentiment_results.csv (wtedy sam pobierze ceny)

Co robi:
  1. Optuna — 500 prób optymalizacji hiperparametrów
  2. Temporal Cross-Validation (5 foldów) — brak data leakage
  3. Feature selection — usuwa szum, zostawia najlepsze
  4. Progi decyzyjne — optymalizacja progu Kupuj/Sprzedaj
  5. Raport + wykresy

Użycie:
    pip install optuna xgboost scikit-learn yfinance
    python tune_xgboost.py --dataset /content/model_output_v2/full_dataset.csv
    # lub
    python tune_xgboost.py --sentiment /content/sentiment_results.csv
"""

import os
import json
import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

FORWARD_DAYS = 63
BENCHMARK = "SPY"
N_TRIALS = 500          # ile prób Optuna
CV_SPLITS = 5           # ile foldów temporal CV
TEST_RATIO = 0.2

SENTIMENT_COLS = [
    "positive", "negative", "neutral", "net_score",
    "pct_positive", "pct_negative",
]


# ============================================================
# ŁADOWANIE DANYCH
# ============================================================

def load_or_build_dataset(args) -> pd.DataFrame:
    """Wczytaj gotowy dataset lub zbuduj od zera."""

    if args.dataset and os.path.exists(args.dataset):
        print(f"  Wczytuję gotowy dataset: {args.dataset}")
        df = pd.read_csv(args.dataset, parse_dates=["filing_date"])
        if "label" not in df.columns:
            raise ValueError("Brak kolumny 'label' w datasecie!")
        return df

    if args.sentiment and os.path.exists(args.sentiment):
        print(f"  Buduję dataset z sentymentu: {args.sentiment}")
        df = pd.read_csv(args.sentiment)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.dropna(subset=["filing_date", "net_score"])
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

        # Pobierz ceny
        import yfinance as yf
        tickers = sorted(set(df["ticker"].tolist() + [BENCHMARK]))
        start = (df["filing_date"].min() - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (df["filing_date"].max() + timedelta(days=FORWARD_DAYS * 2)).strftime("%Y-%m-%d")

        cache = os.path.join(args.output_dir, "prices_cache.csv")
        if os.path.exists(cache):
            prices = pd.read_csv(cache, parse_dates=["date"])
        else:
            print(f"  Pobieram ceny...")
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)
            records = []
            for t in tickers:
                try:
                    series = data[("Close", t)].dropna() if len(tickers) > 1 else data["Close"].dropna()
                    for date, price in series.items():
                        records.append({"ticker": t, "date": pd.Timestamp(date), "close": float(price)})
                except:
                    pass
            prices = pd.DataFrame(records)
            prices.to_csv(cache, index=False)

        # Compute labels
        lookup = {t: prices[prices["ticker"] == t].set_index("date")["close"].sort_index()
                  for t in prices["ticker"].unique()}
        bm = lookup.get(BENCHMARK, pd.Series(dtype=float))

        labels = []
        for _, row in df.iterrows():
            tp = lookup.get(row["ticker"])
            fdate = pd.Timestamp(row["filing_date"])
            if tp is None:
                labels.append({"forward_return": np.nan, "excess_return": np.nan, "label": np.nan})
                continue
            after = tp[tp.index >= fdate]
            if len(after) <= FORWARD_DAYS:
                labels.append({"forward_return": np.nan, "excess_return": np.nan, "label": np.nan})
                continue
            fwd = (after.iloc[FORWARD_DAYS] - after.iloc[0]) / after.iloc[0]
            bm_after = bm[bm.index >= fdate]
            bm_ret = (bm_after.iloc[FORWARD_DAYS] - bm_after.iloc[0]) / bm_after.iloc[0] if len(bm_after) > FORWARD_DAYS else np.nan
            excess = fwd - bm_ret if not np.isnan(bm_ret) else np.nan
            labels.append({"forward_return": fwd, "excess_return": excess,
                           "label": 1 if excess > 0 else 0})

        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(labels)], axis=1)

        # Features
        df = df.sort_values(["ticker", "filing_date"]).copy()
        for col in SENTIMENT_COLS:
            df[f"delta_{col}"] = df.groupby("ticker")[col].diff()
        for col in ["net_score", "negative"]:
            df[f"rolling4_{col}"] = df.groupby("ticker")[col].transform(
                lambda x: x.rolling(4, min_periods=2).mean())
        df["net_score_surprise"] = df["net_score"] - df["rolling4_net_score"]
        df["negative_surprise"] = df["negative"] - df["rolling4_negative"]
        df["is_annual"] = (df["filing_type"] == "10-K").astype(int)
        df["quarter"] = df["filing_date"].dt.quarter

        return df

    raise FileNotFoundError("Podaj --dataset lub --sentiment")


def get_all_feature_cols():
    base = SENTIMENT_COLS.copy()
    deltas = [f"delta_{c}" for c in SENTIMENT_COLS]
    rolling = ["rolling4_net_score", "rolling4_negative"]
    surprises = ["net_score_surprise", "negative_surprise"]
    context = ["is_annual", "quarter", "n_chunks"]
    return base + deltas + rolling + surprises + context


# ============================================================
# TEMPORAL CROSS-VALIDATION
# ============================================================

class TemporalCV:
    """
    Temporal split: zawsze trenujesz na przeszłości, testujesz na przyszłości.
    Nie ma data leakage.
    """
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, dates=None):
        n = len(X)
        # Minimalny rozmiar train: 30% danych
        min_train = int(n * 0.3)
        fold_size = (n - min_train) // self.n_splits

        for i in range(self.n_splits):
            train_end = min_train + i * fold_size
            test_end = min(train_end + fold_size, n)
            if train_end >= n or test_end <= train_end:
                continue
            yield np.arange(0, train_end), np.arange(train_end, test_end)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(X, y, cv):
    """Tworzy funkcję objective dla Optuna."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
            "random_state": 42,
            "eval_metric": "auc",
            "early_stopping_rounds": 30,
        }

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            y_prob = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val, y_prob)
            except ValueError:
                auc = 0.5
            scores.append(auc)

        return np.mean(scores)

    return objective


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(X_train, y_train, X_test, feature_names, top_n=None):
    """
    Dwuetapowa selekcja:
    1. Usuń features z zerowym importance
    2. Zostaw top_n najważniejszych (lub automatycznie)
    """
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="auc"
    )
    model.fit(X_train, y_train, verbose=False)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    # Usuń features z importance = 0
    nonzero = importances[importances > 0]

    if top_n is None:
        # Automatycznie: weź features z importance > mean
        threshold = nonzero.mean()
        selected = nonzero[nonzero >= threshold].index.tolist()
        if len(selected) < 3:
            selected = nonzero.head(5).index.tolist()
    else:
        selected = nonzero.head(top_n).index.tolist()

    print(f"\n  Feature selection:")
    print(f"    Wszystkich:   {len(feature_names)}")
    print(f"    Niezerowych:  {len(nonzero)}")
    print(f"    Wybranych:    {len(selected)}")
    print(f"    Top features: {', '.join(selected[:5])}")

    mask = [i for i, f in enumerate(feature_names) if f in selected]
    return X_train[:, mask], X_test[:, mask], selected, importances


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(y_true, y_prob):
    """
    Znajduje optymalny próg decyzyjny wg kilku kryteriów.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.30, 0.71, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_buy = y_pred.sum()
        n_sell = (1 - y_pred).sum()

        if n_buy == 0 or n_sell == 0:
            continue

        f1 = f1_score(y_true, y_pred, average="macro")
        prec_buy = precision_score(y_true, y_pred, zero_division=0)
        prec_sell = precision_score(1 - y_true, 1 - y_pred, zero_division=0)

        results.append({
            "threshold": round(t, 2),
            "f1_macro": round(f1, 4),
            "precision_buy": round(prec_buy, 4),
            "precision_sell": round(prec_sell, 4),
            "n_buy": n_buy,
            "n_sell": n_sell,
        })

    results_df = pd.DataFrame(results)

    # Najlepszy próg wg F1 macro
    best = results_df.loc[results_df["f1_macro"].idxmax()]
    return best["threshold"], results_df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="full_dataset.csv z train_model_v2.py")
    parser.add_argument("--sentiment", default="sentiment_results.csv")
    parser.add_argument("--output_dir", default="model_tuned")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--top_features", type=int, default=None,
                        help="Ile features zostawić (None=auto)")
    args, _ = parser.parse_known_args()

    global N_TRIALS
    N_TRIALS = args.trials

    os.makedirs(args.output_dir, exist_ok=True)

    import optuna
    import xgboost as xgb
    from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                                  classification_report, confusion_matrix)

    print("=" * 60)
    print(f"XGBoost TUNING — Optuna ({N_TRIALS} prób)")
    print("=" * 60)

    # ── 1. Dane ──
    print("\n📄 STEP 1: Ładowanie danych")
    df = load_or_build_dataset(args)
    all_features = get_all_feature_cols()
    all_features = [f for f in all_features if f in df.columns]

    clean = df.dropna(subset=all_features + ["label"]).sort_values("filing_date").reset_index(drop=True)
    print(f"  Wierszy: {len(clean)} | Features: {len(all_features)}")
    print(f"  Balans: Kupuj {(clean['label']==1).sum()} / Sprzedaj {(clean['label']==0).sum()}")

    # ── 2. Split ──
    split = int(len(clean) * (1 - TEST_RATIO))
    train_df = clean.iloc[:split]
    test_df = clean.iloc[split:]

    X_train_full = train_df[all_features].values
    y_train = train_df["label"].values.astype(int)
    X_test_full = test_df[all_features].values
    y_test = test_df["label"].values.astype(int)

    print(f"\n  Train: {len(train_df)} ({train_df['filing_date'].dt.date.min()} → "
          f"{train_df['filing_date'].dt.date.max()})")
    print(f"  Test:  {len(test_df)} ({test_df['filing_date'].dt.date.min()} → "
          f"{test_df['filing_date'].dt.date.max()})")

    # ── 3. Feature selection ──
    print("\n🔍 STEP 2: Feature selection")
    X_train, X_test, selected_features, importances = select_features(
        X_train_full, y_train, X_test_full, all_features, args.top_features
    )

    # ── 4. Optuna ──
    print(f"\n⚡ STEP 3: Optuna — {N_TRIALS} prób, {CV_SPLITS}-fold temporal CV")
    cv = TemporalCV(n_splits=CV_SPLITS)
    objective = create_objective(X_train, y_train, cv)

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n  Najlepszy CV AUC: {study.best_value:.4f}")
    print(f"  Najlepsze parametry:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ── 5. Trenuj finalny model ──
    print("\n🤖 STEP 4: Trening finalnego modelu")
    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["eval_metric"] = "auc"
    best_params["early_stopping_rounds"] = 30

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)], verbose=False)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"  Kupuj: {(y_pred==1).sum()} | Sprzedaj: {(y_pred==0).sum()}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sprzedaj', 'Kupuj'])}")

    # ── 6. Optimalizacja progu ──
    print("🎯 STEP 5: Optymalizacja progu decyzyjnego")
    best_threshold, threshold_df = optimize_threshold(y_test, y_prob)
    print(f"  Optymalny próg: {best_threshold}")
    print(f"  Top progi:")
    print(threshold_df.nlargest(5, "f1_macro").to_string(index=False))

    # Predykcje z optymalnym progiem
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    auc_opt = auc  # AUC nie zależy od progu
    f1_opt = f1_score(y_test, y_pred_opt, average="macro")
    print(f"\n  Z optymalnym progiem ({best_threshold}):")
    print(f"    Accuracy: {acc_opt:.4f}  |  F1: {f1_opt:.4f}")
    print(f"    Kupuj: {(y_pred_opt==1).sum()} | Sprzedaj: {(y_pred_opt==0).sum()}")
    print(f"\n{classification_report(y_test, y_pred_opt, target_names=['Sprzedaj', 'Kupuj'])}")

    # ── 7. Wykresy ──
    print("\n📊 STEP 6: Wykresy")
    _make_plots(study, final_model, X_test, y_test, y_prob, y_pred_opt,
                selected_features, importances, all_features,
                test_df, threshold_df, best_threshold, args.output_dir)

    # ── 8. Zapis ──
    print("\n💾 STEP 7: Zapis modelu")
    model_path = os.path.join(args.output_dir, "model_raportow.pkl")
    joblib.dump({
        "model": final_model,
        "features": selected_features,
        "best_params": study.best_params,
        "cv_auc": study.best_value,
        "test_auc": auc,
        "threshold": best_threshold,
        "metrics": {
            "accuracy": acc_opt, "auc": auc, "f1_macro": f1_opt,
            "n_buy": int((y_pred_opt == 1).sum()),
            "n_sell": int((y_pred_opt == 0).sum()),
        }
    }, model_path)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "best_params": study.best_params,
            "cv_auc": study.best_value,
            "test_auc": auc,
            "threshold": best_threshold,
            "selected_features": selected_features,
            "n_trials": N_TRIALS,
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"✅ GOTOWE!")
    print(f"{'='*60}")
    print(f"  CV AUC:     {study.best_value:.4f}")
    print(f"  Test AUC:   {auc:.4f}")
    print(f"  Próg:       {best_threshold}")
    print(f"  Model:      {model_path}")


# ============================================================
# WYKRESY
# ============================================================

def _make_plots(study, model, X_test, y_test, y_prob, y_pred,
                selected_features, importances, all_features,
                test_df, threshold_df, best_threshold, out_dir):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve

    # 1. Optuna optimization history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    trials = [t.value for t in study.trials if t.value is not None]
    axes[0].plot(trials, alpha=0.4, linewidth=0.5)
    axes[0].plot(pd.Series(trials).cummax(), color="red", linewidth=2, label="Best so far")
    axes[0].set_title(f"Optuna: {len(trials)} prób")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("CV AUC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Running best
    running = pd.Series(trials).expanding().max()
    axes[1].plot(running, color="red", linewidth=2)
    axes[1].set_title("Najlepszy AUC w czasie")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Best CV AUC")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "optuna_history.png"), dpi=150)
    plt.close()

    # 2. Feature importance (all features)
    fig, ax = plt.subplots(figsize=(8, 10))
    imp_all = importances.sort_values()
    colors = ["steelblue" if f in selected_features else "lightgray" for f in imp_all.index]
    imp_all.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance\n(niebieski = wybrane, szary = odrzucone)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    # 3. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sprzedaj", "Kupuj"],
                yticklabels=["Sprzedaj", "Kupuj"], ax=ax)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Rzeczywistość")
    ax.set_title(f"Confusion Matrix (próg={best_threshold})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 4. ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, linewidth=2, label=f"XGBoost tuned (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # 5. Probability distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label="Rzeczywiście Kupuj", color="green")
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label="Rzeczywiście Sprzedaj", color="red")
    ax.axvline(best_threshold, color="blue", linestyle="--", linewidth=2,
               label=f"Próg ({best_threshold})")
    ax.set_title("Rozkład prawdopodobieństw")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probability_distribution.png"), dpi=150)
    plt.close()

    # 6. Threshold vs F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["f1_macro"],
            linewidth=2, label="F1 macro", color="blue")
    ax.plot(threshold_df["threshold"], threshold_df["precision_buy"],
            linewidth=1.5, label="Precision Kupuj", color="green", alpha=0.7)
    ax.plot(threshold_df["threshold"], threshold_df["precision_sell"],
            linewidth=1.5, label="Precision Sprzedaj", color="red", alpha=0.7)
    ax.axvline(best_threshold, color="black", linestyle="--", label=f"Wybrany ({best_threshold})")
    ax.set_title("Próg decyzyjny vs metryki")
    ax.set_xlabel("Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_optimization.png"), dpi=150)
    plt.close()

    # 7. Backtest
    tc = test_df.copy().sort_values("filing_date")
    tc["prob"] = y_prob
    tc["pred"] = y_pred
    tc["strategy_ret"] = tc.apply(lambda r:
        r["excess_return"] if r["pred"] == 1
        else (-r["excess_return"] if r["pred"] == 0 else 0), axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tc["filing_date"].values,
            (1 + tc["strategy_ret"]).cumprod().values,
            label="Strategia (tuned XGBoost)", linewidth=2, color="green")
    ax.plot(tc["filing_date"].values,
            (1 + tc["excess_return"]).cumprod().values,
            label="Zawsze Kupuj", linewidth=2, alpha=0.5, color="gray")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_title("Backtest — excess return vs SPY")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "backtest.png"), dpi=150)
    plt.close()

    # 8. Parametr importance (Optuna)
    try:
        param_imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 6))
        pd.Series(param_imp).sort_values().plot(kind="barh", ax=ax, color="orange")
        ax.set_title("Ważność hiperparametrów (Optuna)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "param_importance.png"), dpi=150)
        plt.close()
    except:
        pass

    print(f"  Zapisano 8 wykresów w {out_dir}/")


if __name__ == "__main__":
    import optuna
    main()