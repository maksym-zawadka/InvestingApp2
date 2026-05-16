"""
models/sec_finbert.py
─────────────────────
Live pipeline: SEC EDGAR → tekst raportu → FinBERT → wektor sentymentu

Działa dla spółek notowanych na giełdach USA (10-K / 10-Q z SEC EDGAR).
Dla spółek spoza USA (np. .WA, .DE) zgłasza czytelny wyjątek.

Przepływ:
  1. SEC EDGAR Company Search API  → CIK dla tickera
  2. SEC EDGAR Submissions API     → najnowsze zgłoszenie 10-K lub 10-Q
  3. SEC EDGAR Archives            → pełny tekst dokumentu HTML
  4. BeautifulSoup                 → wyodrębnij sekcję MD&A
  5. Chunking (400 słów, overlap 50)
  6. FinBERT (ProsusAI/finbert)    → sentyment per chunk → agregacja
"""

from __future__ import annotations

import gc
import re
import time
import requests
import numpy as np
from bs4 import BeautifulSoup

# ── Stałe (identyczne jak w sentiment.py) ────────────────────────────────────
CHUNK_MAX_WORDS   = 400
CHUNK_OVERLAP_WORDS = 50
BATCH_SIZE        = 16      # małe batche — CPU, brak GPU
FINBERT_MODEL     = "ProsusAI/finbert"

MDA_PATTERNS = [
    r"management.{0,5}s?\s*discussion\s*and\s*analysis",
    r"item\s*7[.\s]",
    r"item\s*2[.\s]",
]

SEC_HEADERS = {
    "User-Agent": "StockPulse-App contact@example.com",   # SEC wymaga User-Agent
    "Accept-Encoding": "gzip, deflate",
}


# ── 1. CIK ───────────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> str:
    """Pobiera CIK (SEC identifier) dla tickera. Rzuca ValueError jeśli nie znaleziono."""
    ticker_clean = ticker.upper().split(".")[0]   # CDR.WA → CDR, AAPL → AAPL

    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker_clean}%22&dateRange=custom&startdt=2000-01-01&enddt=2100-01-01&forms=10-K"
    # Prostsze: company tickers JSON
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_clean:
            cik = str(entry["cik_str"]).zfill(10)
            return cik

    raise ValueError(
        f"Nie znaleziono spółki '{ticker_clean}' w bazie SEC EDGAR.\n"
        "Model XGBoost działa tylko dla spółek notowanych w USA (NYSE/NASDAQ).\n"
        "Dla spółek GPW (.WA) użyj modelu LSTM."
    )


# ── 2. Najnowsze zgłoszenie ───────────────────────────────────────────────────

def get_latest_filing(cik: str, form_types: tuple = ("10-Q", "10-K")) -> dict:
    """Zwraca metadane najnowszego zgłoszenia 10-Q lub 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r   = requests.get(url, headers=SEC_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    accnums = filings.get("accessionNumber", [])
    docs    = filings.get("primaryDocument", [])

    # Szukaj najnowszego 10-Q, fallback 10-K
    for form_type in form_types:
        for i, form in enumerate(forms):
            if form == form_type:
                return {
                    "form_type":  form,
                    "filing_date": dates[i],
                    "accession":  accnums[i].replace("-", ""),
                    "primary_doc": docs[i],
                    "cik":        cik,
                }

    raise ValueError(f"Brak zgłoszeń {form_types} dla CIK={cik}")


# ── 3. Pobieranie tekstu ──────────────────────────────────────────────────────

def fetch_filing_text(filing: dict) -> str:
    """Pobiera HTML dokumentu z SEC EDGAR Archives i zwraca czysty tekst."""
    cik     = filing["cik"]
    acc     = filing["accession"]           # bez myślników
    doc     = filing["primary_doc"]

    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    r   = requests.get(url, headers=SEC_HEADERS, timeout=30)

    if r.status_code != 200:
        # Spróbuj indeks zgłoszenia i weź pierwszy .htm
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{acc}-index.htm"
        r2 = requests.get(idx_url, headers=SEC_HEADERS, timeout=15)
        soup2 = BeautifulSoup(r2.text, "lxml")
        links = [a["href"] for a in soup2.find_all("a", href=True)
                 if a["href"].lower().endswith((".htm", ".html"))
                 and "ex" not in a["href"].lower()]
        if not links:
            raise ValueError(f"Nie można pobrać dokumentu z SEC: {url}")
        url = "https://www.sec.gov" + links[0]
        r   = requests.get(url, headers=SEC_HEADERS, timeout=30)

    r.raise_for_status()

    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    del soup
    return text


# ── 4. Wyodrębnienie MD&A ────────────────────────────────────────────────────

def extract_mda(full_text: str) -> tuple[str, str]:
    """Wyodrębnia sekcję MD&A. Zwraca (tekst, nazwa_sekcji)."""
    text_lower = full_text.lower()

    for pattern in MDA_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            start = match.start()
            remaining_lower = text_lower[start + 100:]
            end_match = re.search(r"\bitem\s*\d+[a-z]?[.\s]", remaining_lower)
            end = start + 100 + end_match.start() if end_match else min(start + 100_000, len(full_text))
            section = full_text[start:end].strip()
            if len(section) >= 200:
                return section, "mda"

    # Fallback: środkowa część dokumentu
    mid = len(full_text) // 4
    return full_text[mid:mid + 80_000], "full_fallback"


# ── 5. Chunking ───────────────────────────────────────────────────────────────

def make_chunks(text: str) -> list[str]:
    """Dzieli tekst na chunki (identycznie jak w sentiment.py)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= CHUNK_MAX_WORDS:
        return [text]
    chunks = []
    s = 0
    while s < len(words):
        e = s + CHUNK_MAX_WORDS
        chunks.append(" ".join(words[s:e]))
        s = e - CHUNK_OVERLAP_WORDS
    return chunks


# ── 6. FinBERT ────────────────────────────────────────────────────────────────

_finbert_cache: dict = {}   # model i tokenizer ładowane raz na sesję


def load_finbert():
    """Ładuje FinBERT (raz, potem z cache)."""
    if "model" not in _finbert_cache:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()

        _finbert_cache["model"]     = model
        _finbert_cache["tokenizer"] = tokenizer
        _finbert_cache["device"]    = device

    return (_finbert_cache["model"],
            _finbert_cache["tokenizer"],
            _finbert_cache["device"])


def run_finbert(chunks: list[str]) -> np.ndarray:
    """
    Przepuszcza chunki przez FinBERT.
    Zwraca array (n, 3): [positive, negative, neutral].
    """
    import torch

    model, tokenizer, device = load_finbert()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch   = chunks[i:i + BATCH_SIZE]
            inputs  = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(device)
            logits  = model(**inputs).logits
            probs   = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            del inputs, logits

    return np.vstack(all_probs)


def aggregate_sentiment(probs: np.ndarray) -> dict:
    """Agreguje wyniki per chunk → jeden wektor sentymentu (jak w sentiment.py)."""
    pos    = probs[:, 0]
    neg    = probs[:, 1]
    labels = np.argmax(probs, axis=1)
    return {
        "positive":     round(float(pos.mean()), 6),
        "negative":     round(float(neg.mean()), 6),
        "neutral":      round(float(probs[:, 2].mean()), 6),
        "net_score":    round(float(pos.mean() - neg.mean()), 6),
        "pct_positive": round(float((labels == 0).mean()), 6),
        "pct_negative": round(float((labels == 1).mean()), 6),
        "n_chunks":     len(probs),
    }


# ── Publiczny interfejs ───────────────────────────────────────────────────────

def fetch_sentiment(ticker: str, status_callback=None) -> dict:
    """
    Główna funkcja — zwraca słownik sentymentu dla najnowszego raportu spółki.

    status_callback: opcjonalna funkcja(str) do raportowania postępu
                     (np. st.status lub print)

    Zwraca:
        {
          "ticker": str,
          "filing_type": str,       # "10-Q" lub "10-K"
          "filing_date": str,       # "2024-11-05"
          "section_used": str,      # "mda" lub "full_fallback"
          "positive": float,
          "negative": float,
          "neutral": float,
          "net_score": float,       # positive - negative
          "pct_positive": float,
          "pct_negative": float,
          "n_chunks": int,
        }
    """
    def log(msg: str):
        if status_callback:
            status_callback(msg)

    log("🔍 Szukam CIK w bazie SEC EDGAR…")
    cik = get_cik(ticker)

    log("📄 Pobieram listę zgłoszeń…")
    filing = get_latest_filing(cik)
    log(f"📑 Najnowszy raport: {filing['form_type']} z {filing['filing_date']}")

    log("⬇️  Pobieram dokument z SEC EDGAR…")
    time.sleep(0.15)   # SEC rate limit: max ~10 req/s
    full_text = fetch_filing_text(filing)
    log(f"✅ Pobrano dokument ({len(full_text):,} znaków)")

    log("✂️  Wyodrębniam sekcję MD&A…")
    mda_text, section_used = extract_mda(full_text)
    del full_text
    gc.collect()
    log(f"📝 Sekcja '{section_used}': {len(mda_text):,} znaków → chunking…")

    chunks = make_chunks(mda_text)
    del mda_text
    log(f"🧩 {len(chunks)} chunków → FinBERT…")

    if not chunks:
        raise ValueError("Nie udało się wyodrębnić tekstu z raportu.")

    log("🤖 Ładuję FinBERT (pierwsze uruchomienie może potrwać ~30s)…")
    probs = run_finbert(chunks)
    gc.collect()

    sentiment = aggregate_sentiment(probs)
    log(f"✅ Sentyment: net_score={sentiment['net_score']:+.3f}, "
        f"positive={sentiment['pct_positive']:.1%}, "
        f"negative={sentiment['pct_negative']:.1%}")

    return {
        "ticker":       ticker,
        "filing_type":  filing["form_type"],
        "filing_date":  filing["filing_date"],
        "section_used": section_used,
        **sentiment,
    }
