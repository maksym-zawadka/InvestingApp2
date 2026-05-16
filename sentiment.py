"""
Ekstrakcja sentymentu z raportów SEC — wersja Colab (GPU)
==========================================================

Zoptymalizowane pod Google Colab (T4 GPU, 10GB VRAM, 12GB RAM):
  - Model FinBERT ładowany raz na GPU (~440MB VRAM)
  - Raporty przetwarzane PO JEDNYM — tekst zwalniany po ekstrakcji
  - Wyniki dopisywane do CSV przyrostowo (nie gubi się przy crashu)
  - Można wznowić po przerwaniu (pomija już przetworzone)

Użycie w Colab:
    # Komórka 1: instalacja
    !pip install -q transformers beautifulsoup4 lxml tqdm

    # Komórka 2: upload lub mount Google Drive z raportami
    from google.colab import drive
    drive.mount('/content/drive')

    # Komórka 3: uruchom
    !python extract_sentiment_colab.py \\
        --filings_dir /content/drive/MyDrive/sec_filings \\
        --output /content/drive/MyDrive/sentiment_results.csv
"""

import os
import re
import gc
import glob
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# KONFIGURACJA
# ============================================================

CHUNK_MAX_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
BATCH_SIZE = 128            # duży batch na GPU
PARSE_WORKERS = 4           # ile procesów do parsowania HTML
FILES_PER_GPU_BATCH = 10    # ile plików zebrać przed wysłaniem na GPU

MDA_PATTERNS = [
    r"management.{0,5}s?\s*discussion\s*and\s*analysis",
    r"item\s*7[.\s]",
    r"item\s*2[.\s]",
]

# Kolumny CSV wynikowego
CSV_COLUMNS = [
    "ticker", "filing_type", "filing_date", "section_used",
    "text_length", "positive", "negative", "neutral",
    "net_score", "pct_positive", "pct_negative", "n_chunks",
]


# ============================================================
# PARSOWANIE + CHUNKING (w osobnych procesach CPU)
# ============================================================


def find_filings(filings_dir: str) -> list[dict]:
    """Skanuje folder i zwraca listę sprawozdań."""
    records = []
    for ticker_dir in sorted(glob.glob(os.path.join(filings_dir, "*"))):
        if not os.path.isdir(ticker_dir):
            continue
        ticker = os.path.basename(ticker_dir)
        for type_dir in glob.glob(os.path.join(ticker_dir, "*")):
            if not os.path.isdir(type_dir):
                continue
            ftype = "10-K" if "10K" in os.path.basename(type_dir).upper() else "10-Q"
            for fpath in sorted(glob.glob(os.path.join(type_dir, "*"))):
                ext = os.path.splitext(fpath)[1].lower()
                if ext not in (".htm", ".html", ".txt"):
                    continue
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(fpath))
                filing_date = date_match.group(1) if date_match else "unknown"
                records.append({
                    "ticker": ticker, "filing_type": ftype,
                    "filing_date": filing_date, "filepath": fpath,
                })
    return records


def parse_single_filing(filing: dict, section: str = "mda") -> dict:
    """
    Parsuje jeden plik HTML → tekst → chunki.
    Działa w osobnym procesie (nie dotyka GPU).
    """
    try:
        filepath = filing["filepath"]
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return {**filing, "chunks": [], "section_used": "error", "text_length": 0}

        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        full_text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        del soup, content

        # Wyodrębnij sekcję
        if section == "mda":
            text = ""
            text_lower = full_text.lower()
            for pattern in MDA_PATTERNS:
                match = re.search(pattern, text_lower)
                if match:
                    start = match.start()
                    remaining = text_lower[start + 100:]
                    end_match = re.search(r"\bitem\s*\d+[a-z]?[.\s]", remaining)
                    end = start + 100 + end_match.start() if end_match else min(start + 100_000, len(full_text))
                    text = full_text[start:end].strip()
                    break

            if len(text) < 200:
                mid = len(full_text) // 4
                text = full_text[mid:mid + 80_000]
                section_used = "full_fallback"
            else:
                section_used = "mda"
        else:
            text = full_text
            section_used = "full"

        # Chunkuj
        words = text.split()
        if len(words) <= CHUNK_MAX_WORDS:
            chunks = [text] if words else []
        else:
            chunks = []
            s = 0
            while s < len(words):
                e = s + CHUNK_MAX_WORDS
                chunks.append(" ".join(words[s:e]))
                s = e - CHUNK_OVERLAP_WORDS

        return {
            **filing,
            "chunks": chunks,
            "section_used": section_used,
            "text_length": len(text),
        }
    except Exception as e:
        return {**filing, "chunks": [], "section_used": "error", "text_length": 0}


# ============================================================
# FinBERT NA GPU
# ============================================================

def load_finbert():
    """Ładuje FinBERT na GPU."""
    model_name = "ProsusAI/finbert"
    print(f"Ładuję {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated(0) / 1e6
        print(f"GPU: {name} ({mem:.1f} GB)")
        print(f"Model zajmuje: {used:.0f} MB VRAM")
    else:
        print("⚠️  Brak GPU — będzie wolno!")

    return model, tokenizer, device


@torch.no_grad()
def run_finbert_on_chunks(all_chunks: list[str], model, tokenizer, device) -> np.ndarray:
    """
    Puszcza WSZYSTKIE chunki przez FinBERT w dużych batchach.
    Zwraca array (n, 3) [positive, negative, neutral].
    """
    all_probs = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        del inputs, logits
    return np.vstack(all_probs)


def chunks_to_sentiment(probs: np.ndarray) -> dict:
    """Agreguje wyniki FinBERT dla chunków jednego dokumentu."""
    pos = probs[:, 0]
    neg = probs[:, 1]
    labels = np.argmax(probs, axis=1)
    return {
        "positive": round(pos.mean(), 6),
        "negative": round(neg.mean(), 6),
        "neutral": round(probs[:, 2].mean(), 6),
        "net_score": round(float(pos.mean() - neg.mean()), 6),
        "pct_positive": round((labels == 0).mean(), 6),
        "pct_negative": round((labels == 1).mean(), 6),
        "n_chunks": len(probs),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings_dir", default="/content/sec_filings")
    parser.add_argument("--output", default="./sentiment_results.csv")
    parser.add_argument("--section", default="mda", choices=["mda", "full"])
    args, _ = parser.parse_known_args()

    # 1. Znajdź pliki
    print("Skanuję sprawozdania...")
    filings = find_filings(args.filings_dir)
    print(f"Znaleziono: {len(filings)} plików")

    if not filings:
        print("Brak plików!")
        return

    # 2. Sprawdź co już zrobione
    done = set()
    file_exists = os.path.exists(args.output)
    if file_exists:
        import pandas as pd
        try:
            existing = pd.read_csv(args.output)
            for _, row in existing.iterrows():
                done.add(f"{row['ticker']}_{row['filing_type']}_{row['filing_date']}")
            print(f"Już przetworzono: {len(done)} — pomijam je")
        except Exception:
            file_exists = False

    remaining = [f for f in filings
                 if f"{f['ticker']}_{f['filing_type']}_{f['filing_date']}" not in done]
    print(f"Do przetworzenia: {len(remaining)}")

    if not remaining:
        print("Wszystko już przetworzone!")
        return

    # 3. Załaduj FinBERT
    model, tokenizer, device = load_finbert()

    # 4. Otwórz CSV
    out_file = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()

    processed = 0
    errors = 0

    # 5. Przetwarzaj w grupach: parsuj równolegle → GPU batchem
    for batch_start in tqdm(range(0, len(remaining), FILES_PER_GPU_BATCH),
                            desc="Batche", total=len(remaining) // FILES_PER_GPU_BATCH + 1):
        batch_filings = remaining[batch_start:batch_start + FILES_PER_GPU_BATCH]

        # --- Parsuj HTML równolegle na CPU ---
        with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as pool:
            futures = {pool.submit(parse_single_filing, f, args.section): f
                       for f in batch_filings}
            parsed = []
            for future in as_completed(futures):
                parsed.append(future.result())

        # --- Zbierz WSZYSTKIE chunki z tej grupy plików ---
        all_chunks = []
        chunk_map = []  # (index_w_parsed, start, end)
        for i, p in enumerate(parsed):
            start = len(all_chunks)
            all_chunks.extend(p["chunks"])
            end = len(all_chunks)
            chunk_map.append((i, start, end))

        # --- Jeden duży GPU batch ---
        if all_chunks:
            all_probs = run_finbert_on_chunks(all_chunks, model, tokenizer, device)
        else:
            all_probs = np.empty((0, 3))

        # --- Zapisz wyniki per plik ---
        for i, start, end in chunk_map:
            p = parsed[i]
            if end > start:
                sentiment = chunks_to_sentiment(all_probs[start:end])
            else:
                sentiment = {
                    "positive": None, "negative": None, "neutral": None,
                    "net_score": None, "pct_positive": None, "pct_negative": None,
                    "n_chunks": 0,
                }

            row = {
                "ticker": p["ticker"],
                "filing_type": p["filing_type"],
                "filing_date": p["filing_date"],
                "section_used": p["section_used"],
                "text_length": p["text_length"],
                **sentiment,
            }
            writer.writerow(row)
            processed += 1

        out_file.flush()
        del all_chunks, all_probs, parsed
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_file.close()

    print(f"\n{'='*50}")
    print(f"GOTOWE!")
    print(f"{'='*50}")
    print(f"  Przetworzono:  {processed}")
    print(f"  Błędy:         {errors}")
    print(f"  Wynik:         {args.output}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(0) / 1e6
        print(f"  Peak GPU RAM:  {peak:.0f} MB")


if __name__ == "__main__":
    main()