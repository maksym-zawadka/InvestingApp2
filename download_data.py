#!/usr/bin/env python3
"""
Pobiera dane historyczne (2010-dziś) dla ~500 akcji z indeksu S&P 500
z Yahoo Finance i zapisuje każdą spółkę do osobnego pliku TXT.

Wymagane biblioteki:
    pip install yfinance pandas requests beautifulsoup4

Użycie:
    python download_sp500.py
"""

import os
import time
import datetime
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup


def get_sp500_tickers() -> list[str]:
    """Pobiera aktualną listę tickerów S&P 500 z Wikipedii."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find("tbody").find_all("tr")[1:]:
        ticker = row.find("td").text.strip()
        # Wikipedia używa kropki, Yahoo używa myślnika (np. BRK.B -> BRK-B)
        ticker = ticker.replace(".", "-")
        tickers.append(ticker)

    print(f"Znaleziono {len(tickers)} tickerów S&P 500.")
    return sorted(tickers)


def download_and_save(
    ticker: str,
    start: str,
    end: str,
    output_dir: str,
) -> bool:
    """
    Pobiera dane OHLCV dla jednego tickera i zapisuje do pliku TXT.
    Zwraca True jeśli sukces, False w razie błędu.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"  [{ticker}] BRAK DANYCH – pominięto.")
            return False

        # yfinance może zwrócić MultiIndex kolumn – spłaszczamy
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Wybieramy i porządkujemy kolumny
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].copy()

        # Zaokrąglamy ceny do 3 miejsc, Volume jako int
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = df[c].round(3)
        df["Volume"] = df["Volume"].fillna(0).astype(int)

        # Formatujemy indeks (datę)
        df.index.name = "Date"
        df.index = df.index.strftime("%Y-%m-%d")

        # Zapis do TXT (format CSV)
        filepath = os.path.join(output_dir, f"{ticker}.txt")
        df.to_csv(filepath, sep=",")

        rows = len(df)
        print(f"  [{ticker}] OK – {rows} wierszy -> {filepath}")
        return True

    except Exception as e:
        print(f"  [{ticker}] BŁĄD: {e}")
        return False


def main():
    # ── Konfiguracja ──────────────────────────────────────────────
    START_DATE = "2010-01-01"
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    OUTPUT_DIR = "sp500_data"
    PAUSE_SECONDS = 0.5  # pauza między requestami (unikanie bana)
    # ──────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Pobieranie danych S&P 500 z Yahoo Finance")
    print(f"  Zakres: {START_DATE}  →  {END_DATE}")
    print(f"  Katalog wyjściowy: {OUTPUT_DIR}/")
    print("=" * 60)

    tickers = get_sp500_tickers()

    success = 0
    failed = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"\n[{i}/{len(tickers)}] Pobieram: {ticker}")
        ok = download_and_save(ticker, START_DATE, END_DATE, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1
            failed_tickers.append(ticker)

        # Krótka pauza, żeby Yahoo nas nie zablokował
        if i < len(tickers):
            time.sleep(PAUSE_SECONDS)

    # ── Podsumowanie ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  GOTOWE!  Sukces: {success}  |  Błędy: {failed}")
    if failed_tickers:
        print(f"  Nieudane tickery: {', '.join(failed_tickers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()