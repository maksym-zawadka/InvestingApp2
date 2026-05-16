"""
SEC EDGAR - Pobieranie sprawozdań kwartalnych (10-Q) i rocznych (10-K)
dla TOP 50 amerykańskich firm wg kapitalizacji rynkowej.

Wymagania:
    pip install requests tqdm

Użycie:
    python sec_edgar_downloader.py

Uwagi:
    - SEC EDGAR wymaga nagłówka User-Agent z danymi kontaktowymi.
      Zmień USER_NAME i USER_EMAIL na swoje dane.
    - SEC ogranicza liczbę zapytań do ~10/sek. Skrypt stosuje throttling.
    - Pliki zapisywane są w folderze ./sec_filings/<TICKER>/<FILING_TYPE>/
"""

import os
import time
import json
import requests
from datetime import datetime
from tqdm import tqdm

# ============================================================
# KONFIGURACJA - ZMIEŃ NA SWOJE DANE
# ============================================================
USER_NAME = "Maksym Zawadka"
USER_EMAIL = "sportwsieci1@gmail.com"
USER_AGENT = f"{USER_NAME} ({USER_EMAIL})"

# Folder wyjściowy
OUTPUT_DIR = "./sec_filings"

# Zakres lat do pobrania (ostatnie 10 lat)
YEARS_BACK = 10
MIN_FILING_DATE = str(datetime.now().year - YEARS_BACK) + "-01-01"

# Ile max sprawozdań każdego typu szukać (bufor - filtrujemy po dacie)
MAX_FILINGS_PER_TYPE = 50  # wystarczająco dużo, by objąć 10 lat

# Typy sprawozdań
FILING_TYPES = ["10-K", "10-Q"]

# Opóźnienie między zapytaniami (SEC limit: 10 req/s)
REQUEST_DELAY = 0.12  # sekundy

# ============================================================
# TOP 50 firm USA (ticker -> CIK)
# CIK można też pobrać automatycznie - patrz funkcja get_cik()
# ============================================================
TOP_50_TICKERS = [
    #"AAPL",   # Apple
    # "MSFT",   # Microsoft
    # "NVDA",   # NVIDIA
    # "AMZN",   # Amazon
    # "GOOGL",  # Alphabet (Class A)
    # "META",   # Meta Platforms
    # "BRK-B",  # Berkshire Hathaway
    # "LLY",    # Eli Lilly
    # "AVGO",   # Broadcom
    # "JPM",    # JPMorgan Chase
    # "TSLA",   # Tesla
    # "WMT",    # Walmart
    # "UNH",    # UnitedHealth
    # "XOM",    # Exxon Mobil
    # "V",      # Visa
    # "MA",     # Mastercard
    # "PG",     # Procter & Gamble
    # "JNJ",    # Johnson & Johnson
    # "COST",   # Costco
    # "HD",     # Home Depot
    # "ORCL",   # Oracle
    # "ABBV",   # AbbVie
    "MRK",    # Merck
    "BAC",    # Bank of America
    "KO",     # Coca-Cola
    "CRM",    # Salesforce
    "CVX",    # Chevron
    "NFLX",   # Netflix
    "AMD",    # AMD
    "PEP",    # PepsiCo
    "TMO",    # Thermo Fisher
    "LIN",    # Linde
    "ACN",    # Accenture
    "ADBE",   # Adobe
    "MCD",    # McDonald's
    "CSCO",   # Cisco
    "ABT",    # Abbott Laboratories
    "WFC",    # Wells Fargo
    "DHR",    # Danaher
    "TXN",    # Texas Instruments
    "QCOM",   # Qualcomm
    "INTC",   # Intel
    "INTU",   # Intuit
    "CMCSA",  # Comcast
    "CAT",    # Caterpillar
    "AMAT",   # Applied Materials
    "VZ",     # Verizon
    "AXP",    # American Express
    "IBM",    # IBM
    "GE",     # GE Aerospace
]


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def create_session():
    """Tworzy sesję HTTP z odpowiednim User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def get_cik_mapping(session: requests.Session) -> dict:
    """
    Pobiera mapowanie ticker -> CIK z SEC EDGAR.
    Zwraca słownik {TICKER: CIK_STRING_10_DIGITS}.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    print("Pobieram mapowanie ticker -> CIK z SEC...")
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik

    print(f"  Znaleziono {len(mapping)} firm w bazie SEC.")
    return mapping


def _extract_filings_from_block(block: dict, filing_type: str,
                                min_date: str) -> list:
    """Wyciąga pasujące sprawozdania z bloku danych (recent lub historyczny)."""
    forms = block.get("form", [])
    accessions = block.get("accessionNumber", [])
    dates = block.get("filingDate", [])
    primary_docs = block.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        filing_date = dates[i]
        if form == filing_type and filing_date >= min_date:
            results.append({
                "accession_number": accessions[i],
                "filing_date": filing_date,
                "primary_document": primary_docs[i],
                "filing_type": filing_type,
            })
    return results


def get_filings_list(session: requests.Session, cik: str, filing_type: str,
                     min_date: str = MIN_FILING_DATE) -> list:
    """
    Pobiera listę sprawozdań danego typu dla danej firmy z ostatnich N lat.
    Obsługuje paginację - jeśli 'recent' nie obejmuje pełnego zakresu dat,
    pobiera dodatkowe pliki historyczne z EDGAR Submissions API.

    Zwraca listę słowników z kluczami:
        - accession_number
        - filing_date
        - primary_document
        - filing_type
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(REQUEST_DELAY)
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()

    filings_section = data.get("filings", {})

    # --- Blok "recent" (najnowsze ~1000 wpisów) ---
    recent = filings_section.get("recent", {})
    results = _extract_filings_from_block(recent, filing_type, min_date)

    # --- Bloki historyczne (starsze sprawozdania) ---
    # SEC zwraca listę plików JSON z kolejnymi porcjami danych
    history_files = filings_section.get("files", [])
    for file_info in history_files:
        filename = file_info.get("name", "")
        if not filename:
            continue

        hist_url = f"https://data.sec.gov/submissions/{filename}"
        time.sleep(REQUEST_DELAY)
        try:
            hist_resp = session.get(hist_url)
            hist_resp.raise_for_status()
            hist_data = hist_resp.json()
        except requests.RequestException:
            continue

        older = _extract_filings_from_block(hist_data, filing_type, min_date)
        results.extend(older)

        # Jeśli najstarszy wpis w tym pliku jest starszy niż min_date,
        # nie ma sensu pobierać kolejnych plików historycznych
        all_dates = hist_data.get("filingDate", [])
        if all_dates and all_dates[-1] < min_date:
            break

    # Sortuj od najnowszych
    results.sort(key=lambda x: x["filing_date"], reverse=True)
    return results


def download_filing(session: requests.Session, cik: str, filing: dict,
                    ticker: str) -> str | None:
    """
    Pobiera pojedyncze sprawozdanie i zapisuje je na dysk.
    Zwraca ścieżkę do zapisanego pliku lub None w razie błędu.
    """
    accession = filing["accession_number"].replace("-", "")
    primary_doc = filing["primary_document"]
    filing_date = filing["filing_date"]
    filing_type = filing["filing_type"]

    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"\
          f"{accession}/{primary_doc}"

    # Folder docelowy
    type_dir = filing_type.replace("-", "")  # 10K, 10Q
    out_dir = os.path.join(OUTPUT_DIR, ticker, type_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Nazwa pliku
    ext = os.path.splitext(primary_doc)[1] or ".html"
    filename = f"{ticker}_{filing_type}_{filing_date}{ext}"
    filepath = os.path.join(out_dir, filename)

    # Jeśli plik już istnieje, pomiń
    if os.path.exists(filepath):
        return filepath

    time.sleep(REQUEST_DELAY)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    BŁĄD pobierania {url}: {e}")
        return None

    with open(filepath, "wb") as f:
        f.write(resp.content)

    return filepath


def save_index(all_filings: dict):
    """Zapisuje indeks wszystkich pobranych sprawozdań jako JSON."""
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_filings, f, indent=2, ensure_ascii=False)
    print(f"\nIndeks zapisany: {index_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SEC EDGAR - Pobieranie sprawozdań 10-K / 10-Q")
    print(f"Top {len(TOP_50_TICKERS)} firm USA")
    print(f"Zakres dat: {MIN_FILING_DATE} — dziś ({YEARS_BACK} lat)")
    print("=" * 60)

    if USER_EMAIL == "your.email@example.com":
        print("\n⚠️  UWAGA: Zmień USER_NAME i USER_EMAIL w skrypcie!")
        print("   SEC EDGAR wymaga prawdziwych danych kontaktowych.\n")

    session = create_session()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Pobierz mapowanie ticker -> CIK
    cik_map = get_cik_mapping(session)

    # 2. Dla każdego tickera znajdź CIK
    tickers_with_cik = []
    not_found = []
    for ticker in TOP_50_TICKERS:
        # Obsługa wariantów (np. BRK-B -> BRK-B lub BRKB)
        cik = cik_map.get(ticker) or cik_map.get(ticker.replace("-", ""))
        if cik:
            tickers_with_cik.append((ticker, cik))
        else:
            not_found.append(ticker)

    if not_found:
        print(f"\n⚠️  Nie znaleziono CIK dla: {', '.join(not_found)}")

    print(f"\nZnaleziono CIK dla {len(tickers_with_cik)}/{len(TOP_50_TICKERS)} firm.\n")

    # 3. Pobieraj sprawozdania
    all_filings_index = {}
    total_downloaded = 0
    total_errors = 0

    for ticker, cik in tqdm(tickers_with_cik, desc="Firmy", unit="firma"):
        all_filings_index[ticker] = {"cik": cik, "filings": []}

        for filing_type in FILING_TYPES:
            filings = get_filings_list(session, cik, filing_type)

            for filing in filings:
                filepath = download_filing(session, cik, filing, ticker)
                filing_record = {
                    "type": filing["filing_type"],
                    "date": filing["filing_date"],
                    "accession": filing["accession_number"],
                    "file": filepath,
                }
                all_filings_index[ticker]["filings"].append(filing_record)

                if filepath:
                    total_downloaded += 1
                else:
                    total_errors += 1

    # 4. Zapisz indeks
    save_index(all_filings_index)

    # 5. Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"  Firm przetworzonych:     {len(tickers_with_cik)}")
    print(f"  Plików pobranych:        {total_downloaded}")
    print(f"  Błędów:                  {total_errors}")
    print(f"  Folder wyjściowy:        {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Struktura: {OUTPUT_DIR}/<TICKER>/<10K|10Q>/")
    print("=" * 60)


if __name__ == "__main__":
    main()