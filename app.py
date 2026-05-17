import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta, datetime
import importlib.util
import os
import urllib.request
import xml.etree.ElementTree as ET
import email.utils
from streamlit_echarts import st_echarts

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --border:    #1e1e2e;
    --accent:    #00ff88;
    --accent2:   #ff6b35;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --up:        #00ff88;
    --down:      #ff4466;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 0.5rem !important;
}
.app-header { display: flex; align-items: baseline; gap: 16px; padding: 32px 0 8px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.app-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; color: var(--text); margin: 0; }
.app-title span { color: var(--accent); }
.app-sub { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--muted); margin: 0; padding-bottom: 4px; }

.controls-bar { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; }

.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-pill { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 18px; font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.stat-pill .label { color: var(--muted); display: block; font-size: 0.65rem; margin-bottom: 2px; }
.stat-pill .value { color: var(--text); font-weight: 700; }
.stat-pill .value.up { color: var(--up); }
.stat-pill .value.down { color: var(--down); }

.section-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin: 28px 0 12px; display: flex; align-items: center; gap: 8px; }
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.pred-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px 28px; margin-top: 8px; }
.pred-card h3 { font-size: 1rem; font-weight: 700; margin: 0 0 16px; color: var(--text); }

.model-badge { display: inline-block; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: var(--accent); font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; margin-bottom: 12px; }

.pred-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--accent); margin: 8px 0 4px; }
.pred-delta { font-family: 'Space Mono', monospace; font-size: 0.85rem; }
.pred-delta.up { color: var(--up); }
.pred-delta.down { color: var(--down); }

.sentiment-row { display: flex; gap: 8px; margin-top: 16px; align-items: center; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); }
.sent-bar { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.sent-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--down), var(--up)); }

/* News Cards */
.news-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; transition: border-color 0.2s; }
.news-card:hover { border-color: rgba(0,255,136,0.3); }
.news-meta { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--muted); margin-bottom: 6px; }
.news-title { color: var(--text); text-decoration: none; font-weight: 700; font-size: 0.95rem; transition: color 0.2s; display: block; line-height: 1.4; }
.news-title:hover { color: var(--accent); }

/* Expander styling */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 20px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}
[data-testid="stExpander"] table {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    width: 100%;
    border-collapse: collapse;
}
[data-testid="stExpander"] th {
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 6px 12px;
    text-align: left;
}
[data-testid="stExpander"] td {
    color: var(--text);
    padding: 6px 12px;
    border-bottom: 1px solid rgba(30,30,46,0.5);
}

/* Multiselect styling */
[data-testid="stMultiSelect"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-baseweb="tag"] {
    background: rgba(0,255,136,0.12) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
}
[data-baseweb="tag"] span { color: var(--accent) !important; }

[data-testid="stTextInput"] input, [data-testid="stSelectbox"] select { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stDateInput"] input { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: 'Space Mono', monospace !important; }
div[data-testid="stButton"] button { background: var(--accent) !important; color: #000 !important; border: none !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border-radius: 8px !important; padding: 10px 24px !important; letter-spacing: 0.5px; transition: opacity 0.15s; }
div[data-testid="stButton"] button:hover { opacity: 0.85 !important; }
label, .stDateInput label, .stTextInput label, .stSelectbox label { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within,
div[data-baseweb="base-input"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: inset 0 0 0 1px var(--accent) !important;
}
input:focus, select:focus {
    outline: none !important;
    border-color: var(--accent) !important;
}
.stTextInput div[data-baseweb="input"] > div:focus-within,
.stDateInput div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: inset 0 0 0 1px var(--accent) !important;
}
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stSelectbox div[data-baseweb="select"] > div:active {
    border-color: var(--accent) !important;
    box-shadow: inset 0 0 0 1px var(--accent) !important;
}
input:focus, select:focus, textarea:focus {
    outline: none !important;
}
[data-testid="stTextInput"] input {
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info

    def safe(key, fmt="{:.2f}", fallback="—"):
        v = info.get(key)
        if v is None or v == 0:
            return fallback
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    return {
        "C/Z": safe("trailingPE"),
        "C/Z (Prognozowana)": safe("forwardPE"),
        "C/WK": safe("priceToBook"),
        "EV/EBITDA": safe("enterpriseToEbitda"),
        "Marża netto": safe("profitMargins", "{:.1%}"),
        "ROE": safe("returnOnEquity", "{:.1%}"),
        "Dług/Kapitał": safe("debtToEquity", "{:.1f}"),
        "Stopa dywidendy": safe("dividendYield", "{:.2f}%"),
        "Beta": safe("beta"),
        "Kapitalizacja": _fmt_mcap(info.get("marketCap")),
        "shortName": info.get("shortName", ticker),
    }

@st.cache_data
def get_sp500_tickers() -> list:
    """Zwraca zahardcodowaną listę tickerów S&P 500 (szybkie i niezawodne)."""
    tickers = [
        "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
        "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
        "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
        "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH",
        "APTV", "ARE", "ATO", "AVGO", "AWK", "AXP", "BA", "BAC", "BK", "BKNG",
        "BLK", "BMY", "BRK-B", "BSX", "C", "CAT", "CB", "CCI", "CDNS", "CI",
        "CME", "CMG", "CMI", "COF", "COP", "COST", "CRM", "CRWD", "CSCO", "CSX",
        "CVS", "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EOG", "EPAM", "ETN",
        "EW", "EXC", "F", "FCX", "FDX", "FI", "FSLR", "GD", "GE", "GILD",
        "GIS", "GLW", "GM", "GOOG", "GOOGL", "GS", "HAL", "HD", "HON", "HPE",
        "HPQ", "IBM", "ICE", "INTC", "INTU", "ISRG", "ITW", "JNJ", "JPM", "K",
        "KHC", "KLAC", "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MAR",
        "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MRO", "MS",
        "MSFT", "MU", "NEE", "NEM", "NFLX", "NKE", "NOC", "NOW", "NVDA", "NXPI",
        "O", "ORCL", "OXY", "PANW", "PEP", "PFE", "PG", "PGR", "PH", "PLD",
        "PM", "PNC", "PYPL", "QCOM", "REGN", "RTX", "SBUX", "SCHW", "SLB", "SNPS",
        "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP",
        "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
    ]
    return sorted(tickers)

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_news(ticker: str) -> list:
    try:
        # Google News RSS
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        news_items = []

        for item in root.findall('.//item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            pubDate = item.find('pubDate').text

            source_tag = item.find('source')
            publisher = source_tag.text if source_tag is not None else "Wiadomości Giełdowe"

            dt_tuple = email.utils.parsedate_tz(pubDate)
            if dt_tuple:
                dt = datetime.fromtimestamp(email.utils.mktime_tz(dt_tuple))
                date_str = dt.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = pubDate

            news_items.append({
                "title": title,
                "link": link,
                "publisher": publisher,
                "date_str": date_str
            })

        return news_items
    except Exception as e:
        return []


def _fmt_mcap(v):
    if not v:
        return "—"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    return str(v)


def get_currency(ticker: str) -> tuple:
    t = ticker.upper()
    for suffix, symbol, after in [(".WA", "zł", True), (".PA", "€", False), (".DE", "€", False), (".MI", "€", False),
                                  (".MC", "€", False), (".AS", "€", False), (".BR", "€", False), (".L", "£", False),
                                  (".HK", "HK$", False), (".T", "¥", False)]:
        if t.endswith(suffix):
            return symbol, after
    return "$", False


def fmt_price(value: float, symbol: str, after: bool) -> str:
    s = f"{value:,.2f}"
    return f"{s} {symbol}" if after else f"{symbol}{s}"


def compute_indicators(df: pd.DataFrame, selected: list) -> dict:
    close = df["Close"]
    result = {}

    if "SMA 20" in selected:
        result["SMA 20"] = [round(v, 2) if not pd.isna(v) else "-" for v in close.rolling(20).mean()]
    if "SMA 50" in selected:
        result["SMA 50"] = [round(v, 2) if not pd.isna(v) else "-" for v in close.rolling(50).mean()]
    if "EMA 20" in selected:
        result["EMA 20"] = [round(v, 2) if not pd.isna(v) else "-" for v in close.ewm(span=20, adjust=False).mean()]
    if "EMA 50" in selected:
        result["EMA 50"] = [round(v, 2) if not pd.isna(v) else "-" for v in close.ewm(span=50, adjust=False).mean()]
    if "Bollinger Bands" in selected:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result["BB górny"] = [round(v, 2) if not pd.isna(v) else "-" for v in (sma20 + 2 * std20)]
        result["BB środek"] = [round(v, 2) if not pd.isna(v) else "-" for v in sma20]
        result["BB dolny"] = [round(v, 2) if not pd.isna(v) else "-" for v in (sma20 - 2 * std20)]
    if "MACD" in selected:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        result["__MACD_line"] = [round(v, 4) if not pd.isna(v) else "-" for v in macd]
        result["__MACD_signal"] = [round(v, 4) if not pd.isna(v) else "-" for v in signal]
        result["__MACD_hist"] = [round(v, 4) if not pd.isna(v) else 0 for v in (macd - signal)]
    if "RSI" in selected:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        result["__RSI"] = [round(v, 2) if not pd.isna(v) else "-" for v in (100 - 100 / (1 + rs))]

    return result


def render_chart(df: pd.DataFrame, ticker: str, interval: str, indicators: dict) -> None:
    if interval in ["5m", "15m", "30m", "1h"]:
        dates = df.index.strftime("%Y-%m-%d %H:%M").tolist()
    else:
        dates = df.index.strftime("%Y-%m-%d").tolist()

    k_data = [[round(o, 2), round(c, 2), round(l, 2), round(h, 2)] for o, c, l, h in
              zip(df['Open'], df['Close'], df['Low'], df['High'])]
    vol_data = [{"value": int(v), "itemStyle": {"color": "rgba(0,255,136,0.4)" if c >= o else "rgba(255,68,102,0.4)"}}
                for v, o, c in zip(df['Volume'], df['Open'], df['Close'])]

    has_macd = "__MACD_line" in indicators
    has_rsi = "__RSI" in indicators
    extra_panels = sum([has_macd, has_rsi])

    overlay_colors = {
        "SMA 20": "#00bfff",
        "SMA 50": "#ff6b35",
        "EMA 20": "#a78bfa",
        "EMA 50": "#f472b6",
        "BB górny": "rgba(255,200,0,0.6)",
        "BB środek": "rgba(255,200,0,0.35)",
        "BB dolny": "rgba(255,200,0,0.6)",
    }

    main_height = 55 - extra_panels * 10
    vol_top = main_height + 8
    grids = [
        {"left": "5%", "right": "5%", "top": "3%", "height": f"{main_height}%"},
        {"left": "5%", "right": "5%", "top": f"{vol_top}%", "height": "10%"},
    ]
    x_axes = [
        {"type": "category", "data": dates, "boundaryGap": False,
         "axisLine": {"lineStyle": {"color": "#1e1e2e"}}, "axisLabel": {"color": "#6b6b80"},
         "splitLine": {"show": False}},
        {"type": "category", "gridIndex": 1, "data": dates,
         "axisLabel": {"show": False}, "axisLine": {"show": False}, "splitLine": {"show": False}},
    ]
    y_axes = [
        {"scale": True, "splitArea": {"show": False},
         "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.04)"}}, "axisLabel": {"color": "#6b6b80"}},
        {"scale": True, "gridIndex": 1, "splitNumber": 2,
         "axisLabel": {"show": False}, "axisLine": {"show": False}, "axisTick": {"show": False},
         "splitLine": {"show": False}},
    ]

    panel_idx = 2
    panel_top = vol_top + 13
    macd_grid_idx = rsi_grid_idx = None

    if has_macd:
        grids.append({"left": "5%", "right": "5%", "top": f"{panel_top}%", "height": "10%"})
        x_axes.append({"type": "category", "gridIndex": panel_idx, "data": dates,
                       "axisLabel": {"color": "#6b6b80", "fontSize": 10},
                       "axisLine": {"lineStyle": {"color": "#1e1e2e"}}, "splitLine": {"show": False}})
        y_axes.append({"scale": True, "gridIndex": panel_idx, "splitNumber": 2,
                       "axisLabel": {"color": "#6b6b80", "fontSize": 9},
                       "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.04)"}}})
        macd_grid_idx = panel_idx
        panel_idx += 1
        panel_top += 13

    if has_rsi:
        grids.append({"left": "5%", "right": "5%", "top": f"{panel_top}%", "height": "10%"})
        x_axes.append({"type": "category", "gridIndex": panel_idx, "data": dates,
                       "axisLabel": {"color": "#6b6b80", "fontSize": 10},
                       "axisLine": {"lineStyle": {"color": "#1e1e2e"}}, "splitLine": {"show": False}})
        y_axes.append({"scale": True, "gridIndex": panel_idx, "splitNumber": 2, "min": 0, "max": 100,
                       "axisLabel": {"color": "#6b6b80", "fontSize": 9},
                       "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.04)"}}})
        rsi_grid_idx = panel_idx

    series = [
        {"name": ticker, "type": "candlestick", "data": k_data,
         "itemStyle": {"color": "#00ff88", "color0": "#ff4466",
                       "borderColor": "#00ff88", "borderColor0": "#ff4466"}},
        {"name": "Volume", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1, "data": vol_data},
    ]

    for name, values in indicators.items():
        if name.startswith("__"):
            continue
        series.append({
            "name": name, "type": "line", "data": values,
            "smooth": True, "showSymbol": False,
            "lineStyle": {
                "color": overlay_colors.get(name, "#ffffff"),
                "width": 1.5,
                "type": "dashed" if "BB" in name else "solid",
            },
        })

    if has_macd and macd_grid_idx is not None:
        hist = indicators["__MACD_hist"]
        hist_colors = [{"value": v, "itemStyle": {"color": "#00ff88" if (v != "-" and v >= 0) else "#ff4466"}} for v in
                       hist]
        series += [
            {"name": "MACD", "type": "line", "xAxisIndex": macd_grid_idx, "yAxisIndex": macd_grid_idx,
             "data": indicators["__MACD_line"], "smooth": True, "showSymbol": False,
             "lineStyle": {"color": "#00bfff", "width": 1.5}},
            {"name": "Signal", "type": "line", "xAxisIndex": macd_grid_idx, "yAxisIndex": macd_grid_idx,
             "data": indicators["__MACD_signal"], "smooth": True, "showSymbol": False,
             "lineStyle": {"color": "#f472b6", "width": 1.5}},
            {"name": "Histogram", "type": "bar", "xAxisIndex": macd_grid_idx, "yAxisIndex": macd_grid_idx,
             "data": hist_colors},
        ]

    if has_rsi and rsi_grid_idx is not None:
        series.append({
            "name": "RSI", "type": "line",
            "xAxisIndex": rsi_grid_idx, "yAxisIndex": rsi_grid_idx,
            "data": indicators["__RSI"], "smooth": True, "showSymbol": False,
            "lineStyle": {"color": "#a78bfa", "width": 1.5},
            "markLine": {
                "silent": True,
                "lineStyle": {"color": "rgba(255,255,255,0.2)", "type": "dashed"},
                "data": [{"yAxis": 30}, {"yAxis": 70}],
            },
        })

    chart_height = 420 + extra_panels * 130

    legend_items = [n for n in indicators if not n.startswith("__")]
    if has_macd:
        legend_items += ["MACD", "Signal"]
    if has_rsi:
        legend_items += ["RSI"]

    option = {
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": "Space Mono, monospace", "color": "#6b6b80"},
        "legend": {
            "data": legend_items,
            "top": 0, "right": "5%",
            "textStyle": {"color": "#6b6b80", "fontSize": 10, "fontFamily": "Space Mono"},
            "inactiveColor": "#2a2a3a",
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross", "label": {"backgroundColor": "#1e1e2e", "fontFamily": "Space Mono"}},
            "backgroundColor": "#13131a", "borderColor": "#1e1e2e", "textStyle": {"color": "#e8e8f0"},
        },
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "dataZoom": [{"type": "inside", "xAxisIndex": list(range(len(grids))), "start": 0, "end": 100}],
        "series": series,
    }
    st_echarts(options=option, height=f"{chart_height}px")


def load_model(model_type: str):
    path = os.path.join(os.path.dirname(__file__), f"models/{model_type}_model.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Brak pliku modelu: {path}\n"
            "Upewnij się, że pliki modeli są w folderze models/ obok app.py."
        )
    spec = importlib.util.spec_from_file_location("model_module", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load()


def run_prediction(model_type: str, df: pd.DataFrame, ticker: str):
    model = load_model(model_type)
    return model.predict(df, ticker)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1 class="app-title">Stock<span>Predictor</span></h1>
    <p class="app-sub">// ML-powered price prediction · LSTM · XGBoost · FinBERT </p>
</div>
""", unsafe_allow_html=True)

# ── Instrukcja ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️  Jak używać aplikacji"):
    st.markdown("""
**Wpisywanie tickerów**

| Rynek | Przykład | Końcówka |
|---|---|---|
| USA (NYSE / NASDAQ) | `AAPL`, `TSLA`, `NVDA` | *(brak)* |
| Polska (GPW) | `CDR.WA`, `PKN.WA`, `PKNORLEN.WA` | `.WA` |
| Niemcy (XETRA) | `SAP.DE`, `BMW.DE` | `.DE` |
| Wielka Brytania (LSE) | `HSBA.L`, `BP.L` | `.L` |
| Francja (Euronext) | `MC.PA`, `AIR.PA` | `.PA` |
| Włochy | `ENI.MI` | `.MI` |
| Hongkong | `9988.HK`, `0700.HK` | `.HK` |
| Japonia | `7203.T` | `.T` |

**Interwały czasowe**
- `5m / 15m / 30m` — dane dostępne maksymalnie do **60 dni** wstecz
- `1h` — dane dostępne maksymalnie do **~2 lat** wstecz
- `1d / 1wk` — pełna historia bez ograniczeń

**Wskaźniki techniczne na wykresie**

| Wskaźnik | Opis |
|---|---|
| **SMA 20 / 50** | Prosta średnia krocząca — wygładza cenę, pokazuje trend. SMA 20 = krótkoterminowy, SMA 50 = średnioterminowy. |
| **EMA 20 / 50** | Wykładnicza średnia krocząca — reaguje szybciej niż SMA na zmiany ceny. Lepiej wykrywa odwrócenia trendu. |
| **Bollinger Bands** | Trzy linie: środkowa (SMA 20) + górna/dolna (±2 odchylenia std). Wyjście poza pasmo = sygnał ekstremalnego ruchu. |
| **MACD** | Różnica EMA 12 i EMA 26 + linia sygnału (EMA 9). Przecięcie linii = potencjalny sygnał kupna/sprzedaży. Wyświetlany w osobnym panelu. |
| **RSI** | Indeks siły względnej (14 dni). Powyżej 70 = wykupienie; poniżej 30 = wyprzedanie. Wyświetlany w osobnym panelu. |

**Wskaźniki fundamentalne**

| Wskaźnik | Opis | Interpretacja |
|---|---|---|
| **C/Z** | Cena / zysk za ostatnie 12 mies. | Im niższy, tym taniej kupujesz zyski. Wysoki może oznaczać przewartościowanie lub oczekiwania wzrostu. |
| **C/Z (Prognozowana)** | Cena / prognozowany zysk | Patrzy w przyszłość — lepiej oddaje bieżące oczekiwania rynku niż C/Z TTM. |
| **C/WK** | Cena / Wartość Księgowa | C/WK < 1 może sugerować niedowartościowanie. Spółki technologiczne mają naturalnie wysokie wartości. |
| **EV/EBITDA** | Wartość firmy / EBITDA | Przydatny przy porównywaniu spółek z różnym zadłużeniem. Niższy = tańsza wycena. |
| **Marża netto** | Zysk netto / przychody | Ile groszy zysku zostaje z każdej złotówki przychodu. Im wyższa, tym lepsza efektywność operacyjna. |
| **ROE** | Zwrot z kapitału własnego | Jak efektywnie spółka zarabia na pieniądzach akcjonariuszy. Powyżej 15% uważane za dobre. |
| **Dług/Kapitał** | Zobowiązania / kapitał własny | Poziom dźwigni finansowej. Wysokie wartości = większe ryzyko, ale normy są bardzo branżowe. |
| **Stopa dywidendy** | Roczna dywidenda / cena akcji | Stopa zwrotu z samej dywidendy. Bardzo wysoka (>8%) może sygnalizować problemy spółki. |
| **Beta** | Zmienność relatywna do rynku | Beta > 1 = bardziej zmienna niż rynek; < 1 = defensywna; < 0 = porusza się odwrotnie do rynku. |
| **Kapitalizacja** | Łączna wartość rynkowa spółki | Mega cap > 200B · Large > 10B · Mid 2–10B · Small < 2B. |

**Predykcja ML** Model analizuje historię cen zamknięcia oraz sentyment newsów (FinBERT).  
Wynik to prognoza ceny na **następną sesję giełdową**.

> ⚠️ Predykcja ma charakter **wyłącznie edukacyjny** i nie stanowi porady inwestycyjnej.
    """)

# ── State & Callbacks ──────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "interval" not in st.session_state:
    st.session_state.interval = "1d"
if "do_load" not in st.session_state:
    st.session_state.do_load = False


def trigger_load():
    st.session_state.do_load = True


# ── Controls ───────────────────────────────────────────────────────────────────
sp500_list = get_sp500_tickers()
opcja_reczna = "--- INNY (Wpisz ręcznie) ---"
pelna_lista = [opcja_reczna] + sp500_list

with st.container():
    #st.markdown('<div class="controls-bar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([2, 1.5, 2, 2, 1.5])
    with c1:
        # Domyślnie ustawiamy na NVDA lub AAPL, jeśli są na liście
        idx_domyslny = pelna_lista.index("AAPL") if "NVDA" in pelna_lista else 1

        wybor_tickera = st.selectbox(
            "Ticker (Wyszukaj lub wybierz)",
            options=pelna_lista,
            index=idx_domyslny,
            on_change=trigger_load
        )

        # Jeśli użytkownik wybierze opcję ręczną, pokazujemy standardowe pole tekstowe
        if wybor_tickera == opcja_reczna:
            ticker_input = st.text_input("Wpisz własny ticker", placeholder="np. CDR.WA",
                                         on_change=trigger_load).upper()
        else:
            ticker_input = wybor_tickera
    with c2:
        interval_input = st.selectbox("Interwał", ["5m", "15m", "30m", "1h", "1d", "1wk"], index=4)
    with c3:
        start_date = st.date_input("Od", value=date.today() - timedelta(days=180), on_change=trigger_load)
    with c4:
        end_date = st.date_input("Do", value=date.today(), on_change=trigger_load)
    with c5:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("Załaduj →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Ładowanie Danych ───────────────────────────────────────────────────────────
if load_btn or st.session_state.do_load:
    st.session_state.do_load = False

    if ticker_input:
        with st.spinner(f"Pobieranie danych dla {ticker_input}…"):
            try:
                days_diff = (date.today() - start_date).days
                if interval_input in ["5m", "15m", "30m"] and days_diff > 60:
                    st.warning(
                        f"Dla interwału {interval_input} dane w Yahoo Finance są dostępne maksymalnie do 60 dni wstecz. Zakres dat został automatycznie skrócony.")
                    start_date = date.today() - timedelta(days=59)
                elif interval_input == "1h" and days_diff > 730:
                    st.warning(
                        "Dla interwału 1h dane w Yahoo Finance są dostępne maksymalnie do ~2 lat wstecz. Zakres dat został automatycznie skrócony.")
                    start_date = date.today() - timedelta(days=728)

                df = fetch_data(ticker_input, start_date, end_date, interval_input)
                if df.empty:
                    st.error("Brak danych. Sprawdź ticker lub przedział dat.")
                else:
                    st.session_state.df = df
                    st.session_state.ticker = ticker_input
                    st.session_state.interval = interval_input
            except Exception as e:
                st.error(f"Błąd pobierania danych: {e}")

# ── Main Panel ─────────────────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    ticker = st.session_state.ticker
    interval = st.session_state.interval

    cur, cur_after = get_currency(ticker)

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    chg = last - prev
    pct = chg / prev * 100
    hi52 = float(df["High"].max())
    lo52 = float(df["Low"].min())
    vol = int(df["Volume"].iloc[-1])

    up_cls = "up" if chg >= 0 else "down"
    sign = "+" if chg >= 0 else ""

    # ── Stat pills ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><span class="label">LAST CLOSE</span><span class="value">{fmt_price(last, cur, cur_after)}</span></div>
        <div class="stat-pill"><span class="label">ZMIANA</span><span class="value {up_cls}">{sign}{chg:,.2f} ({sign}{pct:.2f}%)</span></div>
        <div class="stat-pill"><span class="label">WOLUMEN</span><span class="value">{vol:,}</span></div>
        <div class="stat-pill"><span class="label">MAX OKRESU</span><span class="value">{fmt_price(hi52, cur, cur_after)}</span></div>
        <div class="stat-pill"><span class="label">MIN OKRESU</span><span class="value">{fmt_price(lo52, cur, cur_after)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fundamentals ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Wskaźniki fundamentalne</div>', unsafe_allow_html=True)
    company_name = ticker
    with st.spinner("Pobieranie wskaźników fundamentalnych…"):
        try:
            fund = fetch_fundamentals(ticker)
            company_name = fund.get("shortName", ticker)
            fund_display = {k: v for k, v in fund.items() if k != "shortName"}
            pills_html = "".join(
                f'<div class="stat-pill"><span class="label">{k}</span><span class="value">{v}</span></div>'
                for k, v in fund_display.items()
            )
            st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)
            st.caption(
                "Źródło: Yahoo Finance · odświeżane co 1h · dane mogą być opóźnione lub niedostępne dla niektórych rynków")
        except Exception as e:
            st.warning(f"Nie udało się pobrać wskaźników fundamentalnych: {e}")

    # ── Nazwa spółki ───────────────────────────────────────────────────────────
    st.markdown(f'''
    <div style="display:flex; align-items:baseline; gap:12px; margin: 24px 0 4px;">
        <span style="font-size:1.6rem; font-weight:800; color:var(--text); letter-spacing:-0.5px;">{company_name}</span>
    </div>
    ''', unsafe_allow_html=True)

    # ── Selektor wskaźników technicznych ──────────────────────────────────────
    ind_col, _ = st.columns([4, 1])
    with ind_col:
        selected_indicators = st.multiselect(
            "Wskaźniki techniczne",
            options=["SMA 20", "SMA 50", "EMA 20", "EMA 50", "Bollinger Bands", "MACD", "RSI"],
            default=["SMA 20"],
            placeholder="Wybierz wskaźniki…",
        )

    indicators = compute_indicators(df, selected_indicators)

    st.markdown('<div class="section-label">Wykres świecowy</div>', unsafe_allow_html=True)
    render_chart(df, ticker, interval, indicators)

    # ── Prediction ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Predykcja modelu</div>', unsafe_allow_html=True)

    with st.container():
        #st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        p1, p2 = st.columns([3, 2])
        with p1:
            model_choice = st.selectbox("Wybierz model", ["LSTM", "XGBoost + FinBERT"], key="model_select")
        with p2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("▶  Uruchom predykcję", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        model_key = "lstm" if "LSTM" in model_choice else "xgboost"

        try:
            if model_key == "lstm":
                with st.spinner("Trwa predykcja LSTM…"):
                    result = run_prediction(model_key, df, ticker)

                    # Zabezpieczenie: jeśli Twój model zwraca (cena, prob), bierzemy prob.
                    # Jeśli zwraca tylko jedno (samo prob), bierzemy całość.
                    if isinstance(result, tuple) or isinstance(result, list):
                        prob = result[-1]
                    else:
                        prob = result

                sentiment_info = None

            else:
                # XGBoost: długi pipeline SEC → FinBERT, pokazujemy status
                status_box = st.empty()
                log_lines = []


                def sec_status(msg: str):
                    log_lines.append(msg)
                    status_box.markdown(
                        "<br>".join(f"<span style='font-family:Space Mono,monospace;"
                                    f"font-size:0.75rem;color:#6b6b80'>{l}</span>"
                                    for l in log_lines[-6:]),
                        unsafe_allow_html=True,
                    )


                model_obj = load_model(model_key)
                predicted_price, prob, sentiment_info = model_obj.predict(df, ticker, status_callback=sec_status)
                status_box.empty()

            # Zabezpieczenie: konwersja do float (na wypadek gdyby model zwrócił tablicę numpy np. [0.55])
            if hasattr(prob, "item"):
                prob = float(prob.item())
            else:
                prob = float(prob)

                # ── Wspólne obliczenia ──────────────────────────────────────────
                price_str = fmt_price(last, cur, cur_after)

                if model_key == "lstm":
                    # Logika i kolory dla LSTM
                    if prob > 0.55:
                        kierunek = "▲ WZROST"
                        kierunek_kolor = "var(--up)"
                    elif prob < 0.45:
                        kierunek = "▼ SPADEK"
                        kierunek_kolor = "var(--down)"
                    else:
                        kierunek = "⏸ NEUTRALNIE"
                        kierunek_kolor = "var(--muted)"

                    prob_pct = int(prob * 100)
                    pasek_label = "P(wzrost następnej sesji)"
                    horyzont = "następna sesja"
                    opis = "Model klasyfikuje kierunek ruchu &#8212; nie prognozuje konkretnej ceny."
                    raport_html = ""

                else:
                    # Logika i kolory dla XGBoost z PRZESKALOWANIEM WIZUALNYM
                    orig_threshold = load_model(model_key).threshold

                    # Ustawiamy historyczne "widełki" Twojego modelu, by ładnie rozciągnąć pasek
                    min_p = 0.34
                    max_p = 0.44

                    # Przeskalowanie (zamiana np. 0.39 na 0.50)
                    if prob < orig_threshold:
                        # Skalowanie od min_p do progu -> na 0.00 do 0.50
                        scaled_prob = 0.5 * max(0.0, (prob - min_p) / (orig_threshold - min_p))
                    else:
                        # Skalowanie od progu do max_p -> na 0.50 do 1.00
                        scaled_prob = 0.5 + 0.5 * min(1.0, (prob - orig_threshold) / (max_p - orig_threshold))

                    # Zamieniamy na ułamek dla widoku (0-100%)
                    prob_pct = int(scaled_prob * 100)

                    # Decyzję podejmujemy na zaktualizowanych danych (środek to teraz 50%)
                    kupuj = prob_pct >= 50

                    if kupuj:
                        kierunek = "&#9650; KUPUJ"
                        kierunek_kolor = "var(--up)"
                    else:
                        kierunek = "&#9660; SPRZEDAJ"
                        kierunek_kolor = "var(--down)"

                    pasek_label = "P(pobicie indeksu SPY w ~63 sesjach)"
                    horyzont = "~3 miesi&#261;ce"
                    opis = "Model przewiduje czy spółka pobije SPY w ciągu ~63 sesji &#8212; nie prognozuje konkretnej ceny."

                    if sentiment_info:
                        ft = sentiment_info["filing_type"]
                        fd = sentiment_info["filing_date"]
                        su = sentiment_info["section_used"]
                        nc = sentiment_info["n_chunks"]
                        raport_html = (
                            '<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
                            'color:var(--muted);margin-top:14px;">'
                            f'Raport: <b style="color:var(--text)">{ft}</b> &middot; '
                            f'{fd} &middot; sekcja: {su} &middot; {nc} chunk&#243;w</div>'
                        )
                    else:
                        raport_html = ""

                # ── Rysowanie karty HTML ────────────────────────────────────────
                card_html = (
                    '<div class="pred-card" style="margin-top:12px;">'
                    f'<div class="model-badge">{model_choice}</div>'
                    f'<h2>Prognoza &#8212; {horyzont}</h2>'
                    '<div style="font-family:Space Mono,monospace;font-size:1.4rem;'
                    f'font-weight:800;margin-bottom:12px;color:{kierunek_kolor};">{kierunek}</div>'
                    '<div style="font-family:Space Mono,monospace;font-size:1rem;'
                    'color:var(--muted);margin:12px 0 4px;">Ostatnie zamkni&#281;cie</div>'
                    f'<div class="pred-value" style="color:#ffffff;font-size:1rem;">{price_str}</div>'
                    '<div style="font-family:Space Mono,monospace;font-size:0.8rem;'
                    f'color:var(--muted);margin-top:8px;">{opis}</div>'
                    '<div class="sentiment-row" style="margin-top:16px;">'
                    f'<span>{pasek_label}</span>'
                    '<div class="sent-bar">'
                    f'<div class="sent-fill" style="width:{prob_pct}%"></div>'
                    '</div>'
                    f'<span>{prob_pct}%</span>'
                    '</div>'
                    f'{raport_html}'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                st.caption("&#9888;&#65039; Predykcja wy&#322;&#261;cznie w celach edukacyjnych. Nie stanowi porady inwestycyjnej.")

        except Exception as e:
            st.error(f"Błąd predykcji: {e}")

    # ── News Feed ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Najnowsze wiadomości (Google News)</div>', unsafe_allow_html=True)

    with st.spinner("Pobieranie wiadomości…"):
        news_items = fetch_news(ticker)

        if not news_items:
            st.info("Brak najnowszych wiadomości dla tego waloru.")
        else:
            for item in news_items:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-meta">{item['publisher']} • {item['date_str']}</div>
                    <a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #2a2a3a; font-family: 'Space Mono', monospace; font-size: 0.9rem;">
        <div style="font-size:3rem; margin-bottom:16px;">📈</div>
        Wpisz ticker i wciśnij <strong style="color:#3a3a5a">Enter</strong> lub kliknij <strong style="color:#3a3a5a">Załaduj →</strong>
    </div>
    """, unsafe_allow_html=True)

# ── Footer / Zastrzeżenie Prawne ───────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top: 60px;
    padding-top: 24px;
    padding-bottom: 24px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    line-height: 1.6;
">
    <strong style="color: var(--text);">⚠️ ZASTRZEŻENIE </strong><br><br>
    Aplikacja została stworzona wyłącznie w celach edukacyjnych i informacyjnych. <br>
    Prezentowane dane historyczne, wskaźniki fundamentalne, techniczne oraz wszelkie wyniki działania modeli uczenia maszynowego (predykcje) <br>
    <b>nie stanowią rekomendacji inwestycyjnej ani porady inwestycyjnej</b> w rozumieniu przepisów prawa. <br>
</div>
""", unsafe_allow_html=True)
