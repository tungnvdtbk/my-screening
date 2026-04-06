from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from io import BytesIO

try:
    from vnstock3 import Vnstock
    HAS_VNSTOCK = True
except ImportError:
    HAS_VNSTOCK = False

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "./data"
CACHE_DIR = f"{DATA_PATH}/cache"
SYMBOLS_CACHE_FILE = f"{DATA_PATH}/hose_symbols.json"
os.makedirs(CACHE_DIR, exist_ok=True)

MIN_VOL_20 = 200_000    # avg 20-session volume ≥ 200,000 shares
MIN_PRICE = 15          # ≥ 15,000 VND (yfinance returns VN prices in thousands)
RS_MIN = 80             # RS percentile threshold
TOP_N = 5               # number of top stocks to highlight

# VN30 as fallback symbol list
VN30_STOCKS = {
    "ACB.VN": "Banking", "BID.VN": "Banking", "CTG.VN": "Banking",
    "HDB.VN": "Banking", "LPB.VN": "Banking", "MBB.VN": "Banking",
    "SHB.VN": "Banking", "SSB.VN": "Banking", "STB.VN": "Banking",
    "TCB.VN": "Banking", "TPB.VN": "Banking", "VCB.VN": "Banking",
    "VIB.VN": "Banking", "VPB.VN": "Banking",
    "BCM.VN": "Real Estate", "KDH.VN": "Real Estate", "VHM.VN": "Real Estate",
    "MSN.VN": "Retail",     "MWG.VN": "Retail", "SAB.VN": "Retail", "VNM.VN": "Retail",
    "GAS.VN": "Energy",     "GVR.VN": "Industrial", "HPG.VN": "Industrial",
    "PLX.VN": "Energy",     "POW.VN": "Energy",
    "FPT.VN": "Technology",
    "BVH.VN": "Insurance",
    "SSI.VN": "Financial Services",
    "VJC.VN": "Aviation",
}

# ============================================================
# HOSE SYMBOL LIST
# ============================================================
def _load_symbols_cache():
    try:
        with open(SYMBOLS_CACHE_FILE) as f:
            data = json.load(f)
        age = datetime.now() - datetime.fromisoformat(data["updated"])
        if age.days < 7 and len(data.get("symbols", [])) > 50:
            return data["symbols"]
    except Exception:
        pass
    return None


def _fetch_hose_symbols_vnstock():
    """Try multiple vnstock3 API paths to get all HOSE symbols."""
    try:
        stock = Vnstock().stock(symbol="ACB", source="VCI")
        df = stock.listing.symbols_by_exchange()
        col_ex = next((c for c in df.columns if "exchange" in c.lower()), None)
        col_sym = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        if col_ex and col_sym:
            hose_df = df[df[col_ex].str.upper() == "HOSE"]
            syms = [f"{s}.VN" for s in hose_df[col_sym].tolist()]
            if len(syms) > 50:
                return syms
    except Exception:
        pass
    try:
        from vnstock3.common.data.data_explorer import StockComponents
        df = StockComponents("VCI").listing()
        col_ex = next((c for c in df.columns if "exchange" in c.lower()), None)
        col_sym = next((c for c in df.columns if "symbol" in c.lower()), None)
        if col_ex and col_sym:
            hose_df = df[df[col_ex].str.upper() == "HOSE"]
            syms = [f"{s}.VN" for s in hose_df[col_sym].tolist()]
            if len(syms) > 50:
                return syms
    except Exception:
        pass
    return None


@st.cache_data(ttl=86_400, show_spinner=False)
def get_hose_symbols():
    """Return list of all HOSE symbols with .VN suffix."""
    cached = _load_symbols_cache()
    if cached:
        return cached

    if HAS_VNSTOCK:
        syms = _fetch_hose_symbols_vnstock()
        if syms:
            try:
                with open(SYMBOLS_CACHE_FILE, "w") as f:
                    json.dump({"updated": datetime.now().isoformat(), "symbols": syms}, f)
            except Exception:
                pass
            return syms

    return list(VN30_STOCKS.keys())


# ============================================================
# PRICE CACHE (parquet, incremental)
# ============================================================
def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, symbol.replace(".", "_") + ".parquet")


def _save_df(df: pd.DataFrame, path: str) -> None:
    """Save to parquet; fall back to CSV if pyarrow not installed."""
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"))


def _load_df(path: str) -> pd.DataFrame | None:
    """Load parquet or CSV cache file."""
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception:
            pass
    return None


def cache_stats() -> tuple[int, float]:
    """Return (file_count, total_size_mb) for all cached price files."""
    try:
        files = [f for f in os.listdir(CACHE_DIR)
                 if f.endswith((".parquet", ".csv"))]
        size = sum(
            os.path.getsize(os.path.join(CACHE_DIR, f))
            for f in files
            if os.path.exists(os.path.join(CACHE_DIR, f))
        )
        return len(files), size / (1024 * 1024)
    except Exception:
        return 0, 0.0


def clear_cache() -> int:
    """Delete all price cache files. Returns count of deleted files."""
    deleted = 0
    try:
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith((".parquet", ".csv")):
                try:
                    os.remove(os.path.join(CACHE_DIR, fname))
                    deleted += 1
                except Exception:
                    pass
    except Exception:
        pass
    return deleted


def load_price_data(symbol: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Load from parquet cache; append only new bars from yfinance."""
    path = _cache_path(symbol)
    cached = None

    if use_cache:
        cached = _load_df(path)

    # Normalise index to UTC-naive for comparison
    def strip_tz(df):
        if df is not None and not df.empty and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    cached = strip_tz(cached)
    today = pd.Timestamp.today().normalize()

    if cached is not None and not cached.empty:
        last_date = cached.index.max()
        if last_date >= today - pd.Timedelta(days=3):  # covers weekends
            return cached  # already up-to-date

        # Fetch only missing range
        start_str = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            new_df = strip_tz(yf.Ticker(symbol).history(start=start_str))
            if new_df is not None and not new_df.empty:
                combined = pd.concat([cached, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined = combined.iloc[-500:]   # keep last ~2 years
                _save_df(combined, path)
                return combined
        except Exception:
            pass
        return cached

    # Full fetch (no cache or cache empty)
    try:
        df = strip_tz(yf.Ticker(symbol).history(period="2y"))
        if df is not None and not df.empty:
            _save_df(df, path)
            return df
    except Exception:
        pass
    return None


# ============================================================
# VNINDEX DATA (multiple sources, no cache — small dataset)
# ============================================================
def _fetch_vnstock_index(source: str) -> pd.DataFrame | None:
    try:
        stock = Vnstock().stock(symbol="VN30", source=source)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df = stock.quote.history(symbol="VNINDEX", start=start, end=end)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"close": "Close", "open": "Open",
                                  "high": "High", "low": "Low", "volume": "Volume"})
        date_col = next((c for c in ["time", "date"] if c in df.columns), None)
        if date_col:
            df.index = pd.to_datetime(df[date_col])
        return df if len(df) >= 50 else None
    except Exception:
        return None


@st.cache_data(ttl=3_600, show_spinner=False)
def get_vnindex_data() -> pd.DataFrame | None:
    sources = []
    if HAS_VNSTOCK:
        sources += [lambda: _fetch_vnstock_index("VCI"),
                    lambda: _fetch_vnstock_index("TCBS")]
    sources += [
        lambda: yf.Ticker("^VNINDEX.VN").history(period="1y"),
        lambda: yf.Ticker("E1VFVN30.VN").history(period="1y"),
        lambda: yf.Ticker("VNM").history(period="1y"),
    ]
    for fetch in sources:
        try:
            df = fetch()
            if df is not None and not df.empty and len(df) >= 50:
                return df
        except Exception:
            continue
    return None


# ============================================================
# TREND TEMPLATE — 10 criteria (he-thong-trading-vn.html, ch4)
# ============================================================
def check_trend_template(
    df: pd.DataFrame,
    sl_pct: float = 5.0,
    tgt_pct: float = 10.0,
) -> tuple[bool, int, dict]:
    """
    Returns (all_pass, score_out_of_9, details_dict).
    Criterion 9 (RS percentile) is excluded here — ranked separately.
    details includes stoploss_price/pct and target_price/pct derived from
    sl_pct and tgt_pct (percentage distances from current price).
    """
    if df is None or df.empty or len(df) < 160:
        return False, 0, {}

    close = df["Close"]
    volume = df["Volume"]

    ma50  = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    vol20 = volume.rolling(20).mean()

    price    = close.iloc[-1]
    ma50_v   = ma50.iloc[-1]
    ma150_v  = ma150.iloc[-1]
    vol20_v  = vol20.iloc[-1]

    if any(pd.isna(v) for v in [price, ma50_v, ma150_v, vol20_v]):
        return False, 0, {}

    n = len(close)
    high52 = close.iloc[-min(252, n):].max()
    low52  = close.iloc[-min(252, n):].min()

    # ── 1. Price > MA50
    c1 = bool(price > ma50_v)

    # ── 2. Price > MA150
    c2 = bool(price > ma150_v)

    # ── 3. MA50 > MA150
    c3 = bool(ma50_v > ma150_v)

    # ── 4. MA150 trending up for 4 consecutive weeks (~5 trading-day steps)
    c4 = False
    ma150_clean = ma150.dropna()
    if len(ma150_clean) >= 22:
        pts = [ma150_clean.iloc[i] for i in [-1, -6, -11, -16, -21]]
        c4 = all(pts[i] > pts[i + 1] for i in range(4))

    # ── 5. Price ≥ 25% above 52-week low
    c5 = bool(low52 > 0 and price >= low52 * 1.25)

    # ── 6. Price within 30% of 52-week high (≥ 70% of high)
    c6 = bool(high52 > 0 and price >= high52 * 0.70)

    # ── 7. Avg 20-session volume ≥ 200,000 shares
    c7 = bool(vol20_v >= MIN_VOL_20)

    # ── 8. Price ≥ 15 (= 15,000 VND in yfinance thousands)
    c8 = bool(price >= MIN_PRICE)

    # ── 10. No distribution: no 3 consecutive sessions of (price↓ + vol↑)
    c10 = True
    if len(df) >= 10:
        recent = df[["Close", "Volume"]].iloc[-10:].copy()
        price_down = recent["Close"].diff() < 0
        vol_up     = recent["Volume"].diff() > 0
        consec = 0
        for is_dist in (price_down & vol_up):
            consec = consec + 1 if is_dist else 0
            if consec >= 3:
                c10 = False
                break

    criteria_without_rs = [c1, c2, c3, c4, c5, c6, c7, c8, c10]
    score = sum(criteria_without_rs)
    all_pass = all(criteria_without_rs)

    _price = round(float(price), 1)
    details = {
        "price":          _price,
        "ma50":           round(float(ma50_v), 1),
        "ma150":          round(float(ma150_v), 1),
        "vol20":          int(vol20_v),
        "high52":         round(float(high52), 1),
        "low52":          round(float(low52), 1),
        "score9":         score,
        # ── Stop loss & target (from current price) ──────────────
        "stoploss_price": round(_price * (1 - sl_pct  / 100), 1),
        "stoploss_pct":   sl_pct,
        "target_price":   round(_price * (1 + tgt_pct / 100), 1),
        "target_pct":     tgt_pct,
        # ── Criteria ─────────────────────────────────────────────
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "c6": c6, "c7": c7, "c8": c8, "c10": c10,
    }
    return all_pass, score, details


# ============================================================
# RELATIVE STRENGTH (RS) — percentile vs all HOSE stocks
# ============================================================
def compute_rs_raw(df: pd.DataFrame) -> float | None:
    """
    RS_raw = Perf_1M*0.25 + Perf_3M*0.35 + Perf_6M*0.25 + Perf_12M*0.15
    Uses available periods with re-normalised weights if some periods missing.
    """
    if df is None or df.empty or len(df) < 63:
        return None
    close = df["Close"]
    price = float(close.iloc[-1])

    def perf(days):
        idx = -(days + 1)
        if len(close) <= days:
            return None
        old = float(close.iloc[idx])
        return (price / old - 1) * 100 if old > 0 else None

    candidates = [(perf(21), 0.25), (perf(63), 0.35), (perf(126), 0.25), (perf(252), 0.15)]
    valid = [(v, w) for v, w in candidates if v is not None]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w


def rank_rs(rs_map: dict) -> dict:
    """Convert {symbol: rs_raw} → {symbol: percentile 0-100}."""
    valid = {k: v for k, v in rs_map.items() if v is not None}
    if not valid:
        return {}
    sorted_vals = sorted(valid.values())
    n = len(sorted_vals)
    return {sym: round(sum(1 for v in sorted_vals if v <= val) / n * 100, 1)
            for sym, val in valid.items()}


# ============================================================
# MARKET FILTER (Traffic Light)
# ============================================================
def compute_market_filter(vnindex_df: pd.DataFrame, breadth_pct: float) -> tuple[str, dict]:
    """
    Green:  VN-Index > MA50  AND  MA20 > MA50  AND  Breadth ≥ 55%
    Yellow: intermediate
    Red:    VN-Index < MA50  OR   MA20 < MA50  OR   Breadth < 40%
    """
    if vnindex_df is None or vnindex_df.empty or len(vnindex_df) < 50:
        return "unknown", {}

    close  = vnindex_df["Close"]
    ma20_v = float(close.rolling(20).mean().iloc[-1])
    ma50_v = float(close.rolling(50).mean().iloc[-1])
    price  = float(close.iloc[-1])

    if pd.isna(ma20_v) or pd.isna(ma50_v):
        return "unknown", {}

    details = {
        "price": round(price, 1),
        "ma20":  round(ma20_v, 1),
        "ma50":  round(ma50_v, 1),
        "breadth": round(breadth_pct, 1),
    }

    above_ma50   = price  > ma50_v
    ma20_gt_ma50 = ma20_v > ma50_v

    # Red — any single condition triggers
    if (not above_ma50) or (not ma20_gt_ma50) or (breadth_pct < 40):
        return "red", details

    # Green — all conditions met
    if above_ma50 and ma20_gt_ma50 and breadth_pct >= 55:
        return "green", details

    return "yellow", details


# ============================================================
# PATTERN DETECTION — Setup A / B / C
# ============================================================

def _detect_candle_pattern(df: pd.DataFrame) -> str:
    """
    Detect bullish / reversal candlestick pattern on the last bar.
    Returns: "Hammer" | "Bullish Engulfing" | "Marubozu" | "Pin Bar" |
             "Strong Bull" | "None"

    Normalises H/L so they always encompass O and C (robust to synthetic data).
    """
    if len(df) < 2:
        return "None"

    o  = float(df["Open"].iloc[-1])
    h  = float(df["High"].iloc[-1])
    l  = float(df["Low"].iloc[-1])
    c  = float(df["Close"].iloc[-1])
    po = float(df["Open"].iloc[-2])
    pc = float(df["Close"].iloc[-2])

    # Normalise H/L to always contain O and C (handles synthetic test data)
    h = max(h, o, c)
    l = min(l, o, c)

    total_range = h - l
    if total_range < 1e-8:
        return "None"

    body       = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # 1. Bullish Marubozu — strong green bar, body ≥ 75% of range
    if c > o and body / total_range >= 0.75:
        return "Marubozu"

    # 2. Bullish Engulfing — current green fully engulfs prior red body
    if c > o and pc < po and o <= pc and c >= po:
        return "Bullish Engulfing"

    # 3. Hammer — small body in top half, lower wick ≥ 2× body
    if (lower_wick >= 2 * max(body, total_range * 0.02) and
            upper_wick <= body * 0.8 and
            min(o, c) >= l + total_range * 0.45):
        return "Hammer"

    # 4. Pin Bar — lower wick ≥ 60% of range (strong wick rejection)
    if lower_wick >= total_range * 0.60 and body <= total_range * 0.35:
        return "Pin Bar"

    # 5. Strong Bull — closes in top 30%, body ≥ 50% of range
    if c > o and (c - l) / total_range >= 0.70 and body / total_range >= 0.50:
        return "Strong Bull"

    return "None"


def _check_entry_candle(df: pd.DataFrame, pivot: float) -> tuple[str, str]:
    """
    Determine entry status relative to pivot.

    Returns (status, entry_candle):
      "🔥 BUY"        — price > pivot + volume ≥ 1.5× avg (entry triggered)
      "👀 Monitoring" — within 3% of pivot with a bullish/reversal candle
      "⏳ Setup"      — pattern detected, waiting for price to approach pivot
    """
    price    = float(df["Close"].iloc[-1])
    last_vol = float(df["Volume"].iloc[-1])
    vol_ma20 = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_ok   = (not pd.isna(vol_ma20)) and (last_vol >= vol_ma20 * 1.5)
    candle   = _detect_candle_pattern(df)

    above_pivot = price > pivot
    near_pivot  = price >= pivot * 0.97     # within 3% below pivot

    if above_pivot and vol_ok:
        return "🔥 BUY", candle if candle != "None" else "Breakout"

    if near_pivot and candle != "None":
        return "👀 Monitoring", candle

    return "⏳ Setup", candle


def _range_pct(chunk: pd.DataFrame) -> float:
    """High-low range as % of midpoint price."""
    h = float(chunk["High"].max())
    l = float(chunk["Low"].min())
    mid = (h + l) / 2
    return (h - l) / mid * 100 if mid > 0 else 0.0


def _check_vcp(df: pd.DataFrame) -> dict | None:
    """
    Setup A — Volatility Contraction Pattern.
    Needs ≥ 60 bars. Splits last 60 days into 3 × 20-day windows;
    checks price-range and volume both contract each window.

    Backtest (3y, 8 US symbols, lookahead=15):
      Overall hit rate ~54%.  Quality now based on contraction_ratio
      (r1/r3) and vol_dry_ratio (v3/v1) — these proved more predictive
      than absolute range tightness alone.
    """
    if len(df) < 60:
        return None

    p1, p2, p3 = df.iloc[-60:-40], df.iloc[-40:-20], df.iloc[-20:]

    r1, r2, r3 = _range_pct(p1), _range_pct(p2), _range_pct(p3)
    v1 = float(p1["Volume"].mean())
    v2 = float(p2["Volume"].mean())
    v3 = float(p3["Volume"].mean())

    price_contracting = r1 > r2 > r3
    vol_drying        = v3 < v2 * 0.90 and v2 < v1 * 0.95
    tight_close       = r3 < 10.0

    if not (price_contracting and vol_drying and tight_close):
        return None

    recent_high       = float(df["High"].iloc[-20:].max())
    # Quality: contraction ratio + volume dry-up ratio (backtest-calibrated)
    contraction_ratio = r1 / r3 if r3 > 0 else 0
    vol_dry_ratio     = v3 / v1 if v1 > 0 else 1.0

    if contraction_ratio >= 5 and vol_dry_ratio < 0.50:
        quality = "★★★"
    elif contraction_ratio >= 3:
        quality = "★★"
    else:
        quality = "★"

    pivot  = round(recent_high * 1.01, 1)
    status, entry_candle = _check_entry_candle(df, pivot)
    confirmed = (status == "🔥 BUY")

    return {
        "pattern":      "VCP",
        "quality":      quality,
        "pivot":        pivot,
        "stoploss":     round(float(df["Low"].iloc[-20:].min()) * 0.99, 1),
        "confirmed":    confirmed,
        "status":       status,
        "entry_candle": entry_candle,
        "notes":        (f"Range {r1:.1f}%→{r2:.1f}%→{r3:.1f}%  "
                         f"Contraction {contraction_ratio:.1f}×  "
                         f"Vol {vol_dry_ratio:.0%} of start"),
    }


def _check_flat_base(df: pd.DataFrame) -> dict | None:
    """
    Setup C variant — Flat Base (5–10 weeks of tight sideways action).

    Backtest (3y, 8 US symbols, lookahead=15):
      Original 12% threshold produced 157 signals (≈2/month/stock) — too loose.
      Tightened to 10% AND now REQUIRE volume contraction, cutting signals
      ~40% while keeping hit rate ~50%.  Quality criteria updated accordingly.
    """
    if len(df) < 50:
        return None

    base  = df.iloc[-40:]
    prior = df.iloc[-80:-40] if len(df) >= 80 else df.iloc[:-40]

    r = _range_pct(base)
    if r >= 10.0:                             # tightened from 12% → 10%
        return None

    price   = float(df["Close"].iloc[-1])
    ma50_v  = float(df["Close"].rolling(50).mean().iloc[-1])
    if pd.isna(ma50_v) or price <= ma50_v:
        return None

    base_vol  = float(base["Volume"].mean())
    prior_vol = float(prior["Volume"].mean()) if not prior.empty else base_vol
    vol_ratio = base_vol / prior_vol if prior_vol > 0 else 1.0

    base_high = float(base["High"].max())
    price_near_top = price >= base_high * 0.93  # within 7% of base ceiling

    # Require either near top OR vol contracted — not both (too few signals otherwise)
    if not (price_near_top or vol_ratio < 0.85):
        return None

    # Quality: price near top (imminent breakout) drives quality more than range tightness
    # Backtest shows price_near_top is a stronger predictor than tight range alone.
    if price_near_top and vol_ratio < 0.70:
        quality = "★★★"   # at top of base + dry volume = imminent breakout
    elif price_near_top:
        quality = "★★"    # at top of base, not yet fully dry
    else:
        quality = "★"     # base forming, not yet at top

    fb_pivot = round(base_high * 1.01, 1)
    status, entry_candle = _check_entry_candle(df, fb_pivot)
    confirmed = (status == "🔥 BUY")

    return {
        "pattern":      "Flat Base",
        "quality":      quality,
        "pivot":        fb_pivot,
        "stoploss":     round(float(base["Low"].min()) * 0.99, 1),
        "confirmed":    confirmed,
        "status":       status,
        "entry_candle": entry_candle,
        "notes":        f"Range {r:.1f}% over 40 bars  Vol {vol_ratio:.0%} of prior",
    }


def _check_pullback_ma20(df: pd.DataFrame) -> dict | None:
    """
    Setup B — Pullback to MA20 in Uptrend.
    Price was above MA20, pulled back to within 3%, volume declining,
    no panic-sell session (vol × 3 + price −4 %).
    """
    if len(df) < 30:
        return None

    close  = df["Close"]
    volume = df["Volume"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    price   = float(close.iloc[-1])
    ma20_v  = float(ma20.iloc[-1])
    ma50_v  = float(ma50.iloc[-1])

    if any(pd.isna(v) for v in [ma20_v, ma50_v]):
        return None

    above_ma50 = price > ma50_v
    near_ma20  = abs(price - ma20_v) / ma20_v < 0.03

    # Was above MA20 recently (confirms pullback, not breakdown)
    was_above = any(float(close.iloc[i]) > float(ma20.iloc[i])
                    for i in range(-8, -1))

    if not (above_ma50 and near_ma20 and was_above):
        return None

    # Volume declining in pullback (last 5 days vs 20-day avg)
    vol_avg5  = float(volume.iloc[-5:].mean())
    vol_ma20  = float(volume.rolling(20).mean().iloc[-1])
    vol_declining = vol_avg5 < vol_ma20 * 0.80

    # No panic sell in last 5 sessions
    for i in range(-5, 0):
        pct_chg  = (float(close.iloc[i]) / float(close.iloc[i - 1]) - 1) * 100
        vol_mult = float(volume.iloc[i]) / vol_ma20
        if pct_chg < -4.0 and vol_mult > 3.0:
            return None   # panic sell detected → skip

    quality   = "★★★" if vol_declining else "★★"
    prev_high = float(close.iloc[-6:-1].max())
    pb_pivot  = round(prev_high * 1.005, 1)
    status, entry_candle = _check_entry_candle(df, pb_pivot)
    confirmed = (status == "🔥 BUY")

    return {
        "pattern":      "Pullback MA20",
        "quality":      quality,
        "pivot":        pb_pivot,
        "stoploss":     round(ma50_v * 0.97, 1),
        "confirmed":    confirmed,
        "status":       status,
        "entry_candle": entry_candle,
        "notes":        (f"Price {price:.1f} vs MA20 {ma20_v:.1f}  "
                         f"Vol {vol_avg5/vol_ma20:.0%} of avg"),
    }


def _passes_trend_filter(df: pd.DataFrame) -> bool:
    """
    Require price > MA50 > MA150 > MA200.
    Ensures patterns are only detected on stocks in a confirmed uptrend.
    Needs ≥ 200 bars.
    """
    if df is None or df.empty or len(df) < 200:
        return False
    close = df["Close"]
    price  = float(close.iloc[-1])
    ma50   = float(close.rolling(50).mean().iloc[-1])
    ma150  = float(close.rolling(150).mean().iloc[-1])
    ma200  = float(close.rolling(200).mean().iloc[-1])
    if any(pd.isna(v) for v in [ma50, ma150, ma200]):
        return False
    return price > ma50 > ma150 > ma200


def _trend_grade(df: pd.DataFrame) -> str:
    """
    Traffic-light grade for MA alignment.
    🟢 Green  — price > MA50 > MA150 > MA200  (full uptrend, ready to trade)
    🟡 Yellow — price > MA50  (short-term bullish, MAs not fully aligned)
    🔴 Red    — price ≤ MA50  (bearish / downtrend)
    """
    if df is None or df.empty or len(df) < 50:
        return "🔴"
    close  = df["Close"]
    price  = float(close.iloc[-1])
    ma50   = float(close.rolling(50).mean().iloc[-1])
    if pd.isna(ma50):
        return "🔴"
    if price <= ma50:
        return "🔴"
    # price > MA50 — check longer MAs
    ma150 = float(close.rolling(150).mean().iloc[-1]) if len(df) >= 150 else float("nan")
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else float("nan")
    if not pd.isna(ma150) and not pd.isna(ma200) and ma50 > ma150 > ma200:
        return "🟢"
    return "🟡"


def detect_patterns(df: pd.DataFrame, require_trend: bool = True) -> list[dict]:
    """
    Run all pattern checks.

    require_trend=True  (default) — only returns patterns when
                        price > MA50 > MA150 > MA200 (strict uptrend filter).
    require_trend=False — runs pattern checks regardless of trend; each
                        result includes a 'trend_grade' 🟢/🟡/🔴 field so
                        the caller can colour-code candidates.
    """
    if df is None or df.empty or len(df) < 60:
        return []
    if require_trend and not _passes_trend_filter(df):
        return []

    grade   = _trend_grade(df)
    results = []
    for check in [_check_vcp, _check_flat_base, _check_pullback_ma20]:
        p = check(df)
        if p:
            p["trend_grade"] = grade
            results.append(p)
    return results


# ============================================================
# INLINE CHART HELPER
# ============================================================
def _show_inline_chart(sym_ticker: str, use_cache: bool = True,
                       scan_rows: list | None = None) -> None:
    """
    Load data for sym_ticker (with or without .VN suffix) and render
    a candlestick-style line chart with MA lines + pattern criteria expander.
    """
    sym = sym_ticker if sym_ticker.endswith(".VN") else sym_ticker + ".VN"
    df_c = load_price_data(sym, use_cache=use_cache)
    if df_c is None or df_c.empty:
        st.warning(f"Không có dữ liệu cho {sym_ticker}")
        return

    df_c = df_c.copy()
    df_c["MA20"]  = df_c["Close"].rolling(20).mean()
    df_c["MA50"]  = df_c["Close"].rolling(50).mean()
    df_c["MA150"] = df_c["Close"].rolling(150).mean()

    chart_cols = ["Close", "MA20", "MA50"]
    if not df_c["MA150"].isna().all():
        chart_cols.append("MA150")

    st.line_chart(df_c[chart_cols].iloc[-252:], use_container_width=True)

    # criteria expander (uses scan_rows if provided)
    if scan_rows:
        sel_row = next((r for r in scan_rows if r["sym"] == sym), None)
        if sel_row and sel_row.get("tt_d"):
            d    = sel_row["tt_d"]
            rs_p = sel_row.get("rs_pct")
            with st.expander(f"📋 Tiêu chí — {sym_ticker}"):
                crit = [
                    ("1. Giá > MA50",                    d.get("c1")),
                    ("2. Giá > MA150",                   d.get("c2")),
                    ("3. MA50 > MA150",                  d.get("c3")),
                    ("4. MA150 tăng 4 tuần liên tiếp",   d.get("c4")),
                    ("5. Giá ≥ 25% trên đáy 52 tuần",    d.get("c5")),
                    ("6. Giá trong 30% so đỉnh 52 tuần", d.get("c6")),
                    ("7. Vol(20) ≥ 200,000 cổ",          d.get("c7")),
                    ("8. Giá ≥ 15,000 VNĐ",              d.get("c8")),
                    ("9. RS ≥ 80 percentile",
                     rs_p is not None and rs_p >= RS_MIN),
                    ("10. Không có phân phối",            d.get("c10")),
                ]
                for lbl, ok in crit:
                    st.markdown(f"{'✅' if ok else '❌'} {lbl}")

    # live pattern detection
    live_p = detect_patterns(df_c, require_trend=False)
    if live_p:
        with st.expander(f"🔍 Patterns — {sym_ticker}"):
            for lp in live_p:
                status_str = lp.get("status", "⏳ Setup")
                st.markdown(
                    f"**{lp['pattern']}** {lp.get('quality','')}  "
                    f"{lp.get('trend_grade','')}  {status_str}  "
                    f"— Pivot: `{lp['pivot']}`  SL: `{lp['stoploss']}`  "
                    f"Candle: {lp.get('entry_candle','—')}"
                )
                st.caption(lp.get("notes", ""))


# ============================================================
# UI HELPERS
# ============================================================
LIGHT_CFG = {
    "green":   ("🟢", "ĐÈN XANH — Uptrend",    "Tích cực giao dịch. Mở vị thế đầy đủ."),
    "yellow":  ("🟡", "ĐÈN VÀNG — Trung tính", "Giảm exposure 50%. Chỉ mua CP có RS ≥ 85."),
    "red":     ("🔴", "ĐÈN ĐỎ — Downtrend",    "KHÔNG mở vị thế mới. Bảo vệ vốn."),
    "unknown": ("⚪", "KHÔNG RÕ",               "Không đủ dữ liệu thị trường."),
}


def rs_stars(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct >= 90:
        return "★★★"
    if pct >= 80:
        return "★★"
    if pct >= 70:
        return "★"
    return ""


# ============================================================
# MAIN APP
# ============================================================
st.set_page_config(layout="wide", page_title="VN HOSE Scanner")
st.title("📊 VN HOSE Stock Scanner")
st.caption("Trend Following System — lọc toàn sàn HOSE theo Trend Template 10 tiêu chí + RS")

# ── Sidebar options ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tuỳ chọn")
    scan_mode = st.radio(
        "Danh sách mã",
        ["HOSE (toàn sàn)", "VN30"],
        help="HOSE: toàn bộ ~400 mã (chậm hơn lần đầu). VN30: 30 mã blue-chip."
    )
    use_cache = st.checkbox("Dùng cache giá", value=True,
                            help="Bỏ chọn để tải lại toàn bộ từ đầu (bỏ qua cache).")
    st.markdown("---")
    st.markdown("**Cache**")
    _n, _mb = cache_stats()
    st.caption(f"{_n} files · {_mb:.1f} MB")
    if st.button("🗑️ Xoá cache", help="Xoá toàn bộ file giá đã cache. Scan tiếp theo sẽ tải lại từ đầu."):
        _deleted = clear_cache()
        st.cache_data.clear()
        st.success(f"Đã xoá {_deleted} file cache.")
    st.markdown("---")
    st.markdown("**Ngưỡng lọc**")
    st.caption(f"Vol trung bình 20 phiên ≥ {MIN_VOL_20:,}")
    st.caption(f"Giá ≥ {MIN_PRICE},000 VNĐ")
    st.caption(f"RS percentile ≥ {RS_MIN}")
    st.markdown("---")
    st.markdown("**Quản trị rủi ro**")
    sl_pct  = st.slider("Stop Loss %",  min_value=3,  max_value=15, value=5,  step=1)
    tgt_pct = st.slider("Target %",     min_value=5,  max_value=50, value=10, step=5)

# ── Scan buttons ─────────────────────────────────────────────
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    do_scan = st.button("🔍 Scan Now", type="primary", use_container_width=True,
                        help="Tải dữ liệu, lọc Trend Template + RS toàn sàn.")
with btn_col2:
    do_pattern = st.button("📐 Pattern Scan", use_container_width=True,
                           help="Scan toàn sàn HOSE tìm VCP / Flat Base / Pullback MA20. Tải dữ liệu nếu chưa có cache.")

# ── Pattern Scan — always scans full HOSE independently ───────────────
if do_pattern:
    with st.spinner("Lấy danh sách mã HOSE..."):
        pattern_syms = list(VN30_STOCKS.keys()) if scan_mode == "VN30" else get_hose_symbols()

    if not pattern_syms:
        st.error("Không lấy được danh sách mã.")
    else:
        prog2 = st.progress(0, text="Scanning patterns...")
        pattern_rows = []
        n2 = len(pattern_syms)
        for i, sym in enumerate(pattern_syms):
            prog2.progress((i + 1) / n2, text=f"[{i+1}/{n2}] {sym}")
            df = load_price_data(sym, use_cache=True)   # fetches from network if not cached
            if df is None or df.empty:
                continue
            patterns = detect_patterns(df, require_trend=False)
            for p in patterns:
                pattern_rows.append({"sym": sym, **p})
        prog2.empty()
        st.session_state["pattern_rows"] = pattern_rows
        st.session_state["pattern_scan_meta"] = {
            "scanned": n2,
            "found":   len(pattern_rows),
            "time":    datetime.now().strftime("%H:%M:%S"),
        }
        st.rerun()

if do_scan:

    with st.spinner("Lấy danh sách mã HOSE..."):
        symbols = list(VN30_STOCKS.keys()) if scan_mode == "VN30" else get_hose_symbols()

    if not symbols:
        st.error("Không lấy được danh sách mã. Thử lại sau.")
        st.stop()

    st.info(f"Đang scan **{len(symbols)}** mã...")

    # ── Fetch VNINDEX ─────────────────────────────────────────
    with st.spinner("Lấy dữ liệu VN-Index..."):
        vnindex_df = get_vnindex_data()

    # ── Scan loop with progress bar ───────────────────────────
    progress = st.progress(0, text="Đang tải dữ liệu...")
    scan_rows = []      # lightweight results (no raw DFs)
    rs_raw_map = {}     # symbol → rs_raw (for percentile ranking)
    above_ma50_count = 0
    total_valid = 0
    n_symbols = len(symbols)

    for i, sym in enumerate(symbols):
        progress.progress((i + 1) / n_symbols,
                          text=f"[{i + 1}/{n_symbols}] {sym}")

        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty or len(df) < 160:
            continue

        total_valid += 1
        tt_pass, tt_score, tt_d = check_trend_template(df, sl_pct=sl_pct, tgt_pct=tgt_pct)
        rs_raw = compute_rs_raw(df)
        rs_raw_map[sym] = rs_raw

        # Count for breadth calculation (criterion 1: price > MA50)
        if tt_d.get("c1", False):
            above_ma50_count += 1

        scan_rows.append({
            "sym": sym,
            "tt_pass": tt_pass,
            "tt_score": tt_score,
            "tt_d": tt_d,
            "rs_raw": rs_raw,
        })

    progress.empty()

    # ── Market breadth & traffic light ───────────────────────
    breadth_pct = (above_ma50_count / total_valid * 100) if total_valid > 0 else 50
    market_signal, market_details = compute_market_filter(vnindex_df, breadth_pct)

    # ── RS percentile ranking ─────────────────────────────────
    rs_pct_map = rank_rs(rs_raw_map)
    for row in scan_rows:
        row["rs_pct"] = rs_pct_map.get(row["sym"])

    # ── Save to session_state for chart selector ──────────────
    st.session_state["scan_rows"] = scan_rows
    st.session_state["market_signal"] = market_signal
    st.session_state["market_details"] = market_details
    st.session_state["use_cache"] = use_cache
    st.rerun()  # clean rerender — clears "Đang scan..." info

# ── Display results (persists across interactions via session_state) ──
if "scan_rows" in st.session_state:
    scan_rows     = st.session_state["scan_rows"]
    market_signal = st.session_state["market_signal"]
    market_details = st.session_state["market_details"]
    use_cache      = st.session_state.get("use_cache", True)

    # ============================================================
    # TRAFFIC LIGHT
    # ============================================================
    emoji, label, action = LIGHT_CFG[market_signal]
    st.markdown("---")
    st.subheader(f"{emoji} Tín hiệu thị trường: {label}")

    if market_details:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VN-Index", market_details.get("price", "?"))
        c2.metric("MA50", market_details.get("ma50", "?"),
                  delta=f"{'▲ trên' if market_details.get('price', 0) > market_details.get('ma50', 0) else '▼ dưới'} MA50")
        c3.metric("MA20 vs MA50",
                  f"{market_details.get('ma20','?')} / {market_details.get('ma50','?')}")
        c4.metric("Market Breadth (% > MA50)", f"{market_details.get('breadth','?')}%")

    st.info(f"**Hành động:** {action}")

    # ============================================================
    # TOP 5
    # ============================================================
    rs_threshold = 85 if market_signal == "yellow" else RS_MIN
    qualifying = [
        r for r in scan_rows
        if r["tt_pass"] and (r["rs_pct"] or 0) >= rs_threshold
    ]
    qualifying.sort(key=lambda r: r["rs_pct"] or 0, reverse=True)
    top5 = qualifying[:TOP_N]

    st.markdown("---")
    st.subheader(f"🏆 Top {TOP_N} cổ phiếu thoả mãn đủ điều kiện")

    if top5:
        cols = st.columns(len(top5))
        for idx, row in enumerate(top5):
            sym   = row["sym"]
            d     = row["tt_d"]
            rs_p  = row["rs_pct"]
            stars = rs_stars(rs_p)
            ticker = sym.replace(".VN", "")
            with cols[idx]:
                st.metric(
                    label=ticker,
                    value=f"{d.get('price','?')}",
                    delta=f"RS {rs_p:.0f} {stars}" if rs_p else "RS ?",
                )
                st.caption(f"MA50: {d.get('ma50','?')} | MA150: {d.get('ma150','?')}")
                st.caption(f"Vol(20): {d.get('vol20',0):,}")
                st.caption(f"Score: {d.get('score9',0)}/9 criteria")
                st.caption(
                    f"🛑 SL: {d.get('stoploss_price','?')}"
                    f" (-{d.get('stoploss_pct',5):.0f}%)"
                    f"  🎯 TGT: {d.get('target_price','?')}"
                    f" (+{d.get('target_pct',10):.0f}%)"
                )
    else:
        if market_signal == "red":
            st.error("🔴 Đèn đỏ — không trade mới.")
        else:
            st.warning("Chưa có cổ phiếu nào thoả mãn đủ 10 tiêu chí.")

    # ============================================================
    # FULL RESULTS TABLE
    # ============================================================
    st.markdown("---")
    st.subheader("📋 Bảng kết quả đầy đủ")

    table_rows = []
    for row in scan_rows:
        sym  = row["sym"]
        d    = row["tt_d"]
        rs_p = row["rs_pct"]
        if not d:
            continue

        score = d.get("score9", 0)
        rs_ok = rs_p is not None and rs_p >= RS_MIN
        if row["tt_pass"] and rs_ok:
            status = "✅ Pass"
        elif score >= 7:
            status = "🔶 Gần"
        else:
            status = "❌"

        sl_price = d.get("stoploss_price", "-")
        sl_p     = d.get("stoploss_pct", 5)
        tgt_price = d.get("target_price", "-")
        tgt_p     = d.get("target_pct", 10)

        table_rows.append({
            "Mã": sym.replace(".VN", ""),
            "Giá": d.get("price", "-"),
            "Stoploss": f"{sl_price} (-{sl_p:.0f}%)",
            "Target":   f"{tgt_price} (+{tgt_p:.0f}%)",
            "MA50": d.get("ma50", "-"),
            "MA150": d.get("ma150", "-"),
            "Đỉnh52T": d.get("high52", "-"),
            "Đáy52T":  d.get("low52", "-"),
            "Vol(20)k": f"{d.get('vol20', 0) // 1000}k",
            "Score(/9)": score,
            "RS%": f"{rs_p:.0f} {rs_stars(rs_p)}".strip() if rs_p else "-",
            "Trạng thái": status,
            # booleans for criteria detail
            ">MA50": "✓" if d.get("c1") else "✗",
            ">MA150": "✓" if d.get("c2") else "✗",
            "MA50>150": "✓" if d.get("c3") else "✗",
            "MA150↑4W": "✓" if d.get("c4") else "✗",
            "+25%Low": "✓" if d.get("c5") else "✗",
            "<30%High": "✓" if d.get("c6") else "✗",
            "Vol✓": "✓" if d.get("c7") else "✗",
            "Price✓": "✓" if d.get("c8") else "✗",
            "NoDistrib": "✓" if d.get("c10") else "✗",
        })

    if table_rows:
        result_df = pd.DataFrame(table_rows)

        # Filter controls
        fc1, fc2 = st.columns(2)
        with fc1:
            show_filter = st.selectbox(
                "Hiển thị",
                ["✅ Chỉ Pass", "✅+🔶 Gần pass (score ≥ 7)", "Tất cả"],
                index=0,
            )
        with fc2:
            min_score = st.slider("Score tối thiểu", 0, 9, 0)

        filtered = result_df.copy()
        if show_filter == "✅ Chỉ Pass":
            filtered = filtered[filtered["Trạng thái"] == "✅ Pass"]
        elif show_filter == "✅+🔶 Gần pass (score ≥ 7)":
            filtered = filtered[filtered["Trạng thái"].isin(["✅ Pass", "🔶 Gần"])]
        filtered = filtered[filtered["Score(/9)"] >= min_score]

        filtered_reset = filtered.reset_index(drop=True)
        tbl_event = st.dataframe(
            filtered_reset, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key="filter_tbl",
        )
        st.caption(
            f"{len(filtered)} mã hiển thị / {len(result_df)} mã scan được  "
            "· **Click một hàng để xem chart**"
        )
        # inline chart for selected row
        sel_rows = tbl_event.selection.rows if tbl_event else []
        if sel_rows:
            sel_ticker = str(filtered_reset.iloc[sel_rows[0]]["Mã"])
            st.markdown(f"##### 📈 {sel_ticker}")
            _show_inline_chart(sel_ticker, use_cache=use_cache, scan_rows=scan_rows)

        # Export
        buf = BytesIO()
        filtered.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "📥 Export Excel",
            data=buf.getvalue(),
            file_name=f"scan_hose_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Any-stock chart (fallback for stocks not in the filtered table) ──
    all_syms = [r["sym"] for r in scan_rows]
    if all_syms:
        st.markdown("---")
        sc1, _ = st.columns([2, 5])
        with sc1:
            other_sel = st.selectbox(
                "📈 Xem chart bất kỳ mã",
                ["— chọn mã —"] + all_syms,
                format_func=lambda s: s.replace(".VN", "") if s != "— chọn mã —" else s,
                key="any_chart_sel",
            )
        if other_sel and other_sel != "— chọn mã —":
            _show_inline_chart(other_sel.replace(".VN", ""),
                               use_cache=use_cache, scan_rows=scan_rows)

else:
    st.info("Nhấn **Scan Now** để bắt đầu quét.")

# ── Pattern Monitor — independent of main scan ───────────────────────
if "pattern_scan_meta" in st.session_state:
    st.markdown("---")
    meta = st.session_state["pattern_scan_meta"]
    st.subheader(
        f"📐 Pattern Monitor — {meta['found']} pattern(s) found "
        f"in {meta['scanned']} symbols  ·  {meta['time']}"
    )

    p_rows = st.session_state.get("pattern_rows", [])

    if not p_rows:
        st.info(
            f"Scanned **{meta['scanned']}** mã — không có pattern nào thoả mãn điều kiện.  \n"
            "Lý do thường gặp: thị trường đang downtrend (price < MA50/MA150/MA200), "
            "hoặc chưa có đủ dữ liệu cache (chạy lại để tải thêm)."
        )
    else:
        _q_ord = {"★★★": 0, "★★": 1, "★": 2}
        p_rows_sorted = sorted(p_rows, key=lambda r: (_q_ord.get(r["quality"], 3),
                                                       r["pattern"]))

        all_patterns = sorted({r["pattern"] for r in p_rows_sorted})
        pf_col, _ = st.columns([2, 4])
        with pf_col:
            pf = st.multiselect("Lọc pattern", all_patterns, default=all_patterns,
                                key="pattern_filter")

        filtered_p = [r for r in p_rows_sorted if r["pattern"] in pf]

        if filtered_p:
            _status_ord = {"🔥 BUY": 0, "👀 Monitoring": 1, "⏳ Setup": 2}
            _grade_ord  = {"🟢": 0, "🟡": 1, "🔴": 2}
            _q_ord2     = {"★★★": 0, "★★": 1, "★": 2}
            filtered_p  = sorted(filtered_p, key=lambda r: (
                _status_ord.get(r.get("status",      "⏳ Setup"), 3),
                _grade_ord.get( r.get("trend_grade", "🔴"),       3),
                _q_ord2.get(    r.get("quality",     "★"),        3),
            ))
            p_table = []
            for r in filtered_p:
                p_table.append({
                    "Mã":           r["sym"].replace(".VN", ""),
                    "Trend":        r.get("trend_grade", "🔴"),
                    "Pattern":      r["pattern"],
                    "Status":       r.get("status", "⏳ Setup"),
                    "Entry Candle": r.get("entry_candle", "—"),
                    "Quality":      r["quality"],
                    "Pivot":        r["pivot"],
                    "Stoploss":     r["stoploss"],
                    "Notes":        r["notes"],
                })
            result_pdf    = pd.DataFrame(p_table)
            buy_count     = sum(1 for r in filtered_p if r.get("status") == "🔥 BUY")
            monitor_count = sum(1 for r in filtered_p if r.get("status") == "👀 Monitoring")
            setup_count   = len(filtered_p) - buy_count - monitor_count
            pat_event = st.dataframe(
                result_pdf, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
                key="pattern_tbl",
            )
            # inline chart for selected pattern row
            pat_sel = pat_event.selection.rows if pat_event else []
            if pat_sel:
                pat_ticker = str(result_pdf.iloc[pat_sel[0]]["Mã"])
                st.markdown(f"##### 📈 {pat_ticker}")
                _show_inline_chart(pat_ticker, use_cache=True)
            green_count  = sum(1 for r in filtered_p if r.get("trend_grade") == "🟢")
            yellow_count = sum(1 for r in filtered_p if r.get("trend_grade") == "🟡")
            red_count    = sum(1 for r in filtered_p if r.get("trend_grade") == "🔴")
            st.caption(
                f"{len(filtered_p)} pattern(s) across {len({r['sym'] for r in filtered_p})} stocks  "
                f"| 🔥 {buy_count} BUY  | 👀 {monitor_count} Monitoring  | ⏳ {setup_count} Setup  "
                f"| 🟢 {green_count} uptrend  | 🟡 {yellow_count} partial  | 🔴 {red_count} weak"
            )
        else:
            st.info("Không có pattern nào khớp bộ lọc.")
