from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from vnstock3 import Vnstock
    HAS_VNSTOCK = True
except ImportError:
    HAS_VNSTOCK = False

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = "./data"
CACHE_DIR   = f"{DATA_PATH}/cache"
BACKTEST_DIR = f"{DATA_PATH}/backtest"
SYMBOLS_CACHE_FILE = f"{DATA_PATH}/hose_symbols.json"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BACKTEST_DIR, exist_ok=True)

BG_COLOR   = "#0a0e17"
CARD_COLOR = "#131829"

# ============================================================
# SYMBOL LISTS
# ============================================================
VN30_STOCKS: dict[str, str] = {
    "ACB.VN": "Banking",          "BID.VN": "Banking",
    "CTG.VN": "Banking",          "HDB.VN": "Banking",
    "LPB.VN": "Banking",          "MBB.VN": "Banking",
    "SHB.VN": "Banking",          "SSB.VN": "Banking",
    "STB.VN": "Banking",          "TCB.VN": "Banking",
    "TPB.VN": "Banking",          "VCB.VN": "Banking",
    "VIB.VN": "Banking",          "VPB.VN": "Banking",
    "BCM.VN": "Real Estate",      "KDH.VN": "Real Estate",
    "VHM.VN": "Real Estate",      "MSN.VN": "Retail",
    "MWG.VN": "Retail",           "SAB.VN": "Retail",
    "VNM.VN": "Retail",           "GAS.VN": "Energy",
    "GVR.VN": "Industrial",       "HPG.VN": "Industrial",
    "PLX.VN": "Energy",           "POW.VN": "Energy",
    "FPT.VN": "Technology",       "BVH.VN": "Insurance",
    "SSI.VN": "Financial Svcs",   "VJC.VN": "Aviation",
}

VNMID_STOCKS: dict[str, str] = {
    "MSB.VN": "Banking",     "OCB.VN": "Banking",     "EIB.VN": "Banking",
    "VIX.VN": "Securities",  "VCI.VN": "Securities",  "HCM.VN": "Securities",
    "FTS.VN": "Securities",  "BSI.VN": "Securities",
    "NLG.VN": "Real Estate", "DIG.VN": "Real Estate", "NVL.VN": "Real Estate",
    "PDR.VN": "Real Estate", "KBC.VN": "Real Estate", "HDG.VN": "Real Estate",
    "DXG.VN": "Real Estate", "AGG.VN": "Real Estate", "SJS.VN": "Real Estate",
    "CII.VN": "Real Estate", "HDC.VN": "Real Estate", "TDH.VN": "Real Estate",
    "GMD.VN": "Logistics",   "HAH.VN": "Logistics",   "PVT.VN": "Logistics",
    "VTP.VN": "Logistics",   "DRC.VN": "Industrial",  "VGC.VN": "Industrial",
    "PHR.VN": "Industrial",  "HSG.VN": "Steel",       "NKG.VN": "Steel",
    "CSV.VN": "Industrial",  "BSR.VN": "Energy",      "PVD.VN": "Energy",
    "NT2.VN": "Energy",      "VSH.VN": "Energy",      "PC1.VN": "Energy",
    "REE.VN": "Energy",      "TBC.VN": "Energy",      "PNJ.VN": "Retail",
    "DGW.VN": "Technology",  "FRT.VN": "Retail",      "VHC.VN": "Seafood",
    "ANV.VN": "Seafood",     "IDI.VN": "Seafood",     "HVN.VN": "Aviation",
    "KDC.VN": "FMCG",        "SBT.VN": "FMCG",        "DBC.VN": "Agriculture",
    "PAN.VN": "Agriculture", "BAF.VN": "Agriculture",
    "CMG.VN": "Technology",  "ELC.VN": "Technology",  "SGT.VN": "Technology",
    "DHG.VN": "Pharma",      "IMP.VN": "Pharma",      "TRA.VN": "Pharma",
    "DMC.VN": "Pharma",      "CTD.VN": "Construction","VCG.VN": "Construction",
    "FCN.VN": "Construction",
}

VN100_STOCKS: dict[str, str] = {**VN30_STOCKS, **VNMID_STOCKS}


# ============================================================
# PRICE CACHE (parquet, incremental)
# ============================================================
def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, symbol.replace(".", "_") + ".parquet")


def _save_df(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.replace(".parquet", ".csv"))


def _load_df(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv = path.replace(".parquet", ".csv")
    if os.path.exists(csv):
        try:
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        except Exception:
            pass
    return None


def cache_stats() -> tuple[int, float]:
    try:
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith((".parquet", ".csv"))]
        size  = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files
                    if os.path.exists(os.path.join(CACHE_DIR, f)))
        return len(files), size / (1024 * 1024)
    except Exception:
        return 0, 0.0


def clear_cache() -> int:
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
    path   = _cache_path(symbol)
    cached = _load_df(path) if use_cache else None

    def strip_tz(df):
        if df is not None and not df.empty and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    cached = strip_tz(cached)
    today  = pd.Timestamp.today().normalize()

    if cached is not None and not cached.empty:
        last_date = cached.index.max()
        if last_date >= today - pd.Timedelta(days=3):
            return cached
        start_str = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            new_df = strip_tz(yf.Ticker(symbol).history(start=start_str))
            if new_df is not None and not new_df.empty:
                combined = pd.concat([cached, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined = combined.iloc[-500:]
                _save_df(combined, path)
                return combined
        except Exception:
            pass
        return cached

    try:
        df = strip_tz(yf.Ticker(symbol).history(period="2y"))
        if df is not None and not df.empty:
            _save_df(df, path)
            return df
    except Exception:
        pass
    return None


# ============================================================
# VNINDEX — market filter
# ============================================================
def _fetch_vnstock_index(source: str) -> pd.DataFrame | None:
    try:
        stock = Vnstock().stock(symbol="VN30", source=source)
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df    = stock.quote.history(symbol="VNINDEX", start=start, end=end)
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


def compute_market_filter(vnindex_df: pd.DataFrame | None) -> tuple[str, dict]:
    if vnindex_df is None or len(vnindex_df) < 55:
        return "⚪ Không có dữ liệu", {}
    close   = vnindex_df["Close"]
    ma20    = close.rolling(20).mean()
    ma50    = close.rolling(50).mean()
    price   = close.iloc[-1]
    ma20_v  = ma20.iloc[-1]
    ma50_v  = ma50.iloc[-1]
    details = {
        "VNINDEX": round(price, 1),
        "MA20":    round(ma20_v, 1),
        "MA50":    round(ma50_v, 1),
    }
    if price > ma50_v and ma20_v > ma50_v:
        label = "🟢 Thị trường thuận lợi"
    elif price < ma50_v or ma20_v < ma50_v:
        label = "🔴 Thị trường bất lợi"
    else:
        label = "🟡 Thị trường trung tính"
    return label, details


# ============================================================
# INDICATORS
# ============================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns used by scan rules.
    Index convention:
      [-1] = signal candle (last row of df)
      shift(2).rolling(N) → covers [-N-1] to [-2], excludes signal candle
    """
    d = df.copy()
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift(1)).abs(),
        (d["Low"]  - d["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    # ATR(10) on [-11] to [-2]
    d["atr10"]        = tr.shift(2).rolling(10).mean()
    # avg volume on [-21] to [-2]
    d["avg_vol20"]    = d["Volume"].shift(2).rolling(20).mean()
    # avg volume on [-6] to [-2]
    d["avg_vol_pre5"] = d["Volume"].shift(2).rolling(5).mean()
    # max high on [-11] to [-2] and [-21] to [-2]
    d["high10"]       = d["High"].shift(2).rolling(10).max()
    d["high20"]       = d["High"].shift(2).rolling(20).max()
    # MAs at signal candle
    d["ma20"]         = d["Close"].rolling(20).mean()
    d["ma50"]         = d["Close"].rolling(50).mean()
    d["ma200"]        = d["Close"].rolling(200).mean()
    # MA from N bars ago (for slope checks)
    d["ma20_prev5"]   = d["ma20"].shift(5)
    d["ma50_prev5"]   = d["ma50"].shift(5)
    d["ma200_prev20"] = d["ma200"].shift(20)
    # Previous candle's high — close-above confirmation
    d["high_prev1"]   = d["High"].shift(1)
    # Candle range for NR7 detection
    d["candle_range"] = d["High"] - d["Low"]
    d["nr7"]          = d["candle_range"] <= d["candle_range"].rolling(7).min()
    # Gap vs prior close
    d["gap_pct"]      = (d["Open"] - d["Close"].shift(1)) / d["Close"].shift(1)
    return d


# ============================================================
# VOLUME TIER
# ============================================================
def _vol_tier(vol: float, avg_vol20: float, avg_vol_pre5: float) -> str:
    if pd.isna(avg_vol20) or avg_vol20 == 0:
        return "NO_VOL_DATA"
    spike    = bool(vol > 2.0 * avg_vol20)
    contract = (not pd.isna(avg_vol_pre5)) and bool(avg_vol_pre5 < avg_vol20 * 0.75)
    if spike and contract:
        return "TIER1"
    if spike:
        return "TIER2"
    return "TIER3"


# ============================================================
# RELATIVE STRENGTH (4-week vs VNINDEX)
# ============================================================
def compute_rs4w(df: pd.DataFrame, vnindex_df: pd.DataFrame | None) -> float | None:
    """
    4-week RS = (stock 21-day return) / (VNINDEX 21-day return).
    > 1.0 → outperforming. > 1.05 → clearly outperforming.
    """
    if vnindex_df is None or len(vnindex_df) < 22 or len(df) < 22:
        return None
    try:
        stock_ret = df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1
        idx_col   = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
        idx_ret   = vnindex_df[idx_col].iloc[-1] / vnindex_df[idx_col].iloc[-22] - 1
        if abs(idx_ret) < 0.001:
            return None
        return round((1 + stock_ret) / (1 + idx_ret), 3)
    except Exception:
        return None


# ============================================================
# NR7-SPECIFIC QUALITY HELPERS
# ============================================================
def check_inside_bar(df: pd.DataFrame) -> tuple[bool, int]:
    """
    Returns (is_ib, ib_chain_len) for the signal candle.
    is_ib: signal candle high < prev high AND low > prev low.
    ib_chain_len: how many consecutive inside bars end at signal candle.
    """
    if df is None or len(df) < 2:
        return False, 0
    chain = 0
    for j in range(len(df) - 1, 0, -1):
        cur  = df.iloc[j]
        prev = df.iloc[j - 1]
        if cur["High"] < prev["High"] and cur["Low"] > prev["Low"]:
            chain += 1
        else:
            break
    is_ib = chain > 0
    return is_ib, chain


def nearest_resistance_atr(df: pd.DataFrame, signal_high: float, atr: float,
                            lookback: int = 60) -> float | None:
    """
    Find the nearest swing-high resistance above signal_high within 4 × ATR.
    Returns distance in ATR units, or None if no resistance found.
    A swing high is: high[i] > high[i-1] AND high[i] > high[i+1].
    """
    if df is None or len(df) < lookback + 2 or atr <= 0:
        return None
    view    = df.iloc[-lookback - 1:-1]
    ceiling = signal_high + 4.0 * atr
    highs   = view["High"].values
    found   = []
    for i in range(1, len(highs) - 1):
        h = highs[i]
        if signal_high < h <= ceiling and h > highs[i - 1] and h > highs[i + 1]:
            found.append(h)
    if not found:
        return None
    nearest = min(found)
    return round((nearest - signal_high) / atr, 2)


def nr7_vol_quality(row: pd.Series, avg_vol20: float) -> tuple[bool, str]:
    """
    NR7 wants LOW volume on the signal candle (market in equilibrium).
    Returns (vol_is_quiet, label).
    """
    if pd.isna(avg_vol20) or avg_vol20 == 0:
        return False, "NO_DATA"
    ratio = row["Volume"] / avg_vol20
    if ratio < 0.50:
        return True,  "QUIET++"   # very low — perfect coil
    if ratio < 0.75:
        return True,  "QUIET"     # low — good
    if ratio < 1.00:
        return False, "NORMAL"
    return False, "HIGH"          # high vol on NR7 day = less ideal


# ============================================================
# SIGNAL QUALITY HELPERS
# ============================================================
def compute_vol_character(df: pd.DataFrame, lookback: int = 15) -> str:
    """
    Compare total volume on up-close vs down-close days over last `lookback` bars
    (excludes signal candle). Returns ACCUM, DISTRIB, or NEUTRAL.
    """
    if df is None or len(df) < lookback + 2:
        return "NEUTRAL"
    view = df.iloc[-lookback - 1:-1]
    up_vol   = view.loc[view["Close"] >= view["Open"], "Volume"].sum()
    down_vol = view.loc[view["Close"] <  view["Open"], "Volume"].sum()
    if down_vol == 0:
        return "ACCUM"
    ratio = up_vol / down_vol
    if ratio >= 1.3:   return "ACCUM"
    if ratio <= 0.75:  return "DISTRIB"
    return "NEUTRAL"


def count_tight_days(df: pd.DataFrame, lookback: int = 20) -> int:
    """Count candles in last `lookback` bars (excl. signal) where range < 0.5 × ATR10."""
    if df is None or len(df) < lookback + 2:
        return 0
    atr = df["atr10"].iloc[-1]
    if pd.isna(atr) or atr == 0:
        return 0
    view   = df.iloc[-lookback - 1:-1]
    ranges = view["High"] - view["Low"]
    return int((ranges < 0.5 * atr).sum())


def check_weekly_trend(df: pd.DataFrame) -> bool:
    """Returns True if weekly close > 20-week MA (resampled from daily data)."""
    if df is None or len(df) < 105:   # ~21 weeks minimum
        return False
    try:
        weekly = df["Close"].resample("W").last().dropna()
        if len(weekly) < 21:
            return False
        return bool(weekly.iloc[-1] > weekly.rolling(20).mean().iloc[-1])
    except Exception:
        return False


def has_overhead_supply(df: pd.DataFrame, breakout_high: float, atr: float,
                        lookback: int = 60) -> bool:
    """
    Returns True if a prior swing high sits between breakout_high and
    breakout_high + 2 × ATR (resistance zone that could stall the move).
    Swing high: high[i] strictly greater than both neighbours.
    """
    if df is None or len(df) < lookback + 2 or atr <= 0:
        return False
    view    = df.iloc[-lookback - 1:-1]
    ceiling = breakout_high + 2.0 * atr
    highs   = view["High"].values
    for i in range(1, len(highs) - 1):
        h = highs[i]
        if breakout_high < h <= ceiling and h > highs[i - 1] and h > highs[i + 1]:
            return True
    return False


def _assign_rs_pct(items: list[dict]) -> None:
    """
    Rank items by rs4w within the list and add rs_pct (0–100 percentile) in-place.
    Mutates dicts directly.
    """
    ranked = [(i, r["rs4w"]) for i, r in enumerate(items) if r.get("rs4w") is not None]
    if len(ranked) < 2:
        return
    ranked.sort(key=lambda x: x[1])
    for pos, (i, _) in enumerate(ranked):
        items[i]["rs_pct"] = round((pos + 1) / len(ranked) * 100)


# ============================================================
# SCAN — Breakout Momentum (Phương Án 1)
# ============================================================
def scan_breakout(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Signal candle = df.iloc[-1].
    Returns BREAKOUT_STRONG (high > HIGH20) or BREAKOUT_EARLY (high > HIGH10 only).
    SIZE relaxed to 1.0× ATR. Body filter replaced with close > high_prev1.
    """
    if df is None or len(df) < 60:
        return None
    row = df.iloc[-1]

    required = ["atr10", "avg_vol20", "high10", "high20", "ma50", "ma50_prev5", "high_prev1"]
    if any(pd.isna(row[c]) for c in required):
        return None

    atr10      = row["atr10"]
    avg_vol20  = row["avg_vol20"]
    high10     = row["high10"]
    high20     = row["high20"]
    ma50       = row["ma50"]
    ma50_prev5 = row["ma50_prev5"]
    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    vol   = row["Volume"]

    # [1] TREND
    if not (close > ma50 and ma50 > ma50_prev5):
        return None
    # [2] Bull candle
    if not (close > open_):
        return None
    # [3] Close near high (≤0.2% below own candle high)
    if not (close >= high * 0.998):
        return None
    # [3b] Close above previous candle's high — confirms clean break
    if not (close > row["high_prev1"]):
        return None
    # [4] Candle range > 1.0× ATR10 (relaxed from 1.5×)
    if not ((high - low) > 1.0 * atr10):
        return None
    # [5] Breakout — must clear at least the 10-day high
    if not (high > high10):
        return None
    # Filter: not over-extended (>8% above MA50)
    if close > ma50 * 1.08:
        return None

    signal_type = "BREAKOUT_STRONG" if high > high20 else "BREAKOUT_EARLY"
    vol_tier    = _vol_tier(vol, avg_vol20, row.get("avg_vol_pre5", float("nan")))
    rs4w        = compute_rs4w(df, vnindex_df)
    sl          = round(low, 2)
    entry       = round(close, 2)
    tp          = round(entry + 2.0 * atr10, 2)
    rr          = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "high10":          round(high10, 2),
        "high20":          round(high20, 2),
        "atr10":           round(atr10, 2),
        "ma50":            round(ma50, 2),
        "ma200":           round(row.get("ma200", float("nan")), 2),
        "sl":              sl,
        "tp":              tp,
        "rr":              rr,
        "vol_tier":        vol_tier,
        "rs4w":            rs4w,
        "volume":          int(vol),
        "avg_vol20":       int(avg_vol20),
        # quality metrics
        "vol_char":        compute_vol_character(df),
        "tight_days":      count_tight_days(df),
        "weekly_ok":       check_weekly_trend(df),
        "supply_overhead": has_overhead_supply(df, high, atr10),
    }


# ============================================================
# SCAN — NR7 Breakout (narrow coil → break)
# ============================================================
def scan_nr7(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    NR7: signal candle has the smallest range of the last 7 days,
    yet closes above the 10-day high. Low-risk entry — tight SL, good R:R.
    """
    if df is None or len(df) < 60:
        return None
    row = df.iloc[-1]

    required = ["atr10", "avg_vol20", "high10", "ma50", "ma50_prev5", "nr7", "candle_range"]
    if any(pd.isna(row[c]) for c in required):
        return None

    atr10      = row["atr10"]
    avg_vol20  = row["avg_vol20"]
    high10     = row["high10"]
    high20     = row.get("high20", float("nan"))
    ma50       = row["ma50"]
    ma50_prev5 = row["ma50_prev5"]
    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    vol   = row["Volume"]

    # [1] Uptrend
    if not (close > ma50 and ma50 > ma50_prev5):
        return None
    # [2] NR7 — narrowest candle of last 7 days (coiling)
    if not row["nr7"]:
        return None
    # [3] Bull candle
    if not (close > open_):
        return None
    # [4] Close breaks above 10-day high
    if not (high > high10):
        return None
    # [5] Close in upper half of candle (buyers in control)
    if not (close >= (high + low) / 2):
        return None
    # Filter: not over-extended
    if close > ma50 * 1.08:
        return None

    # ── NR7 specific quality factors (computed before signal type) ──
    is_ib, ib_chain          = check_inside_bar(df)
    resist_atr               = nearest_resistance_atr(df, high, atr10)
    vol_quiet, vol_quiet_lbl = nr7_vol_quality(row, avg_vol20)
    avg_vol_pre5             = row.get("avg_vol_pre5", float("nan"))
    vol_declining            = (not pd.isna(avg_vol_pre5)) and bool(avg_vol_pre5 < avg_vol20 * 0.80)

    # ── Filter 1: Reject HIGH volume on NR7 day ──
    # High volume = contested candle, not quiet compression → 12% historical WR
    if vol_quiet_lbl == "HIGH":
        return None

    signal_type = "NR7_STRONG" if (not pd.isna(high20) and high > high20) else "NR7_EARLY"

    # Composite NR7 setup score (0–100)
    nr7_score = 0
    if is_ib:                                    nr7_score += 25
    if ib_chain >= 2:                            nr7_score += 15
    if resist_atr is not None:
        if resist_atr <= 1.0:                    nr7_score += 25
        elif resist_atr <= 2.0:                  nr7_score += 15
        elif resist_atr <= 3.0:                  nr7_score += 8
    if vol_quiet_lbl == "QUIET++":               nr7_score += 25
    elif vol_quiet_lbl == "QUIET":               nr7_score += 15
    if vol_declining:                            nr7_score += 10

    # ── Filter 2: Drop NR7_EARLY unless score is high enough ──
    # NR7_EARLY historical WR = 17% — only keep if score suggests strong compression
    if signal_type == "NR7_EARLY" and nr7_score < 50:
        return None

    # ── Filter 3: Minimum score gate ──
    # score < 30 → setup lacks enough quality factors → 33% WR not worth it
    NR7_MIN_SCORE = 30
    if nr7_score < NR7_MIN_SCORE:
        return None

    vol_tier = _vol_tier(vol, avg_vol20, avg_vol_pre5)
    rs4w     = compute_rs4w(df, vnindex_df)
    sl       = round(low, 2)
    entry    = round(close, 2)
    tp       = round(entry + 2.0 * atr10, 2)
    rr       = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "high10":          round(high10, 2),
        "high20":          round(high20, 2) if not pd.isna(high20) else None,
        "atr10":           round(atr10, 2),
        "ma50":            round(ma50, 2),
        # NR7 quality
        "is_inside_bar":   is_ib,
        "ib_chain":        ib_chain,
        "resist_atr":      resist_atr,
        "vol_quiet":       vol_quiet_lbl,
        "nr7_score":       nr7_score,
        "ma200":           round(row.get("ma200", float("nan")), 2),
        "sl":              sl,
        "tp":              tp,
        "rr":              rr,
        "vol_tier":        vol_tier,
        "vol_char":        compute_vol_character(df),
        "tight_days":      count_tight_days(df),
        "weekly_ok":       check_weekly_trend(df),
        "supply_overhead": has_overhead_supply(df, high, atr10),
        "rs4w":      rs4w,
        "volume":    int(vol),
        "avg_vol20": int(avg_vol20),
    }


# ============================================================
# SCAN — Gap-Up Breakout (institutional conviction)
# ============================================================
def scan_gap(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Gap-up ≥ 0.5% above prior close, holds the gap (no fade),
    and closes above the 10-day high. Gaps on volume = institutional demand.
    """
    if df is None or len(df) < 60:
        return None
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    required = ["atr10", "avg_vol20", "high10", "ma50", "ma50_prev5", "gap_pct"]
    if any(pd.isna(row[c]) for c in required):
        return None

    atr10      = row["atr10"]
    avg_vol20  = row["avg_vol20"]
    high10     = row["high10"]
    high20     = row.get("high20", float("nan"))
    ma50       = row["ma50"]
    ma50_prev5 = row["ma50_prev5"]
    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    vol   = row["Volume"]
    gap_pct = row["gap_pct"]

    # [1] Uptrend
    if not (close > ma50 and ma50 > ma50_prev5):
        return None
    # [2] Gap up ≥ 0.5% vs prior close
    if not (gap_pct >= 0.005):
        return None
    # [3] Gap holds — close above prior close (no full fade)
    if not (close > prev["Close"]):
        return None
    # [4] Bull candle
    if not (close > open_):
        return None
    # [5] Breakout above 10-day high
    if not (high > high10):
        return None
    # Filter: not over-extended
    if close > ma50 * 1.08:
        return None

    signal_type = "GAP_STRONG" if (not pd.isna(high20) and high > high20) else "GAP_EARLY"
    vol_tier    = _vol_tier(vol, avg_vol20, row.get("avg_vol_pre5", float("nan")))
    rs4w        = compute_rs4w(df, vnindex_df)
    sl          = round(low, 2)
    entry       = round(close, 2)
    tp          = round(entry + 2.0 * atr10, 2)
    rr          = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "gap_pct":         round(gap_pct * 100, 2),
        "high10":          round(high10, 2),
        "high20":          round(high20, 2) if not pd.isna(high20) else None,
        "atr10":           round(atr10, 2),
        "ma50":            round(ma50, 2),
        "ma200":           round(row.get("ma200", float("nan")), 2),
        "sl":              sl,
        "tp":              tp,
        "rr":              rr,
        "vol_char":        compute_vol_character(df),
        "tight_days":      count_tight_days(df),
        "weekly_ok":       check_weekly_trend(df),
        "supply_overhead": has_overhead_supply(df, high, atr10),
        "vol_tier":  vol_tier,
        "rs4w":      rs4w,
        "volume":    int(vol),
        "avg_vol20": int(avg_vol20),
    }


# ============================================================
# SCAN — Trend Filter (Pullback to MA in uptrend)
# ============================================================
MIN_AVG_VOLUME_TF = 100_000   # minimum liquidity gate

def scan_trend_filter(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Trend Filter — two signal types inside a strong uptrend:

    TF_MA20  Shallower pullback, stronger trend, vol ≥ 1.2× avg, close > HIGH10
    TF_MA50  Deeper pullback, early-trend only, vol ≥ 1.5× avg, close > close[-5]

    Shared rules (1-8, 10-17):
      1. Low > MA200
      2. Close > MA200
      3. MA20 > MA50 > MA200
      4. MA20 > MA20[5]
      5. MA50 > MA50[5]
      6. MA200 > MA200[20]
      7. Close > Open
     10. (MA20-MA200)/MA200 ≤ 12%
     11. (MA50-MA200)/MA200 ≤ 8%
     12. (Close-MA20)/MA20 ≤ 5%
     13. (Close-MA50)/MA50 ≤ 8%
     14. AvgVolume(20) ≥ 100,000
     16. (High-Low) > 0.8 × ATR10
     17. RS4W ≥ 0.95

    TF_MA20 specific:
      8a. Close > HIGH10           (breaks 10-day high)
      9a. Low ≤ MA20 × 1.03       (tested MA20)
     15a. Volume > 1.2 × avg_vol20

    TF_MA50 specific:
      8b. Close > close[-5]        (5-day momentum, less demanding than HIGH10)
      9b. Low ≤ MA50 × 1.03 AND Close > MA50
     11b. (MA50-MA200)/MA200 ≤ 0.05  (early uptrend only)
     15b. Volume > 1.5 × avg_vol20   (needs stronger buying to recover from deep pullback)

    SL = min(Low, touched_MA) × 0.995
    """
    if df is None or len(df) < 210:
        return None
    row = df.iloc[-1]

    required = ["atr10", "avg_vol20", "ma20", "ma50", "ma200",
                "ma20_prev5", "ma50_prev5", "ma200_prev20", "high10"]
    if any(pd.isna(row[c]) for c in required):
        return None

    ma20         = row["ma20"]
    ma50         = row["ma50"]
    ma200        = row["ma200"]
    ma20_prev5   = row["ma20_prev5"]
    ma50_prev5   = row["ma50_prev5"]
    ma200_prev20 = row["ma200_prev20"]
    high10       = row["high10"]
    atr10        = row["atr10"]
    avg_vol20    = row["avg_vol20"]
    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    vol   = row["Volume"]

    # ── Shared rules 1-7 ──────────────────────────────────────────
    if not (low > ma200):           return None   # [1]
    if not (close > ma200):         return None   # [2]
    if not (ma20 > ma50 > ma200):   return None   # [3]
    if not (ma20 > ma20_prev5):     return None   # [4]
    if not (ma50 > ma50_prev5):     return None   # [5]
    if not (ma200 > ma200_prev20):  return None   # [6]
    if not (close > open_):         return None   # [7]

    # ── Shared rules 10-17 ────────────────────────────────────────
    if (ma20 - ma200) / ma200 > 0.12:  return None   # [10]
    if (close - ma20) / ma20  > 0.05:  return None   # [12]
    if (close - ma50) / ma50  > 0.08:  return None   # [13]
    if avg_vol20 < MIN_AVG_VOLUME_TF:  return None   # [14]
    if (high - low) < 0.8 * atr10:    return None   # [16]

    vol_tier = _vol_tier(vol, avg_vol20, row.get("avg_vol_pre5", float("nan")))
    rs4w     = compute_rs4w(df, vnindex_df)
    if rs4w is not None and rs4w < 0.95:  return None   # [17]

    # ── Try TF_MA20 first (higher quality) ───────────────────────
    if (low <= ma20 * 1.03             # [9a] tested MA20
            and close > high10         # [8a] breaks 10-day high
            and vol >= 1.2 * avg_vol20  # [15a]
            and (ma50 - ma200) / ma200 <= 0.08):   # [11]
        signal_type = "TF_MA20"
        touched_ma  = ma20

    # ── Try TF_MA50 (deeper pullback, stricter gates) ─────────────
    elif (low <= ma50 * 1.03           # [9b] tested MA50
            and close > ma50           # [9b] closed back above MA50
            and (ma50 - ma200) / ma200 <= 0.05     # [11b] early uptrend only
            and vol >= 1.5 * avg_vol20  # [15b] stronger vol required
            and len(df) >= 6 and close > df.iloc[-6]["Close"]):  # [8b] 5-day momentum
        signal_type = "TF_MA50"
        touched_ma  = ma50

    else:
        return None

    sl_raw = min(low, touched_ma) * 0.995
    sl     = round(sl_raw, 2)
    entry  = round(close, 2)
    tp     = round(entry + 2.0 * atr10, 2)
    rr     = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "ma20":            round(ma20, 2),
        "ma50":            round(ma50, 2),
        "ma200":           round(ma200, 2),
        "high10":          round(high10, 2),
        "atr10":           round(atr10, 2),
        "touched_ma":      "MA20" if signal_type == "TF_MA20" else "MA50",
        "sl":              sl,
        "tp":              tp,
        "rr":              rr,
        "vol_tier":        vol_tier,
        "rs4w":            rs4w,
        "volume":          int(vol),
        "avg_vol20":       int(avg_vol20),
        "vol_char":        compute_vol_character(df),
        "weekly_ok":       check_weekly_trend(df),
        "supply_overhead": has_overhead_supply(df, high, atr10),
    }


# ============================================================
# DEVELOPING SETUP SCORER (watchlist — not yet triggered)
# ============================================================
def score_developing(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Score how close a stock is to triggering a signal.
    Covers two scenarios:
      - BREAKOUT developing: uptrend, approaching HIGH10/HIGH20
      - REVERSAL developing: downtrend, near MA200 or approaching MA50 reclaim
    Returns a scored dict (0–100) or None if not interesting.
    """
    if df is None or len(df) < 60:
        return None
    row = df.iloc[-1]

    required = ["atr10", "avg_vol20", "high10", "high20", "ma50", "ma50_prev5", "avg_vol_pre5"]
    if any(pd.isna(row[c]) for c in required):
        return None

    close        = row["Close"]
    ma50         = row["ma50"]
    ma50_prev5   = row["ma50_prev5"]
    high10       = row["high10"]
    high20       = row["high20"]
    atr10        = row["atr10"]
    avg_vol20    = row["avg_vol20"]
    avg_vol_pre5 = row["avg_vol_pre5"]
    ma200        = row.get("ma200", float("nan"))
    vol          = row["Volume"]
    rs4w         = compute_rs4w(df, vnindex_df)

    vol_ratio = avg_vol_pre5 / avg_vol20 if avg_vol20 > 0 else 1.0
    vol_tier  = _vol_tier(vol, avg_vol20, avg_vol_pre5)

    # ── Scenario A: BREAKOUT developing (uptrend, approaching high) ──
    if close > ma50 and not (close > high10) and close <= ma50 * 1.08:
        pct_from_high10 = (high10 - close) / close * 100
        pct_from_high20 = (high20 - close) / close * 100
        if pct_from_high10 > 8.0:
            return None

        score = 0
        notes = []
        score += 20 if ma50 > ma50_prev5 else 5
        if ma50 > ma50_prev5: notes.append("MA50↑")

        if pct_from_high10 <= 2.0:   score += 40; notes.append(f"{pct_from_high10:.1f}% to H10")
        elif pct_from_high10 <= 4.0: score += 28; notes.append(f"{pct_from_high10:.1f}% to H10")
        elif pct_from_high10 <= 6.0: score += 15; notes.append(f"{pct_from_high10:.1f}% to H10")
        else:                        score += 5;  notes.append(f"{pct_from_high10:.1f}% to H10")

        if vol_ratio < 0.70:   score += 25; notes.append("vol coil")
        elif vol_ratio < 0.80: score += 15; notes.append("vol↓")
        elif vol_ratio < 0.90: score += 8

        if rs4w and rs4w >= 1.10:  score += 15; notes.append(f"RS {rs4w:.2f}")
        elif rs4w and rs4w >= 1.05: score += 10; notes.append(f"RS {rs4w:.2f}")

        if score < 35:
            return None
        dev_type = "→ STRONG" if pct_from_high20 <= 3.0 else "→ EARLY"

        explain_parts = [
            f"✅ Uptrend: giá ({close:,.0f}) trên MA50 ({ma50:,.0f})" +
            (" — MA50 đang dốc lên." if ma50 > ma50_prev5 else "."),
            f"📍 Cách đỉnh {10 if pct_from_high10 <= 8 else 20} ngày {pct_from_high10:.1f}% "
            f"(HIGH10={high10:,.0f}). Chưa phá — chờ nến breakout.",
        ]
        if vol_ratio < 0.80:
            explain_parts.append(
                f"🔇 Volume 5 ngày gần nhất ({vol_ratio*100:.0f}% baseline) — "
                "thị trường đang tích lũy lặng lẽ, bên bán yếu dần."
            )
        if rs4w and rs4w >= 1.05:
            explain_parts.append(
                f"💪 RS4W = {rs4w:.2f} — cổ phiếu mạnh hơn thị trường 4 tuần qua."
            )
        explain_parts.append(
            "⏳ Trigger khi: nến ngày đóng > đỉnh 10/20 ngày + volume tăng."
        )

        return {
            "signal": "WATCH", "dev_type": dev_type, "score": score,
            "notes": ", ".join(notes), "explain": "\n".join(explain_parts),
            "date": df.index[-1],
            "close": round(close, 2), "high10": round(high10, 2),
            "high20": round(high20, 2), "pct_from_high10": round(pct_from_high10, 1),
            "pct_from_high20": round(pct_from_high20, 1),
            "atr10": round(atr10, 2), "ma50": round(ma50, 2),
            "vol_tier": vol_tier, "rs4w": rs4w, "volume": int(vol), "avg_vol20": int(avg_vol20),
        }

    # ── Scenario B: REVERSAL developing (downtrend, near support) ──
    if close < ma50 and not pd.isna(ma200) and close >= ma200 * 0.88:
        pct_from_ma50  = (ma50  - close) / close * 100
        pct_from_ma200 = (ma200 - close) / close * 100   # negative = price above MA200
        atr_dist_ma200 = abs(close - ma200) / atr10 if atr10 > 0 else 99

        score = 0
        notes = []

        # Proximity to MA50 reclaim (max 35) — closer = higher score
        if pct_from_ma50 <= 1.0:   score += 35; notes.append(f"{pct_from_ma50:.1f}% to MA50")
        elif pct_from_ma50 <= 3.0: score += 25; notes.append(f"{pct_from_ma50:.1f}% to MA50")
        elif pct_from_ma50 <= 5.0: score += 15; notes.append(f"{pct_from_ma50:.1f}% to MA50")
        elif pct_from_ma50 <= 8.0: score += 8;  notes.append(f"{pct_from_ma50:.1f}% to MA50")
        else:                      score += 0

        # Near MA200 support (max 30)
        if atr_dist_ma200 <= 1.0:   score += 30; notes.append("at MA200")
        elif atr_dist_ma200 <= 2.0: score += 20; notes.append(f"{atr_dist_ma200:.1f}ATR to MA200")
        elif atr_dist_ma200 <= 3.0: score += 10; notes.append(f"{atr_dist_ma200:.1f}ATR to MA200")
        # Price already above MA200 is also interesting (support held)
        if pct_from_ma200 < 0:
            score += 10; notes.append("above MA200")

        # Volume declining = sellers exhausting (max 20)
        if vol_ratio < 0.70:   score += 20; notes.append("vol↓↓")
        elif vol_ratio < 0.85: score += 12; notes.append("vol↓")

        # RS4W improving even if < 1.0 (max 15)
        if rs4w and rs4w >= 1.05:  score += 15; notes.append(f"RS {rs4w:.2f}")
        elif rs4w and rs4w >= 0.95: score += 8; notes.append(f"RS {rs4w:.2f}")

        if score < 30:
            return None

        dev_type = "→ REVERSAL"

        explain_parts = [
            f"📉 Downtrend: giá ({close:,.0f}) dưới MA50 ({ma50:,.0f}), "
            f"cách MA50 {pct_from_ma50:.1f}%.",
        ]
        if atr_dist_ma200 <= 2.0:
            if pct_from_ma200 < 0:
                explain_parts.append(
                    f"🛡️ Giá đang trên MA200 ({ma200:,.0f}) — vùng hỗ trợ dài hạn đang giữ."
                )
            else:
                explain_parts.append(
                    f"🛡️ Giá cách MA200 ({ma200:,.0f}) chỉ {atr_dist_ma200:.1f}× ATR — "
                    "đang tiếp cận vùng hỗ trợ dài hạn."
                )
        if pct_from_ma50 <= 3.0:
            explain_parts.append(
                f"🔑 Chỉ cần tăng {pct_from_ma50:.1f}% là reclaim MA50 — "
                "tín hiệu chuyển trend tiềm năng."
            )
        if vol_ratio < 0.85:
            explain_parts.append(
                f"🔇 Volume 5 ngày giảm còn {vol_ratio*100:.0f}% baseline — "
                "áp lực bán đang kiệt dần."
            )
        if rs4w and rs4w >= 0.95:
            explain_parts.append(f"📊 RS4W = {rs4w:.2f} vs thị trường.")
        explain_parts.append(
            "⏳ Trigger khi: nến tăng mạnh từ vùng MA200, đóng gần đỉnh, "
            "volume bùng nổ → scan_reversal bắt tín hiệu."
        )

        return {
            "signal": "WATCH", "dev_type": dev_type, "score": score,
            "notes": ", ".join(notes), "explain": "\n".join(explain_parts),
            "date": df.index[-1],
            "close": round(close, 2), "high10": round(high10, 2),
            "high20": round(high20, 2), "pct_from_high10": round(pct_from_ma50, 1),
            "pct_from_high20": round(pct_from_ma200, 1),
            "atr10": round(atr10, 2), "ma50": round(ma50, 2), "ma200": round(ma200, 2),
            "vol_tier": vol_tier, "rs4w": rs4w, "volume": int(vol), "avg_vol20": int(avg_vol20),
        }

    return None


# ============================================================
# SCAN RUNNER
# ============================================================
_SIGNAL_PRIORITY = {
    "BREAKOUT_STRONG": 0,
    "GAP_STRONG":      1,
    "NR7_STRONG":      2,
    "BREAKOUT_EARLY":  3,
    "GAP_EARLY":       4,
    "NR7_EARLY":       5,
    "TF_MA20":         6,
    "TF_MA50":         7,
}

def run_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
    watchlist_top: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Scan all symbols in parallel.
    Returns (signals, watchlist) where watchlist = top developing setups
    for stocks that didn't trigger any signal.
    """
    signals:   list[dict] = []
    watchlist: list[dict] = []
    total = len(symbols)
    done  = 0

    def _scan_one(sym: str) -> tuple[str, dict] | None:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty or len(df) < 60:
            return None
        df = compute_indicators(df)
        sig = (
            scan_breakout(df, vnindex_df)     or
            scan_gap(df, vnindex_df)           or
            scan_nr7(df, vnindex_df)           or
            scan_trend_filter(df, vnindex_df)
        )
        if sig:
            sig["symbol"] = sym.replace(".VN", "")
            sig["sector"] = symbols.get(sym, "")
            return ("signal", sig)
        # No trigger — score for watchlist
        cand = score_developing(df, vnindex_df)
        if cand:
            cand["symbol"] = sym.replace(".VN", "")
            cand["sector"] = symbols.get(sym, "")
            return ("watch", cand)
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_scan_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    kind, data = res
                    if kind == "signal":
                        signals.append(data)
                    else:
                        watchlist.append(data)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    # Market regime gate — suppress breakout signals in downtrend market
    _BREAKOUT_TYPES = {
        "BREAKOUT_STRONG", "BREAKOUT_EARLY",
        "GAP_STRONG", "GAP_EARLY",
        "NR7_STRONG", "NR7_EARLY",
    }
    market_downtrend = False
    if vnindex_df is not None:
        try:
            idx_col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
            vn_series = vnindex_df[idx_col].dropna()
            if len(vn_series) >= 51:
                market_downtrend = bool(vn_series.iloc[-1] < vn_series.rolling(50).mean().iloc[-1])
        except Exception:
            pass
    if market_downtrend:
        signals = [s for s in signals if s["signal"] not in _BREAKOUT_TYPES]

    def _sig_sort(r: dict):
        sig_order  = _SIGNAL_PRIORITY.get(r["signal"], 9)
        vol_order  = {"TIER1": 0, "TIER2": 1, "TIER3": 2}.get(r.get("vol_tier", ""), 3)
        rs_order   = 0 if (r.get("rs4w") or 0) >= 1.05 else 1
        supply_pen = 1 if r.get("supply_overhead") else 0
        nr7_rank   = -r.get("nr7_score", 0)   # higher score = earlier in list
        return (sig_order, supply_pen, nr7_rank, vol_order, rs_order, r["symbol"])

    watchlist_sorted = sorted(watchlist, key=lambda x: -x["score"])[:watchlist_top]

    # RS percentile ranking within the combined signal + watchlist universe
    _assign_rs_pct(signals + watchlist_sorted)

    return sorted(signals, key=_sig_sort), watchlist_sorted, market_downtrend


# ============================================================
# PLOTLY CHART
# ============================================================
def show_chart(symbol: str, sig: dict | None = None, use_cache: bool = True) -> None:
    df = load_price_data(symbol + ".VN", use_cache=use_cache)
    if df is None or df.empty:
        st.warning(f"Không có dữ liệu cho {symbol}")
        return

    df   = compute_indicators(df)
    view = df.iloc[-120:]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"],  close=view["Close"],
        name=symbol,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Moving averages
    ma_specs = [("ma20", "#34d399", "MA20"), ("ma50", "#f59e0b", "MA50"), ("ma200", "#818cf8", "MA200")]
    for col_name, color, label in ma_specs:
        if col_name in view.columns and not view[col_name].isna().all():
            fig.add_trace(go.Scatter(
                x=view.index, y=view[col_name], name=label,
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # Signal reference lines
    if sig:
        if "sl" in sig and not pd.isna(sig["sl"]):
            fig.add_hline(y=sig["sl"], line=dict(color="#ef5350", dash="dash", width=1.5),
                          annotation_text="SL", annotation_position="right",
                          row=1, col=1)
        for tp_key in ["tp", "tp2"]:
            if tp_key in sig and not pd.isna(sig[tp_key]):
                fig.add_hline(y=sig[tp_key], line=dict(color="#26a69a", dash="dash", width=1.5),
                              annotation_text="TP", annotation_position="right",
                              row=1, col=1)
        if "tp1" in sig and not pd.isna(sig["tp1"]):
            fig.add_hline(y=sig["tp1"], line=dict(color="#34d399", dash="dot", width=1.2),
                          annotation_text="TP1", annotation_position="right",
                          row=1, col=1)
        if "high20" in sig and sig["high20"] and not pd.isna(sig["high20"]):
            fig.add_hline(y=sig["high20"], line=dict(color="#fbbf24", dash="dot", width=1),
                          annotation_text="HIGH20", annotation_position="right",
                          row=1, col=1)
        # MR specific: range boundaries
        if "range_high" in sig and sig.get("range_high"):
            fig.add_hline(y=sig["range_high"],
                          line=dict(color="#f97316", dash="dash", width=1.2),
                          annotation_text="R.High", annotation_position="right",
                          row=1, col=1)
        if "range_low" in sig and sig.get("range_low"):
            fig.add_hline(y=sig["range_low"],
                          line=dict(color="#22c55e", dash="dash", width=1.2),
                          annotation_text="R.Low / Support", annotation_position="right",
                          row=1, col=1)
        # PB+BO specific: consolidation zone
        if "consol_high" in sig and sig.get("consol_high"):
            fig.add_hline(y=sig["consol_high"],
                          line=dict(color="#a78bfa", dash="dash", width=1.2),
                          annotation_text="Consol H", annotation_position="right",
                          row=1, col=1)
        if "consol_low" in sig and sig.get("consol_low"):
            fig.add_hline(y=sig["consol_low"],
                          line=dict(color="#a78bfa", dash="dot", width=1.0),
                          annotation_text="Consol L", annotation_position="right",
                          row=1, col=1)

    # Volume bars
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(view["Close"], view["Open"])
    ]
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.7), row=2, col=1)
    if "avg_vol20" in view.columns:
        fig.add_trace(go.Scatter(x=view.index, y=view["avg_vol20"],
                                 name="AvgVol20", line=dict(color="#fbbf24", width=1)),
                      row=2, col=1)

    signal_label = f" — {sig['signal']}" if sig else ""
    status_label = f" ({sig.get('status', '')})" if sig and sig.get("status") else ""
    fig.update_layout(
        title=f"{symbol}{signal_label}{status_label}",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.08),
        height=600,
        margin=dict(l=50, r=80, t=60, b=30),
    )
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")

    st.plotly_chart(fig, use_container_width=True)

    if sig:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", sig.get("close", ""))
        c1.metric("SL", sig.get("sl", sig.get("range_low", "—")))
        if "tp" in sig:
            c2.metric("TP", sig["tp"])
            c2.metric("R:R", sig.get("rr", ""))
        if "tp1" in sig:
            c2.metric("TP1", sig["tp1"])
            c2.metric("TP2", sig.get("tp2", ""))
        c3.metric("ATR10", sig.get("atr10", sig.get("atr14", "—")))
        rs4w = sig.get("rs4w")
        rs_pct = sig.get("rs_pct")
        c3.metric("RS4W", (f"{rs4w:.2f}" + (f" ({rs_pct:.0f}%)" if rs_pct else "")) if rs4w else "—")
        c4.metric("Volume", sig.get("vol_tier", ""))
        qf = []
        vc = sig.get("vol_char", "")
        if vc: qf.append(vc)
        td = sig.get("tight_days", 0)
        if td: qf.append(f"{td} tight days")
        if sig.get("weekly_ok"):       qf.append("Weekly OK")
        if sig.get("supply_overhead"): qf.append("⚠ Supply")
        if sig.get("is_inside_bar"):
            chain = sig.get("ib_chain", 1)
            qf.append(f"IB×{chain}" if chain > 1 else "Inside Bar")
        ra = sig.get("resist_atr")
        if ra is not None: qf.append(f"Resist {ra:.1f}ATR above")
        vq = sig.get("vol_quiet", "")
        if vq in ("QUIET", "QUIET++"): qf.append(f"Vol {vq}")
        c4.metric("Quality", " · ".join(qf) if qf else "—")
        ns = sig.get("nr7_score")
        if ns is not None:
            c5.metric("NR7 Score", f"{ns}/100")
        # Position sizing
        capital  = st.session_state.get("capital", 100_000_000)
        risk_pct = st.session_state.get("risk_pct", 0.01)
        entry    = sig["close"]
        sl_price = sig.get("sl", 0)
        risk_per = entry - sl_price
        if risk_per > 0:
            shares = (capital * risk_pct) / risk_per
            c5.metric("Pos size", f"{shares:,.0f} cp")
            c5.metric("Pos value", f"{shares * entry / 1e6:.1f}M")
        else:
            c5.metric("Pos size", "—")
        if "ma200" in sig:
            c5.metric("MA200", sig["ma200"])


# ============================================================
# WATCHLIST TABLE
# ============================================================
def _render_watchlist(rows: list[dict], use_cache: bool) -> None:
    if not rows:
        st.info("Không có cổ phiếu đang phát triển pattern.")
        return

    st.caption("Chưa trigger — đang tiếp cận vùng breakout. Sắp xếp theo điểm tiềm năng.")

    table_rows = []
    for r in rows:
        rs4w    = r.get("rs4w")
        rs_str  = f"{rs4w:.2f}" if rs4w is not None else "—"
        rs_icon = "🟢" if rs4w and rs4w >= 1.05 else ""
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        dev_type = r.get("dev_type", "")
        if "REVERSAL" in dev_type:
            key_pct = f"{r.get('pct_from_high10', '')}% to MA50"
        else:
            key_pct = f"{r.get('pct_from_high10', '')}% to H10"
        table_rows.append({
            "Mã":       r["symbol"],
            "Điểm":     r["score"],
            "Loại":     dev_type,
            "Giá":      r["close"],
            "Key %":    key_pct,
            "MA50":     r.get("ma50", ""),
            "MA200":    r.get("ma200", ""),
            "Vol":      f"{vol_icon} {r.get('vol_tier', '')}",
            "RS4W":     f"{rs_icon} {rs_str}",
            "Ghi chú":  r.get("notes", ""),
            "Ngành":    r.get("sector", ""),
        })

    df_display = pd.DataFrame(table_rows)
    try:
        selected = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="wl_table",
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["wl_sel"] = rows[sel_rows[0]]
            st.session_state.pop("sig_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# RESULTS TABLE
# ============================================================
def _render_results(rows: list[dict], use_cache: bool, key: str = "sig_table") -> None:
    if not rows:
        st.info("Không có tín hiệu")
        return

    table_rows = []
    for r in rows:
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        sig_icon = {
            "BREAKOUT_STRONG": "🚀", "BREAKOUT_EARLY": "📊",
            "NR7_STRONG": "🔩",     "NR7_EARLY": "🔧",
            "GAP_STRONG": "⚡",     "GAP_EARLY": "🌩",
            "TF_MA20": "🎯", "TF_MA50": "📌",
        }.get(r["signal"], "")
        rs4w     = r.get("rs4w")
        rs_pct   = r.get("rs_pct")
        rs_str   = f"{rs4w:.2f}" + (f" ({rs_pct:.0f}%)" if rs_pct else "") if rs4w is not None else "—"
        rs_icon  = "🟢" if rs4w and rs4w >= 1.05 else ("🔴" if rs4w and rs4w < 0.95 else "")
        gap_str  = f" gap {r['gap_pct']:+.1f}%" if r.get("gap_pct") is not None else ""
        # Quality flags
        qf = []
        vc = r.get("vol_char", "")
        if vc == "ACCUM":     qf.append("A")
        elif vc == "DISTRIB": qf.append("D")
        td = r.get("tight_days", 0)
        if td >= 3: qf.append(f"{td}T")
        if r.get("weekly_ok"):       qf.append("W")
        if r.get("supply_overhead"): qf.append("⚠S")
        # NR7-specific flags
        if r.get("is_inside_bar"):
            chain = r.get("ib_chain", 1)
            qf.append(f"IB{'×' + str(chain) if chain > 1 else ''}")
        ra = r.get("resist_atr")
        if ra is not None and ra <= 2.0: qf.append(f"R{ra:.1f}ATR")
        vq = r.get("vol_quiet", "")
        if vq == "QUIET++":  qf.append("Q++")
        elif vq == "QUIET":  qf.append("Q")
        ns = r.get("nr7_score")
        if ns is not None:   qf.append(f"S{ns}")
        quality = "·".join(qf) if qf else "—"
        touched = r.get("touched_ma", "")
        table_rows.append({
            "Mã":      r["symbol"],
            "Loại":    f"{sig_icon} {r['signal']}{gap_str}",
            "Giá":     r["close"],
            "SL":      r["sl"],
            "TP":      r.get("tp", r.get("tp2", "")),
            "R:R":     r.get("rr", ""),
            "RS4W":    f"{rs_icon} {rs_str}",
            "Volume":  f"{vol_icon} {r.get('vol_tier', '')}",
            "Quality": quality,
            "Touch":   touched,
            "Ngành":   r.get("sector", ""),
        })

    df_display = pd.DataFrame(table_rows)

    try:
        selected = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=key,
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["sig_sel"] = rows[sel_rows[0]]
            st.session_state.pop("wl_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# MEAN REVERSION RANGE SCAN
# ============================================================
MR_CONFIG: dict = {
    "range_window":          50,
    "ema_fast":              20,
    "ema_slow":              50,
    "volume_window":         20,
    "atr_window":            14,
    "rsi_window":             3,
    "slope_lookback":         5,
    "support_tolerance":   0.03,
    "bottom_zone_threshold": 0.25,
    "min_avg_volume":    100_000,
    "min_avg_traded_value": 1_000_000_000,   # 1B VND/day
    "min_range_pct":        0.08,
    "max_range_pct":        0.30,
    "max_abs_ema20_slope":  0.02,
    "max_abs_ema50_slope": 0.015,
    "max_10bar_drop":      -0.10,
    "min_history":           80,
}


def _compute_mr_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all MR-specific indicators on a single-ticker df."""
    d = df.copy()

    # EMAs
    d["mr_ema20"] = d["Close"].ewm(span=cfg["ema_fast"],  adjust=False).mean()
    d["mr_ema50"] = d["Close"].ewm(span=cfg["ema_slow"],  adjust=False).mean()

    # ATR14 (Wilder smoothing = EWM with span=atr_window)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift(1)).abs(),
        (d["Low"]  - d["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["mr_atr14"] = tr.ewm(span=cfg["atr_window"], adjust=False).mean()

    # RSI(3) using EWM
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).ewm(span=cfg["rsi_window"], adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=cfg["rsi_window"], adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    d["mr_rsi3"] = 100 - (100 / (1 + rs))

    # Volume metrics
    d["mr_avg_vol20"] = d["Volume"].rolling(cfg["volume_window"]).mean()
    d["mr_avg_tv20"]  = (d["Volume"] * d["Close"]).rolling(cfg["volume_window"]).mean()

    # 50-bar range (no lookahead — uses only past bars up to current row)
    d["mr_range_high"] = d["High"].rolling(cfg["range_window"]).max()
    d["mr_range_low"]  = d["Low"].rolling(cfg["range_window"]).min()

    range_size = d["mr_range_high"] - d["mr_range_low"]
    safe_low   = d["mr_range_low"].replace(0, float("nan"))
    d["mr_range_size_pct"]    = range_size / safe_low
    d["mr_position_in_range"] = (d["Close"] - d["mr_range_low"]) / range_size.replace(0, float("nan"))
    d["mr_dist_to_support"]   = (d["Close"] - d["mr_range_low"]) / safe_low

    # EMA slopes
    lb = cfg["slope_lookback"]
    d["mr_ema20_slope"] = (d["mr_ema20"] - d["mr_ema20"].shift(lb)) / d["mr_ema20"].shift(lb).replace(0, float("nan"))
    d["mr_ema50_slope"] = (d["mr_ema50"] - d["mr_ema50"].shift(lb)) / d["mr_ema50"].shift(lb).replace(0, float("nan"))

    return d


def _mr_is_valid_range(d: pd.DataFrame, row: pd.Series, cfg: dict) -> bool:
    """Return True if current bar satisfies the range-bound conditions."""
    rsp  = float(row["mr_range_size_pct"])
    e20s = abs(float(row["mr_ema20_slope"]))
    e50s = abs(float(row["mr_ema50_slope"]))
    close = float(row["Close"])
    rh    = float(row["mr_range_high"])
    rl    = float(row["mr_range_low"])
    tol   = cfg["support_tolerance"]

    if not (cfg["min_range_pct"] <= rsp <= cfg["max_range_pct"]): return False
    if e20s > cfg["max_abs_ema20_slope"]: return False
    if e50s > cfg["max_abs_ema50_slope"]: return False
    if close >= rh:                        return False   # breakout high

    # At least 2 touches of each boundary in last range_window bars
    window = d.iloc[-cfg["range_window"]:]
    highs_near = (window["High"] >= rh * (1 - tol)).sum()
    lows_near  = (window["Low"]  <= rl * (1 + tol)).sum()
    if highs_near < 2 or lows_near < 2:
        return False

    return True


def _mr_is_near_support(row: pd.Series, cfg: dict) -> bool:
    """
    Price is in the bottom zone near support.
    Uses position_in_range as the primary filter (avoids over-strictness
    from absolute distance checks when the range is wide).
    """
    pir   = float(row["mr_position_in_range"])
    close = float(row["Close"])
    rl    = float(row["mr_range_low"])
    return (
        pir <= cfg["bottom_zone_threshold"] and   # bottom X% of range
        close > rl * 0.97                          # above range_low (not a breakdown)
    )


def _mr_detect_reversal(d: pd.DataFrame) -> str | None:
    """Detect the first matching reversal signal on the last bar."""
    if len(d) < 3:
        return None
    today = d.iloc[-1]; prev = d.iloc[-2]; prev2 = d.iloc[-3]
    c, h, l, o  = float(today["Close"]), float(today["High"]), float(today["Low"]),  float(today["Open"])
    pc, ph, pl   = float(prev["Close"]),  float(prev["High"]),  float(prev["Low"])
    pp2c         = float(prev2["Close"])
    rsi3         = float(today.get("mr_rsi3", 50))

    if c > ph:                                         return "CLOSE_ABOVE_PREV_HIGH"
    if c > pc and l <= pl:                             return "HIGHER_CLOSE_LOWER_LOW"
    candle_range = h - l
    if candle_range > 0:
        lower_wick = min(o, c) - l
        body       = abs(c - o)
        if lower_wick > body and c >= (h + l) / 2:    return "HAMMER"
    if c > pc > pp2c:                                  return "TWO_HIGHER_CLOSES"
    if not pd.isna(rsi3) and rsi3 < 20 and c > pc:    return "RSI3_BOUNCE"
    return None


def _mr_is_rejected(d: pd.DataFrame, row: pd.Series, cfg: dict) -> bool:
    """Return True if the setup should be rejected (trend too strong / breaking down)."""
    close = float(row["Close"])
    ema50 = float(row["mr_ema50"])
    ema20 = float(row["mr_ema20"])
    e50s  = float(row["mr_ema50_slope"])
    atr14 = float(row["mr_atr14"])

    # Strong downtrend
    if close < ema50 and ema20 < ema50 and e50s < -0.02:
        return True
    # 10-bar drop too deep
    if len(d) >= 11:
        c10 = float(d.iloc[-11]["Close"])
        if c10 > 0 and (close - c10) / c10 < cfg["max_10bar_drop"]:
            return True
    # 3 consecutive large bearish candles
    if len(d) >= 4 and atr14 > 0:
        if all(
            float(d.iloc[-i]["Close"]) < float(d.iloc[-i]["Open"]) and
            (float(d.iloc[-i]["Open"]) - float(d.iloc[-i]["Close"])) > 0.5 * atr14
            for i in range(1, 4)
        ):
            return True
    return False


def _mr_score(row: pd.Series, reversal: str, cfg: dict) -> dict:
    """Compute component scores and weighted final score."""
    import math
    pir  = max(0.0, min(1.0, float(row["mr_position_in_range"])))
    e20s = abs(float(row["mr_ema20_slope"]))
    e50s = abs(float(row["mr_ema50_slope"]))
    rsi3 = min(100.0, max(0.0, float(row.get("mr_rsi3", 50))))
    avg_tv = max(1.0, float(row.get("mr_avg_tv20", 1.0)))

    support_score   = max(0.0, 1.0 - pir / max(cfg["bottom_zone_threshold"], 0.01))
    reversal_score  = {"CLOSE_ABOVE_PREV_HIGH": 1.0, "HAMMER": 0.9, "RSI3_BOUNCE": 0.85,
                       "HIGHER_CLOSE_LOWER_LOW": 0.75, "TWO_HIGHER_CLOSES": 0.70}.get(reversal, 0.5)
    range_score     = max(0.0, 1.0 - (e20s / max(cfg["max_abs_ema20_slope"], 1e-6) +
                                       e50s / max(cfg["max_abs_ema50_slope"], 1e-6)) / 2)
    rsi_score       = max(0.0, (30.0 - min(rsi3, 30.0)) / 30.0)
    liq_score       = min(1.0, math.log10(avg_tv) / 13.0)   # ~10T VND = 1.0

    final = (0.30 * support_score + 0.30 * reversal_score +
             0.20 * range_score   + 0.10 * rsi_score + 0.10 * liq_score)
    return {"final": final, "support": support_score, "reversal": reversal_score,
            "range": range_score, "rsi": rsi_score, "liq": liq_score}


def scan_mean_reversion(df: pd.DataFrame, vnindex_df=None, config: dict | None = None) -> dict | None:
    """
    Mean Reversion Range scan for a single ticker.
    Finds stocks in a stable horizontal range, near support, with reversal signal.
    Returns signal dict or None.
    """
    cfg = config or MR_CONFIG
    if df is None or len(df) < cfg["min_history"]:
        return None

    d = _compute_mr_indicators(df, cfg)
    row = d.iloc[-1]

    required = ["mr_ema20", "mr_ema50", "mr_atr14", "mr_rsi3", "mr_range_high",
                "mr_range_low", "mr_range_size_pct", "mr_position_in_range",
                "mr_dist_to_support", "mr_ema20_slope", "mr_ema50_slope",
                "mr_avg_vol20", "mr_avg_tv20"]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    # Liquidity
    if float(row["mr_avg_vol20"]) < cfg["min_avg_volume"]:
        return None

    # Range condition
    if not _mr_is_valid_range(d, row, cfg):
        return None

    # Near support
    if not _mr_is_near_support(row, cfg):
        return None

    # Reversal signal
    reversal = _mr_detect_reversal(d)
    if reversal is None:
        return None

    # Trend rejection
    if _mr_is_rejected(d, row, cfg):
        return None

    scores = _mr_score(row, reversal, cfg)
    rs4w   = compute_rs4w(df, vnindex_df)

    return {
        "signal":              "MR_LONG",
        "date":                df.index[-1],
        "close":               round(float(row["Close"]), 2),
        "range_high":          round(float(row["mr_range_high"]), 2),
        "range_low":           round(float(row["mr_range_low"]), 2),
        "range_size_pct":      round(float(row["mr_range_size_pct"]) * 100, 1),
        "position_in_range":   round(float(row["mr_position_in_range"]) * 100, 1),
        "dist_support_pct":    round(float(row["mr_dist_to_support"]) * 100, 2),
        "ema20":               round(float(row["mr_ema20"]), 2),
        "ema50":               round(float(row["mr_ema50"]), 2),
        "ema20_slope":         round(float(row["mr_ema20_slope"]) * 100, 3),
        "ema50_slope":         round(float(row["mr_ema50_slope"]) * 100, 3),
        "rsi3":                round(float(row["mr_rsi3"]), 1),
        "atr14":               round(float(row["mr_atr14"]), 2),
        "avg_vol20":           int(row["mr_avg_vol20"]),
        "reversal_signal":     reversal,
        "final_score":         round(scores["final"], 3),
        "score_support":       round(scores["support"], 3),
        "score_reversal":      round(scores["reversal"], 3),
        "score_range":         round(scores["range"], 3),
        "rs4w":                rs4w,
        "vol_tier":            _vol_tier(float(row["Volume"]), float(row["mr_avg_vol20"]),
                                         float(row.get("avg_vol_pre5", float("nan")))),
    }


def run_mr_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    config: dict | None = None,
    progress_cb=None,
) -> list[dict]:
    """Run scan_mean_reversion across all symbols in parallel."""
    cfg     = config or MR_CONFIG
    results: list[dict] = []
    total   = len(symbols)
    done    = 0

    def _one(sym: str) -> dict | None:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty:
            return None
        sig = scan_mean_reversion(df, vnindex_df, cfg)
        if sig:
            sig["symbol"] = sym.replace(".VN", "")
            sig["sector"] = symbols.get(sym, "")
        return sig

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return sorted(results, key=lambda r: -r["final_score"])


# ============================================================
# MEAN REVERSION RESULTS TABLE
# ============================================================
def _render_mr_results(rows: list[dict], use_cache: bool, key: str = "mr_table") -> None:
    if not rows:
        st.info("Không có tín hiệu Mean Reversion.")
        return

    st.caption("Long-only range swing — mua gần support, bán gần kháng cự | Sắp xếp theo Final Score")

    reversal_icons = {
        "CLOSE_ABOVE_PREV_HIGH":  "🚀",
        "HAMMER":                 "🔨",
        "RSI3_BOUNCE":            "📈",
        "HIGHER_CLOSE_LOWER_LOW": "↗️",
        "TWO_HIGHER_CLOSES":      "⬆️",
    }

    table_rows = []
    for r in rows:
        rs4w     = r.get("rs4w")
        rs_str   = f"{rs4w:.2f}" if rs4w is not None else "—"
        rs_icon  = "🟢" if rs4w and rs4w >= 1.05 else ("🔴" if rs4w and rs4w < 0.95 else "")
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        rev      = r.get("reversal_signal", "")
        table_rows.append({
            "Mã":          r["symbol"],
            "Score":       f"{r['final_score']:.2f}",
            "Giá":         r["close"],
            "Pos%":        f"{r.get('position_in_range', '')}%",
            "Dist%":       f"{r.get('dist_support_pct', '')}%",
            "Range%":      f"{r.get('range_size_pct', '')}%",
            "RSI3":        r.get("rsi3", ""),
            "Reversal":    f"{reversal_icons.get(rev, '')} {rev}",
            "EMA20s":      f"{r.get('ema20_slope', '')}%",
            "EMA50s":      f"{r.get('ema50_slope', '')}%",
            "RS4W":        f"{rs_icon} {rs_str}",
            "Vol":         f"{vol_icon} {r.get('vol_tier', '')}",
            "Ngành":       r.get("sector", ""),
        })

    df_display = pd.DataFrame(table_rows)
    try:
        selected = st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key=key,
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["mr_sel"] = rows[sel_rows[0]]
            st.session_state.pop("pb_sel",  None)
            st.session_state.pop("sig_sel", None)
            st.session_state.pop("wl_sel",  None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# SCAN — Trend Pullback + Breakout (independent strategy)
# ============================================================
def scan_pullback_breakout(
    df: pd.DataFrame,
    vnindex_df=None,
    *,
    consol_bars: int        = 5,
    pullback_bars: int      = 15,
    consol_range_max: float = 0.05,   # max 5% range = "tight" consolidation
    pullback_min: float     = 0.03,   # pullback must be at least 3% from prior high/low
    pullback_max: float     = 0.20,   # but not more than 20% (not a reversal)
    rr_ratio: float         = 2.0,
) -> dict | None:
    """
    Trend Pullback + Breakout pattern.
    Long : uptrend → price pulls back → consolidation → close breaks above consol high
    Short: downtrend → price bounces → consolidation → close breaks below consol low
    No look-ahead: signal candle = df.iloc[-1]; all windows use prior data only.
    """
    min_bars = 55 + consol_bars + pullback_bars
    if df is None or len(df) < min_bars:
        return None

    row  = df.iloc[-1]
    required = ["ma50", "ma50_prev5", "atr10", "avg_vol20"]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    ma50  = row["ma50"];  ma50_prev5 = row["ma50_prev5"]
    atr10 = row["atr10"]; avg_vol20  = row["avg_vol20"]
    vol   = row["Volume"]

    # Consolidation window: last consol_bars candles BEFORE signal candle
    consol = df.iloc[-consol_bars - 1: -1]
    if len(consol) < consol_bars:
        return None
    consol_high = float(consol["High"].max())
    consol_low  = float(consol["Low"].min())
    consol_mid  = (consol_high + consol_low) / 2
    consol_range_pct = (consol_high - consol_low) / consol_mid if consol_mid > 0 else 1.0

    # Consolidation must be tight
    if consol_range_pct > consol_range_max:
        return None

    # Prior trend window: before the consolidation
    prior = df.iloc[-pullback_bars - consol_bars - 1: -consol_bars - 1]
    if len(prior) < 5:
        return None
    prior_high = float(prior["High"].max())
    prior_low  = float(prior["Low"].min())

    # Determine direction from signal candle first — avoids ambiguity when
    # both MA50 conditions could apply simultaneously.
    # Long only — signal candle must close above consolidation high (bull breakout)
    if not (close > consol_high and close > open_):
        return None
    signal_type = "PB_LONG"

    # Trend: prior window had highs above MA50 (uptrend was active before pullback)
    if not (prior_high > ma50):
        return None
    # Sanity: consolidation not too far below MA50
    if consol_low < ma50 * 0.85:
        return None
    # Pullback: consolidation sits below the prior high by pullback_min..pullback_max
    if prior_high <= 0:
        return None
    pullback_pct = (prior_high - consol_high) / prior_high
    if not (pullback_min <= pullback_pct <= pullback_max):
        return None
    sl        = round(consol_low, 2)
    entry     = round(close, 2)
    tp        = round(entry + rr_ratio * (entry - sl), 2)
    rr        = round((tp - entry) / max(entry - sl, 0.001), 2)
    direction = "Long"

    vol_tier = _vol_tier(vol, avg_vol20, row.get("avg_vol_pre5", float("nan")))
    rs4w     = compute_rs4w(df, vnindex_df)
    vol_char = compute_vol_character(df)
    weekly   = check_weekly_trend(df)

    return {
        "signal":           signal_type,
        "direction":        direction,
        "date":             df.index[-1],
        "close":            round(close, 2),
        "consol_high":      round(consol_high, 2),
        "consol_low":       round(consol_low, 2),
        "prior_high":       round(prior_high, 2),
        "prior_low":        round(prior_low, 2),
        "consol_range_pct": round(consol_range_pct * 100, 1),
        "pullback_pct":     round(pullback_pct * 100, 1),
        "atr10":            round(atr10, 2),
        "ma50":             round(ma50, 2),
        "ma200":            round(row.get("ma200", float("nan")), 2),
        "sl":               sl,
        "tp":               tp,
        "rr":               rr,
        "vol_tier":         vol_tier,
        "vol_char":         vol_char,
        "weekly_ok":        weekly,
        "rs4w":             rs4w,
        "volume":           int(vol),
        "avg_vol20":        int(avg_vol20),
    }


def run_pb_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
    pb_params: dict | None = None,
) -> tuple[list[dict], bool]:
    """
    Run scan_pullback_breakout across all symbols in parallel.
    Returns (signals, market_downtrend).
    """
    pb_params     = pb_params or {}
    results: list[dict] = []
    total = len(symbols)
    done  = 0

    def _scan_one(sym: str) -> dict | None:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty:
            return None
        df  = compute_indicators(df)
        sig = scan_pullback_breakout(df, vnindex_df, **pb_params)
        if sig:
            sig["symbol"] = sym.replace(".VN", "")
            sig["sector"] = symbols.get(sym, "")
        return sig

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_scan_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    # Market regime
    market_downtrend = False
    if vnindex_df is not None:
        try:
            idx_col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
            vn = vnindex_df[idx_col].dropna()
            if len(vn) >= 51:
                market_downtrend = bool(vn.iloc[-1] < vn.rolling(50).mean().iloc[-1])
        except Exception:
            pass

    def _sort(r: dict):
        sig_order = 0 if r["signal"] == "PB_LONG" else 1
        vol_order = {"TIER1": 0, "TIER2": 1}.get(r.get("vol_tier", ""), 2)
        rs_order  = 0 if (r.get("rs4w") or 0) >= 1.05 else 1
        return (sig_order, vol_order, rs_order, r["symbol"])

    return sorted(results, key=_sort), market_downtrend


# ============================================================
# PB+BO RESULTS TABLE
# ============================================================
def _render_pb_results(rows: list[dict], use_cache: bool, key: str = "pb_table") -> None:
    if not rows:
        st.info("Không có tín hiệu PB+BO.")
        return

    st.caption(
        "Trend Pullback + Breakout — "
        "Long: uptrend → pullback → breakout trên consolidation | "
        "Short: downtrend → bounce → breakdown dưới consolidation"
    )

    table_rows = []
    for r in rows:
        rs4w     = r.get("rs4w")
        rs_str   = f"{rs4w:.2f}" if rs4w is not None else "—"
        rs_icon  = "🟢" if rs4w and rs4w >= 1.05 else ("🔴" if rs4w and rs4w < 0.95 else "")
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        sig_icon = "📈" if r["signal"] == "PB_LONG" else "📉"
        qf = []
        if r.get("vol_char") == "ACCUM":   qf.append("A")
        if r.get("weekly_ok"):             qf.append("W")
        table_rows.append({
            "Mã":        r["symbol"],
            "Hướng":     f"{sig_icon} {r['signal']}",
            "Giá":       r["close"],
            "Consol H":  r.get("consol_high", ""),
            "Consol L":  r.get("consol_low", ""),
            "PB%":       f"{r.get('pullback_pct', '')}%",
            "Range%":    f"{r.get('consol_range_pct', '')}%",
            "SL":        r["sl"],
            "TP":        r["tp"],
            "R:R":       r.get("rr", ""),
            "RS4W":      f"{rs_icon} {rs_str}",
            "Vol":       f"{vol_icon} {r.get('vol_tier', '')}",
            "Quality":   "·".join(qf) if qf else "—",
            "Ngành":     r.get("sector", ""),
        })

    df_display = pd.DataFrame(table_rows)
    try:
        selected = st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key=key,
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["pb_sel"] = rows[sel_rows[0]]
            st.session_state.pop("sig_sel", None)
            st.session_state.pop("wl_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# STREAMLIT APP
# ============================================================
def main() -> None:
    st.set_page_config(
        page_title="VN Stock Screener — Swing D1",
        page_icon="📈",
        layout="wide",
    )

    # ── Header
    st.title("📈 VN Stock Screener — Swing D1")
    st.caption("Breakout Momentum & Trend Filter | Scan sau khi nến ngày đóng cửa")

    # ── Sidebar
    with st.sidebar:
        st.header("Thị trường")
        vnindex_df = get_vnindex_data()
        mkt_label, mkt_info = compute_market_filter(vnindex_df)
        st.markdown(f"**{mkt_label}**")
        if mkt_info:
            st.caption(
                f"VNINDEX {mkt_info['VNINDEX']} | "
                f"MA20 {mkt_info['MA20']} | "
                f"MA50 {mkt_info['MA50']}"
            )

        st.divider()
        st.header("Cache")
        n_files, size_mb = cache_stats()
        st.caption(f"{n_files} files · {size_mb:.1f} MB")
        if st.button("Xóa cache"):
            n = clear_cache()
            st.success(f"Đã xóa {n} files")
        use_cache = st.checkbox("Dùng cache", value=True)

        st.divider()
        st.header("Quản lý vốn")
        capital = st.number_input(
            "Vốn (triệu VND)", min_value=10, max_value=10000, value=100, step=10
        ) * 1_000_000
        risk_pct = st.slider("Risk/lệnh (%)", 0.5, 3.0, 1.0, 0.5) / 100
        st.session_state["capital"]  = capital
        st.session_state["risk_pct"] = risk_pct

        st.divider()
        st.header("Mean Reversion Config")
        mr_range_min  = st.slider("Range min (%)",    5,  15,  8) / 100
        mr_range_max  = st.slider("Range max (%)",   15,  40, 25) / 100
        mr_support_tol = st.slider("Support tol (%)", 1,   6,  3) / 100
        mr_bot_zone   = st.slider("Bottom zone (%)", 15,  40, 25) / 100
        mr_rr         = st.slider("MR R:R",         1.0, 3.0, 2.0, 0.5)
        mr_config = {**MR_CONFIG,
            "min_range_pct":        mr_range_min,
            "max_range_pct":        mr_range_max,
            "support_tolerance":    mr_support_tol,
            "bottom_zone_threshold": mr_bot_zone,
        }
        st.session_state["mr_config"] = mr_config

        st.divider()
        st.header("PB+BO Config")
        pb_consol_bars  = st.slider("Consolidation bars",    3, 10,  5)
        pb_consol_range = st.slider("Consol range max (%)",  1, 10,  5) / 100
        pb_pb_min       = st.slider("Pullback min (%)",      1,  8,  3) / 100
        pb_pb_max       = st.slider("Pullback max (%)",     10, 30, 20) / 100
        pb_rr           = st.slider("R:R ratio",           1.0, 4.0, 2.0, 0.5)
        pb_params = dict(
            consol_bars       = pb_consol_bars,
            consol_range_max  = pb_consol_range,
            pullback_min      = pb_pb_min,
            pullback_max      = pb_pb_max,
            rr_ratio          = pb_rr,
        )
        st.session_state["pb_params"] = pb_params

        st.divider()
        st.header("Chiến lược")
        st.markdown("""
**🚀 Breakout** *(Size ≥ 1.0× ATR)*
- Uptrend: giá > MA50 đang dốc lên
- Đóng > high nến hôm qua + đỉnh 10/20 ngày
- Strong = phá HIGH20 · Early = phá HIGH10

**🔩 NR7** *(Narrow Range — coil)*
- Nến hôm nay hẹp nhất 7 ngày
- Đóng phá HIGH10 → bùng nổ nhỏ = R:R tốt

**⚡ Gap** *(Institutional demand)*
- Gap ≥ 0.5% so với close hôm qua
- Gap không bị lấp, đóng phá HIGH10

**🎯 TF_MA20** *(Pullback nhẹ đến MA20)*
- Trend mạnh: MA20 > MA50 > MA200 (đều dốc lên)
- Low chạm MA20, đóng phá HIGH10 + vol ≥ 1.2×

**📌 TF_MA50** *(Pullback sâu đến MA50 — early uptrend)*
- MA50 gần MA200 (≤5%), low chạm MA50, đóng trên MA50
- 5-day momentum + vol ≥ 1.5× (cần bùng nổ mạnh hơn)

**🟢 RS4W** — RS > 1.05 = cổ phiếu mạnh hơn thị trường

**Volume**: 🔥 TIER1 (2×+contract) · 📈 TIER2 (2×) · TIER3
        """)

    # ── Scan buttons
    col_vn30, col_vn100 = st.columns(2)
    with col_vn30:
        do_vn30 = st.button("Scan VN30 — 30 mã", use_container_width=True)
    with col_vn100:
        do_vn100 = st.button("Scan VN100 — 100 mã", use_container_width=True)

    if do_vn30:
        prog = st.progress(0, text="Scanning VN30… 0 / 30")
        def _cb30(done, total):
            prog.progress(done / total, text=f"Scanning VN30… {done} / {total}")
        sigs, watch, mkt_down = run_scan(VN30_STOCKS,  use_cache=use_cache, vnindex_df=vnindex_df, progress_cb=_cb30)
        st.session_state["scan_results"]       = sigs
        st.session_state["scan_watchlist"]     = watch
        st.session_state["scan_universe"]      = "VN30"
        st.session_state["market_downtrend"]   = mkt_down
        prog.empty()

    if do_vn100:
        prog = st.progress(0, text="Scanning VN100… 0 / 100")
        def _cb100(done, total):
            prog.progress(done / total, text=f"Scanning VN100… {done} / {total}")
        sigs, watch, mkt_down = run_scan(VN100_STOCKS, use_cache=use_cache, vnindex_df=vnindex_df, progress_cb=_cb100)
        st.session_state["scan_results"]       = sigs
        st.session_state["scan_watchlist"]     = watch
        st.session_state["scan_universe"]      = "VN100"
        st.session_state["market_downtrend"]   = mkt_down
        prog.empty()

    # ── Market downtrend banner
    if st.session_state.get("market_downtrend"):
        st.warning(
            "⚠️ Thị trường downtrend (VNIndex < MA50) — "
            "Breakout signals đã bị tắt. Chỉ hiển thị Trend Filter & Watchlist."
        )

    # ── Results
    results   = st.session_state.get("scan_results", [])
    watchlist = st.session_state.get("scan_watchlist", [])
    universe  = st.session_state.get("scan_universe", "")

    if results or watchlist:
        n_bo  = sum(1 for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY"))
        n_nr7 = sum(1 for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY"))
        n_gap = sum(1 for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY"))
        n_tf  = sum(1 for r in results if r["signal"] in ("TF_MA20", "TF_MA50"))
        n_rs  = sum(1 for r in results if (r.get("rs4w") or 0) >= 1.05)
        st.subheader(
            f"Kết quả {universe} — {len(results)} tín hiệu "
            f"(🚀{n_bo} · 🔩{n_nr7} · ⚡{n_gap} · 🎯{n_tf} · 🟢RS{n_rs})"
        )

        tab_all, tab_bo, tab_nr7, tab_gap, tab_tf, tab_watch = st.tabs([
            f"Tất cả ({len(results)})",
            f"🚀 Breakout ({n_bo})",
            f"🔩 NR7 ({n_nr7})",
            f"⚡ Gap ({n_gap})",
            f"🎯 Trend Filter ({n_tf})",
            f"👀 Watchlist ({len(watchlist)})",
        ])
        with tab_all:
            _render_results(results, use_cache, key="tab_all")
        with tab_bo:
            _render_results([r for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY")], use_cache, key="tab_bo")
        with tab_nr7:
            _render_results([r for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY")], use_cache, key="tab_nr7")
        with tab_gap:
            _render_results([r for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY")], use_cache, key="tab_gap")
        with tab_tf:
            _render_results([r for r in results if r["signal"] in ("TF_MA20", "TF_MA50")], use_cache, key="tab_tf")
        with tab_watch:
            _render_watchlist(watchlist, use_cache)

    # ── Chart panel — rendered OUTSIDE tabs so rerun never hides it ──
    wl_sel  = st.session_state.get("wl_sel")
    sig_sel = st.session_state.get("sig_sel")

    if wl_sel:
        st.divider()
        st.subheader(f"{wl_sel['symbol']} — {wl_sel.get('dev_type', '')}  (Score: {wl_sel['score']})")
        for line in wl_sel.get("explain", "").split("\n"):
            st.markdown(line)
        st.divider()
        show_chart(wl_sel["symbol"], sig=None, use_cache=use_cache)
    elif sig_sel:
        st.divider()
        st.subheader(f"Chart — {sig_sel['symbol']}")
        show_chart(sig_sel["symbol"], sig=sig_sel, use_cache=use_cache)

    if not results and not watchlist and not do_vn30 and not do_vn100 and not wl_sel and not sig_sel:
        st.info("Nhấn Scan VN30 hoặc Scan VN100 để bắt đầu.")

    # ══════════════════════════════════════════════════════════════
    # PB+BO — Independent Trend Pullback + Breakout scan
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📐 Trend Pullback + Breakout Scan")
    st.caption("Independent scan — Long/Short based on trend → pullback → consolidation → breakout")

    col_pb30, col_pb100 = st.columns(2)
    with col_pb30:
        do_pb30  = st.button("Scan PB+BO VN30",  use_container_width=True)
    with col_pb100:
        do_pb100 = st.button("Scan PB+BO VN100", use_container_width=True)

    pb_params = st.session_state.get("pb_params", {})

    if do_pb30:
        prog = st.progress(0, text="Scanning PB+BO VN30… 0 / 30")
        def _pb_cb30(done, total):
            prog.progress(done / total, text=f"Scanning PB+BO VN30… {done} / {total}")
        pb_sigs, _ = run_pb_scan(VN30_STOCKS,  use_cache=use_cache,
                                  vnindex_df=vnindex_df, progress_cb=_pb_cb30,
                                  pb_params=pb_params)
        st.session_state["pb_results"]  = pb_sigs
        st.session_state["pb_universe"] = "VN30"
        prog.empty()

    if do_pb100:
        prog = st.progress(0, text="Scanning PB+BO VN100… 0 / 100")
        def _pb_cb100(done, total):
            prog.progress(done / total, text=f"Scanning PB+BO VN100… {done} / {total}")
        pb_sigs, _ = run_pb_scan(VN100_STOCKS, use_cache=use_cache,
                                  vnindex_df=vnindex_df, progress_cb=_pb_cb100,
                                  pb_params=pb_params)
        st.session_state["pb_results"]  = pb_sigs
        st.session_state["pb_universe"] = "VN100"
        prog.empty()

    pb_results  = st.session_state.get("pb_results", [])
    pb_universe = st.session_state.get("pb_universe", "")

    if pb_results:
        st.subheader(f"PB+BO {pb_universe} — {len(pb_results)} tín hiệu 📈 Long")
        _render_pb_results(pb_results, use_cache, key="pb_table_main")

    elif not do_pb30 and not do_pb100:
        st.caption("Nhấn Scan PB+BO để bắt đầu. Cấu hình tham số trong sidebar.")

    # ══════════════════════════════════════════════════════════════
    # MEAN REVERSION RANGE SCAN
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔁 Mean Reversion Range Scan")
    st.caption("Sideways/range market — mua gần support, bán gần kháng cự")

    col_mr30, col_mr100 = st.columns(2)
    with col_mr30:
        do_mr30  = st.button("Scan MR VN30",  use_container_width=True)
    with col_mr100:
        do_mr100 = st.button("Scan MR VN100", use_container_width=True)

    mr_config = st.session_state.get("mr_config", MR_CONFIG)

    if do_mr30:
        prog = st.progress(0, text="Scanning MR VN30… 0 / 30")
        def _mr_cb30(done, total):
            prog.progress(done / total, text=f"Scanning MR VN30… {done} / {total}")
        mr_sigs = run_mr_scan(VN30_STOCKS,  use_cache=use_cache,
                               vnindex_df=vnindex_df, config=mr_config,
                               progress_cb=_mr_cb30)
        st.session_state["mr_results"]  = mr_sigs
        st.session_state["mr_universe"] = "VN30"
        prog.empty()

    if do_mr100:
        prog = st.progress(0, text="Scanning MR VN100… 0 / 100")
        def _mr_cb100(done, total):
            prog.progress(done / total, text=f"Scanning MR VN100… {done} / {total}")
        mr_sigs = run_mr_scan(VN100_STOCKS, use_cache=use_cache,
                               vnindex_df=vnindex_df, config=mr_config,
                               progress_cb=_mr_cb100)
        st.session_state["mr_results"]  = mr_sigs
        st.session_state["mr_universe"] = "VN100"
        prog.empty()

    mr_results  = st.session_state.get("mr_results", [])
    mr_universe = st.session_state.get("mr_universe", "")

    if mr_results:
        st.subheader(f"MR {mr_universe} — {len(mr_results)} candidates 🔁")
        _render_mr_results(mr_results, use_cache, key="mr_table_main")
    elif not do_mr30 and not do_mr100:
        st.caption("Nhấn Scan MR để bắt đầu. Cấu hình tham số trong sidebar.")

    # MR chart panel — outside tabs
    mr_sel = st.session_state.get("mr_sel")
    if mr_sel:
        st.divider()
        st.subheader(
            f"Chart — {mr_sel['symbol']} 🔁 MR | "
            f"Score {mr_sel['final_score']:.2f} | "
            f"Pos {mr_sel.get('position_in_range', '')}% | "
            f"{mr_sel.get('reversal_signal', '')}"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Range High",  mr_sel.get("range_high", ""))
        c1.metric("Range Low",   mr_sel.get("range_low", ""))
        c2.metric("Close",       mr_sel.get("close", ""))
        c2.metric("Dist to Sup", f"{mr_sel.get('dist_support_pct', '')}%")
        c3.metric("RSI3",        mr_sel.get("rsi3", ""))
        c3.metric("ATR14",       mr_sel.get("atr14", ""))
        c4.metric("EMA20 slope", f"{mr_sel.get('ema20_slope', '')}%")
        c4.metric("EMA50 slope", f"{mr_sel.get('ema50_slope', '')}%")
        show_chart(mr_sel["symbol"], sig=mr_sel, use_cache=use_cache)

    # PB chart panel — outside tabs
    pb_sel = st.session_state.get("pb_sel")
    if pb_sel:
        st.divider()
        direction = pb_sel.get("direction", "")
        st.subheader(f"Chart — {pb_sel['symbol']} ({pb_sel['signal']}) "
                     f"| PB {pb_sel.get('pullback_pct', '')}% | "
                     f"Consol {pb_sel.get('consol_range_pct', '')}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Consol High", pb_sel.get("consol_high", ""))
        c1.metric("Consol Low",  pb_sel.get("consol_low", ""))
        c2.metric("SL", pb_sel.get("sl", ""))
        c2.metric("TP", pb_sel.get("tp", ""))
        c3.metric("R:R",      pb_sel.get("rr", ""))
        c3.metric("Pullback", f"{pb_sel.get('pullback_pct', '')}%")
        show_chart(pb_sel["symbol"], sig=pb_sel, use_cache=use_cache)


if __name__ == "__main__":
    main()
