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
    d["ma50"]         = d["Close"].rolling(50).mean()
    d["ma200"]        = d["Close"].rolling(200).mean()
    # MA50 from 5 days ago
    d["ma50_prev5"]   = d["ma50"].shift(5)
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

    signal_type = "NR7_STRONG" if (not pd.isna(high20) and high > high20) else "NR7_EARLY"
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
        "high20":          round(high20, 2) if not pd.isna(high20) else None,
        "atr10":           round(atr10, 2),
        "ma50":            round(ma50, 2),
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
# SCAN — Reversal Hunter (Phương Án 2)
# ============================================================
def scan_reversal(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Signal candle = df.iloc[-1].
    Status is always PENDING (confirm requires next candle closing above high[-1]).
    SIZE relaxed to 1.2× ATR. Body filter removed.
    """
    if df is None or len(df) < 210:
        return None
    row  = df.iloc[-1]
    prev = df.iloc[-2]

    required = ["atr10", "avg_vol20", "ma50", "ma200"]
    if any(pd.isna(row[c]) for c in required):
        return None

    atr10     = row["atr10"]
    avg_vol20 = row["avg_vol20"]
    ma50      = row["ma50"]
    ma200     = row["ma200"]
    close = row["Close"]; open_ = row["Open"]
    high  = row["High"];  low   = row["Low"]
    vol   = row["Volume"]

    # [1] Price below MA50 (downtrend)
    if not (close < ma50):
        return None
    # [2] Near MA200 support (within 1× ATR10)
    if not (abs(close - ma200) <= 1.0 * atr10):
        return None
    # [3] Rejection — tested lower than prev low
    if not (low < prev["Low"]):
        return None
    # [4] Bull candle
    if not (close > open_):
        return None
    # [5] Close near high
    if not (close >= high * 0.998):
        return None
    # [6] Candle range > 1.2× ATR10 (relaxed from 1.5×, body filter removed)
    if not ((high - low) > 1.2 * atr10):
        return None
    # Filter: not more than 8% below MA200
    if close < ma200 * 0.92:
        return None

    vol_tier = _vol_tier(vol, avg_vol20, row.get("avg_vol_pre5", float("nan")))
    rs4w     = compute_rs4w(df, vnindex_df)
    sl       = round(low, 2)
    tp1      = round(ma50, 2)
    tp2      = round(close + 2.0 * atr10, 2)

    return {
        "signal":     "REVERSAL",
        "status":     "PENDING",
        "date":       df.index[-1],
        "close":      round(close, 2),
        "ma50":       round(ma50, 2),
        "ma200":      round(ma200, 2),
        "atr10":      round(atr10, 2),
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
        "vol_tier":   vol_tier,
        "rs4w":       rs4w,
        "volume":     int(vol),
        "vol_char":   compute_vol_character(df),
        "tight_days": count_tight_days(df),
        "weekly_ok":  check_weekly_trend(df),
        "avg_vol20": int(avg_vol20),
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
    "REVERSAL":        6,
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
            scan_breakout(df, vnindex_df) or
            scan_gap(df, vnindex_df)      or
            scan_nr7(df, vnindex_df)      or
            scan_reversal(df, vnindex_df)
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
        sig_order = _SIGNAL_PRIORITY.get(r["signal"], 9)
        vol_order = {"TIER1": 0, "TIER2": 1, "TIER3": 2}.get(r.get("vol_tier", ""), 3)
        rs_order  = 0 if (r.get("rs4w") or 0) >= 1.05 else 1
        supply_pen = 1 if r.get("supply_overhead") else 0
        return (sig_order, supply_pen, vol_order, rs_order, r["symbol"])

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
    ma_specs = [("ma50", "#f59e0b", "MA50"), ("ma200", "#818cf8", "MA200")]
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
        if "high20" in sig and not pd.isna(sig["high20"]):
            fig.add_hline(y=sig["high20"], line=dict(color="#fbbf24", dash="dot", width=1),
                          annotation_text="HIGH20", annotation_position="right",
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
        c1.metric("Close", sig["close"])
        c1.metric("SL", sig["sl"])
        if "tp" in sig:
            c2.metric("TP", sig["tp"])
            c2.metric("R:R", sig.get("rr", ""))
        if "tp1" in sig:
            c2.metric("TP1", sig["tp1"])
            c2.metric("TP2", sig.get("tp2", ""))
        c3.metric("ATR10", sig["atr10"])
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
        if sig.get("supply_overhead"): qf.append("⚠ Supply overhead")
        c4.metric("Quality", " · ".join(qf) if qf else "—")
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
def _render_results(rows: list[dict], use_cache: bool) -> None:
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
            "REVERSAL": "🔄",
        }.get(r["signal"], "")
        rs4w     = r.get("rs4w")
        rs_pct   = r.get("rs_pct")
        rs_str   = f"{rs4w:.2f}" + (f" ({rs_pct:.0f}%)" if rs_pct else "") if rs4w is not None else "—"
        rs_icon  = "🟢" if rs4w and rs4w >= 1.05 else ("🔴" if rs4w and rs4w < 0.95 else "")
        gap_str  = f" gap {r['gap_pct']:+.1f}%" if r.get("gap_pct") is not None else ""
        # Quality flags: A/D · tight days · weekly · supply overhead
        qf = []
        vc = r.get("vol_char", "")
        if vc == "ACCUM":   qf.append("A")
        elif vc == "DISTRIB": qf.append("D")
        td = r.get("tight_days", 0)
        if td >= 3: qf.append(f"{td}T")
        if r.get("weekly_ok"):       qf.append("W")
        if r.get("supply_overhead"): qf.append("⚠S")
        quality = "·".join(qf) if qf else "—"
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
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["sig_sel"] = rows[sel_rows[0]]
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
    st.caption("Breakout Momentum & Reversal Hunter | Scan sau khi nến ngày đóng cửa")

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

**🔄 Reversal**
- Giá < MA50, gần MA200 (±1× ATR)
- Test đáy thấp hơn rồi từ chối
- Cần confirm nến sau đóng > đỉnh tín hiệu

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
            "Breakout signals đã bị tắt. Chỉ hiển thị Reversal & Watchlist."
        )

    # ── Results
    results   = st.session_state.get("scan_results", [])
    watchlist = st.session_state.get("scan_watchlist", [])
    universe  = st.session_state.get("scan_universe", "")

    if results or watchlist:
        n_bo  = sum(1 for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY"))
        n_nr7 = sum(1 for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY"))
        n_gap = sum(1 for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY"))
        n_rev = sum(1 for r in results if r["signal"] == "REVERSAL")
        n_rs  = sum(1 for r in results if (r.get("rs4w") or 0) >= 1.05)
        st.subheader(
            f"Kết quả {universe} — {len(results)} tín hiệu "
            f"(🚀{n_bo} · 🔩{n_nr7} · ⚡{n_gap} · 🔄{n_rev} · 🟢RS{n_rs})"
        )

        tab_all, tab_bo, tab_nr7, tab_gap, tab_rev, tab_watch = st.tabs([
            f"Tất cả ({len(results)})",
            f"🚀 Breakout ({n_bo})",
            f"🔩 NR7 ({n_nr7})",
            f"⚡ Gap ({n_gap})",
            f"🔄 Reversal ({n_rev})",
            f"👀 Watchlist ({len(watchlist)})",
        ])
        with tab_all:
            _render_results(results, use_cache)
        with tab_bo:
            _render_results([r for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY")], use_cache)
        with tab_nr7:
            _render_results([r for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY")], use_cache)
        with tab_gap:
            _render_results([r for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY")], use_cache)
        with tab_rev:
            _render_results([r for r in results if r["signal"] == "REVERSAL"], use_cache)
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


if __name__ == "__main__":
    main()
