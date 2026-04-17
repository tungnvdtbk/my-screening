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
        # Always fetch from last cached date to now — ensures today's bar is included
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
    # Swing low on [-21] to [-2] — horizontal support for pin bar context
    d["swing_low20"]  = d["Low"].shift(2).rolling(20).min()
    # Swing high on [-21] to [-2] — for pin bar TP target
    d["swing_high20"] = d["High"].shift(2).rolling(20).max()

    # ── Pin bar quality indicators ──
    # ATR(30) for dead-market filter
    d["atr30"]        = tr.shift(2).rolling(30).mean()
    # RSI(14) — Wilder smoothing
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))
    # Pullback count: how many of [-4] to [-2] are bearish candles (close < open)
    bearish = (d["Close"] < d["Open"]).astype(int)
    d["pullback_count"] = bearish.shift(1) + bearish.shift(2) + bearish.shift(3)
    # Volume dry-up: how many of [-4] to [-2] have below-average volume
    below_avg = (d["Volume"] < d["avg_vol20"]).astype(int)
    d["vol_quiet_count"] = below_avg.shift(1) + below_avg.shift(2) + below_avg.shift(3)
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


def _pinbar_vol_tier(vol: float, avg_vol20: float, avg_vol_pre5: float) -> str:
    """Pin bar vol tier — uses 1.5x threshold (not 2.0x like breakout)."""
    if pd.isna(avg_vol20) or avg_vol20 == 0:
        return "NO_VOL_DATA"
    spike    = bool(vol > 1.5 * avg_vol20)
    contract = (not pd.isna(avg_vol_pre5)) and bool(avg_vol_pre5 < avg_vol20 * 0.80)
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
    entry       = round(close * 1.001, 2)   # 0.1% slippage
    tp          = round(entry + 2.0 * atr10, 2)
    rr          = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0

    # Quality metrics
    vol_char        = compute_vol_character(df)
    tight_days      = count_tight_days(df)
    weekly_ok       = check_weekly_trend(df)
    supply_overhead = has_overhead_supply(df, high, atr10)
    risk_pct        = (entry - sl) / entry * 100 if entry > 0 else 99

    # ── Quality Tier A/B gate ──
    # Tier A: BREAKOUT_STRONG + TIER1 vol + weekly trend + no overhead supply + R:R >= 2
    # Tier B: BREAKOUT_STRONG + (TIER1 or TIER2) vol + R:R >= 2
    if (signal_type == "BREAKOUT_STRONG" and vol_tier == "TIER1"
            and weekly_ok and not supply_overhead and rr >= 2.0):
        bo_tier = "A"
    elif (signal_type == "BREAKOUT_STRONG" and vol_tier in ("TIER1", "TIER2")
            and rr >= 2.0 and risk_pct < 5.0):
        bo_tier = "B"
    else:
        return None

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "bo_tier":         bo_tier,
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
        "vol_char":        vol_char,
        "tight_days":      tight_days,
        "weekly_ok":       weekly_ok,
        "supply_overhead": supply_overhead,
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
        if resist_atr >= 3.0:                    nr7_score += 25   # ample room above
        elif resist_atr >= 2.0:                  nr7_score += 15
        elif resist_atr >= 1.0:                  nr7_score += 8
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
    entry    = round(close * 1.001, 2)   # 0.1% slippage
    tp       = round(entry + 2.0 * atr10, 2)
    rr       = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0
    supply_overhead = has_overhead_supply(df, high, atr10)

    # ── Quality Tier A/B gate ──
    # Tier A: NR7_STRONG + inside bar + score >= 60 + room above resistance + R:R >= 2
    # Tier B: NR7_STRONG + score >= 40 + R:R >= 2
    if (signal_type == "NR7_STRONG" and is_ib and nr7_score >= 60
            and (resist_atr is None or resist_atr >= 1.0) and rr >= 2.0):
        nr7_tier = "A"
    elif (signal_type == "NR7_STRONG" and nr7_score >= 40 and rr >= 2.0):
        nr7_tier = "B"
    else:
        return None

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "nr7_tier":        nr7_tier,
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
        "supply_overhead": supply_overhead,
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
    entry       = round(close * 1.001, 2)   # 0.1% slippage
    tp          = round(entry + 2.0 * atr10, 2)
    rr          = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0
    weekly_ok       = check_weekly_trend(df)
    supply_overhead = has_overhead_supply(df, high, atr10)
    risk_pct        = (entry - sl) / entry * 100 if entry > 0 else 99

    # ── Quality Tier A/B gate ──
    # Tier A: GAP_STRONG + TIER1 vol + weekly trend + no overhead supply + R:R >= 2
    # Tier B: GAP_STRONG + (TIER1 or TIER2) vol + R:R >= 2 + risk < 5%
    if (signal_type == "GAP_STRONG" and vol_tier == "TIER1"
            and weekly_ok and not supply_overhead and rr >= 2.0):
        gap_tier = "A"
    elif (signal_type == "GAP_STRONG" and vol_tier in ("TIER1", "TIER2")
            and rr >= 2.0 and risk_pct < 5.0):
        gap_tier = "B"
    else:
        return None

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "gap_tier":        gap_tier,
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
        "weekly_ok":       weekly_ok,
        "supply_overhead": supply_overhead,
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
    if rs4w is not None and rs4w < 1.0:  return None   # [17] must outperform VNINDEX

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
    entry  = round(close * 1.001, 2)   # 0.1% slippage
    tp     = round(entry + 2.0 * atr10, 2)
    rr     = round((tp - entry) / (entry - sl), 2) if entry > sl else 0.0
    weekly_ok       = check_weekly_trend(df)
    supply_overhead = has_overhead_supply(df, high, atr10)
    risk_pct        = (entry - sl) / entry * 100 if entry > 0 else 99

    # ── Quality Tier A/B gate ──
    # Volume already validated by scanner (1.2x for MA20, 1.5x for MA50).
    # Use vol_spike ratio directly instead of vol_tier (which requires 2.0x — too strict).
    vol_spike = vol / max(avg_vol20, 1e-9)
    # Tier A: TF_MA20 + strong vol (1.5x+) + weekly + no overhead + R:R >= 2 + risk < 3%
    if (signal_type == "TF_MA20" and vol_spike >= 1.5
            and weekly_ok and not supply_overhead and rr >= 2.0 and risk_pct < 3.0):
        tf_tier = "A"
    # Tier B: TF_MA20 + vol already passed (1.2x+) + R:R >= 2 + risk < 5%
    elif (signal_type == "TF_MA20" and rr >= 2.0 and risk_pct < 5.0):
        tf_tier = "B"
    # Tier B: TF_MA50 + vol already passed (1.5x+) + weekly + R:R >= 2 + risk < 3%
    elif (signal_type == "TF_MA50" and weekly_ok and rr >= 2.0 and risk_pct < 3.0):
        tf_tier = "B"
    else:
        return None

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "tf_tier":         tf_tier,
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
        "weekly_ok":       weekly_ok,
        "supply_overhead": supply_overhead,
    }


# ============================================================
# SCAN — Pin Bar at Context (Price Action Rejection)
# ============================================================
def scan_pinbar(df: pd.DataFrame, vnindex_df=None, d1_trend_up: bool | None = None) -> dict | None:
    """
    Detect bullish pin bar at meaningful support — improved quality scoring.

    Quality score (0-13 points):
      +3  MTF trend aligned (D1 up for 4H scan, or MA50 rising for D1)
      +2  Pullback structure (>=2 bearish candles before pin bar)
      +2  Context confluence (>=2 support levels)
      +2  Volume spike (TIER1 or TIER2)
      +1  Volume dry-up (3 quiet bars before pin bar)
      +1  Bull close (close > open)
      +1  Body in upper 25% of range
      +1  RSI oversold (<40)

    Tier A: score >= 7   Tier B: score >= 4   Below 4: rejected
    d1_trend_up: for 4H scan, pass True/False from D1 data. None = skip MTF check.
    """
    if df is None or len(df) < 60:
        return None
    row = df.iloc[-1]

    required = ["atr10", "avg_vol20", "ma20", "ma50", "ma50_prev5", "swing_low20"]
    if any(pd.isna(row[c]) for c in required):
        return None

    close  = row["Close"]; open_ = row["Open"]
    high   = row["High"];  low   = row["Low"]
    vol    = row["Volume"]
    atr10  = row["atr10"]
    atr30  = row.get("atr30", float("nan"))
    ma20   = row["ma20"]
    ma50   = row["ma50"]
    ma200  = row.get("ma200", float("nan"))
    ma50_prev5   = row["ma50_prev5"]
    avg_vol20    = row["avg_vol20"]
    avg_vol_pre5 = row.get("avg_vol_pre5", float("nan"))
    swing_low20  = row["swing_low20"]
    swing_high20 = row.get("swing_high20", float("nan"))
    rsi14        = row.get("rsi14", float("nan"))
    pullback_cnt = row.get("pullback_count", 0)
    vol_quiet    = row.get("vol_quiet_count", 0)

    candle_range = high - low
    if candle_range <= 0:
        return None

    body        = abs(close - open_)
    lower_wick  = min(open_, close) - low
    upper_wick  = high - max(open_, close)

    # ── Pin bar shape detection ──
    if not (lower_wick >= 0.60 * candle_range):  return None   # [1] long lower wick
    if not (body <= 0.33 * candle_range):         return None   # [2] small body
    if not (upper_wick <= 0.25 * candle_range):   return None   # [3] short upper wick
    if not (candle_range >= 0.5 * atr10):         return None   # [4] minimum size

    # ── Safety filters ──
    if not pd.isna(ma200) and close < ma200 * 0.88:
        return None   # [X1] freefall
    # [X2] Dead market — ATR10 collapsed vs ATR30
    if not pd.isna(atr30) and atr30 > 0 and atr10 < 0.3 * atr30:
        return None

    # ── Body position filter — body must sit in upper 25% of range ──
    body_top = max(open_, close)
    body_in_upper = body_top >= high - 0.25 * candle_range

    # ── Context conditions — tighter proximity (0.3 * ATR) ──
    proximity = 0.3 * atr10
    contexts = []

    # [C1] AT_MA20 — pullback in uptrend
    if (abs(low - ma20) <= proximity
            and close > ma20
            and ma50 > ma50_prev5
            and close <= ma50 * 1.10):
        contexts.append("MA20")

    # [C2] AT_MA50
    if abs(low - ma50) <= proximity and close > ma50 * 0.98:
        contexts.append("MA50")

    # [C3] AT_MA200
    if not pd.isna(ma200):
        if abs(low - ma200) <= proximity and close >= ma200 * 0.98:
            contexts.append("MA200")

    # [C4] AT_SWING — 20-bar swing low
    if abs(low - swing_low20) <= proximity and close > swing_low20:
        contexts.append("SWING")

    if not contexts:
        return None   # no context = no signal

    context_count = len(contexts)

    # ── Signal type — priority: MA200 > MA50 > MA20 > SWING ──
    _ctx_priority = {"MA200": 0, "MA50": 1, "MA20": 2, "SWING": 3}
    primary_ctx = min(contexts, key=lambda c: _ctx_priority[c])
    signal_type = f"PINBAR_{primary_ctx}"

    # ── Volume tier (1.5x for pin bars) ──
    vol_tier = _pinbar_vol_tier(vol, avg_vol20, avg_vol_pre5)

    # ── SL / TP / R:R ──
    sl    = round(low, 2)
    entry = round(close * 1.001, 2)
    risk  = entry - sl
    if risk <= 0:
        return None

    # TP1 = MA50 (if below MA50)
    tp_ma50 = round(ma50, 2) if close < ma50 else 0
    # TP2 = prior swing high (if available and above entry)
    tp_swing = 0
    if not pd.isna(swing_high20) and swing_high20 > entry:
        tp_swing = round(swing_high20, 2)
    # TP3 = entry + 2 × ATR10
    tp_rr2  = round(entry + 2.0 * atr10, 2)
    tp      = max(tp_ma50, tp_swing, tp_rr2)
    rr      = round((tp - entry) / risk, 2)

    if rr < 2.0:
        return None

    risk_pct   = round(risk / entry * 100, 2) if entry > 0 else 99
    if risk_pct > 7.0:
        return None

    bull_close  = close > open_
    rsi_oversold = (not pd.isna(rsi14)) and rsi14 < 40
    has_pullback = (not pd.isna(pullback_cnt)) and pullback_cnt >= 2
    has_vol_dryup = (not pd.isna(vol_quiet)) and vol_quiet >= 3
    rs4w        = compute_rs4w(df, vnindex_df)

    # ── MTF trend alignment ──
    if d1_trend_up is not None:
        mtf_aligned = d1_trend_up
    else:
        # D1 scan: check own MA50 rising as trend proxy
        mtf_aligned = bool(ma50 > ma50_prev5)

    # ── Quality scoring (0-13 points) ──
    score = 0
    score_details = []
    if mtf_aligned:
        score += 3; score_details.append("trend+3")
    if has_pullback:
        score += 2; score_details.append(f"pb{int(pullback_cnt)}+2")
    if context_count >= 2:
        score += 2; score_details.append(f"ctx{context_count}+2")
    if vol_tier in ("TIER1", "TIER2"):
        score += 2; score_details.append(f"{vol_tier}+2")
    if has_vol_dryup:
        score += 1; score_details.append("quiet+1")
    if bull_close:
        score += 1; score_details.append("bull+1")
    if body_in_upper:
        score += 1; score_details.append("pos+1")
    if rsi_oversold:
        score += 1; score_details.append(f"rsi{rsi14:.0f}+1")

    # ── Tier gate ──
    if score >= 7:
        pin_tier = "A"
    elif score >= 4:
        pin_tier = "B"
    else:
        return None   # below quality threshold

    wick_ratio  = round(lower_wick / candle_range, 2)
    body_ratio  = round(body / candle_range, 2)
    context_str = "+".join(contexts)

    return {
        "signal":          signal_type,
        "date":            df.index[-1],
        "close":           round(close, 2),
        "status":          "PENDING",
        "pin_tier":        pin_tier,
        "pin_score":       score,
        "score_detail":    " ".join(score_details),
        "context":         context_str,
        "context_count":   context_count,
        "wick_ratio":      wick_ratio,
        "body_ratio":      body_ratio,
        "body_pos_upper":  body_in_upper,
        "pullback":        int(pullback_cnt) if not pd.isna(pullback_cnt) else 0,
        "rsi14":           round(rsi14, 1) if not pd.isna(rsi14) else None,
        "mtf_trend":       mtf_aligned,
        "ma20":            round(ma20, 2),
        "ma50":            round(ma50, 2),
        "ma200":           round(ma200, 2) if not pd.isna(ma200) else None,
        "sl":              sl,
        "tp":              tp,
        "rr":              rr,
        "atr10":           round(atr10, 2),
        "vol_tier":        vol_tier,
        "vol_quiet":       has_vol_dryup,
        "rs4w":            rs4w,
        "volume":          int(vol),
        "avg_vol20":       int(avg_vol20),
    }


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
    "PINBAR_MA200":    6,
    "PINBAR_MA50":     7,
    "PINBAR_MA20":     8,
    "PINBAR_SWING":    9,
    "TF_MA20":         10,
    "TF_MA50":         11,
}

def run_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
) -> tuple[list[dict], bool]:
    """
    Scan all symbols in parallel.
    Returns (signals, market_downtrend).
    """
    signals: list[dict] = []
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
            scan_pinbar(df, vnindex_df)        or
            scan_trend_filter(df, vnindex_df)
        )
        if sig:
            sig["symbol"] = sym.replace(".VN", "")
            sig["sector"] = symbols.get(sym, "")
            return ("signal", sig)
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
        sig_order  = _SIGNAL_PRIORITY.get(r["signal"], 99)
        vol_order  = {"TIER1": 0, "TIER2": 1, "TIER3": 2}.get(r.get("vol_tier", ""), 3)
        rs_order   = 0 if (r.get("rs4w") or 0) >= 1.05 else 1
        supply_pen = 1 if r.get("supply_overhead") else 0
        nr7_rank   = -r.get("nr7_score", 0)   # higher score = earlier in list
        return (sig_order, supply_pen, nr7_rank, vol_order, rs_order, r["symbol"])

    _assign_rs_pct(signals)

    return sorted(signals, key=_sig_sort), market_downtrend


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
        ctx = sig.get("context", "")
        if ctx: qf.append(f"@{ctx}")
        wr = sig.get("wick_ratio")
        if wr is not None: qf.append(f"Wick {wr:.0%}")
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
            "PINBAR_MA200": "📍", "PINBAR_MA50": "📍",
            "PINBAR_MA20": "📍",  "PINBAR_SWING": "📍",
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
        # Pin bar-specific flags
        ctx = r.get("context", "")
        if ctx:
            qf.append(ctx)
        ps = r.get("pin_score")
        if ps is not None:
            qf.append(f"Q{ps}/13")
        wr = r.get("wick_ratio")
        if wr is not None:
            qf.append(f"W{wr:.0%}")
        pb = r.get("pullback", 0)
        if pb >= 2:
            qf.append(f"PB{pb}")
        rsi = r.get("rsi14")
        if rsi is not None and rsi < 40:
            qf.append(f"RSI{rsi:.0f}")
        if r.get("mtf_trend"):
            qf.append("MTF")
        if r.get("status") == "PENDING":
            qf.append("PEND")
        quality = "·".join(qf) if qf else "—"
        touched = r.get("touched_ma", "")
        # Tier from any of the scanners
        tier = (r.get("bo_tier") or r.get("nr7_tier") or r.get("gap_tier")
                or r.get("pin_tier") or r.get("tf_tier") or "")
        table_rows.append({
            "Mã":      r["symbol"],
            "Tier":    tier,
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
    if close < ema50 and ema20 < ema50 and e50s < -0.015:
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
    pir    = float(row["mr_position_in_range"])
    rsi3   = float(row["mr_rsi3"])
    close  = float(row["Close"])
    range_low = float(row["mr_range_low"])
    range_high = float(row["mr_range_high"])

    # ── Quality Tier A/B gate ──
    # Compute R:R with 0.1% slippage (entry=close*1.001, SL=range_low*0.98, TP=range_high)
    mr_entry = round(close * 1.001, 2)   # 0.1% slippage
    mr_sl = round(range_low * 0.98, 2)
    mr_risk = mr_entry - mr_sl
    mr_reward = range_high - mr_entry
    mr_rr = mr_reward / max(mr_risk, 1e-9) if mr_risk > 0 else 0

    # Tier A: strongest reversal + deep near support + oversold + R:R >= 2
    # Tier B: any decent reversal + near support + R:R >= 2
    if (reversal in ("CLOSE_ABOVE_PREV_HIGH", "HAMMER")
            and pir < 0.15 and rsi3 < 20 and mr_rr >= 2.0):
        mr_tier = "A"
    elif (reversal in ("CLOSE_ABOVE_PREV_HIGH", "HAMMER", "HIGHER_CLOSE_LOWER_LOW")
            and pir < 0.25 and mr_rr >= 2.0):
        mr_tier = "B"
    else:
        return None

    return {
        "signal":              "MR_LONG",
        "date":                df.index[-1],
        "close":               round(close, 2),
        "mr_tier":             mr_tier,
        "range_high":          round(range_high, 2),
        "range_low":           round(range_low, 2),
        "range_size_pct":      round(float(row["mr_range_size_pct"]) * 100, 1),
        "position_in_range":   round(pir * 100, 1),
        "dist_support_pct":    round(float(row["mr_dist_to_support"]) * 100, 2),
        "ema20":               round(float(row["mr_ema20"]), 2),
        "ema50":               round(float(row["mr_ema50"]), 2),
        "ema20_slope":         round(float(row["mr_ema20_slope"]) * 100, 3),
        "ema50_slope":         round(float(row["mr_ema50_slope"]) * 100, 3),
        "rsi3":                round(rsi3, 1),
        "atr14":               round(float(row["mr_atr14"]), 2),
        "avg_vol20":           int(row["mr_avg_vol20"]),
        "reversal_signal":     reversal,
        "final_score":         round(scores["final"], 3),
        "score_support":       round(scores["support"], 3),
        "score_reversal":      round(scores["reversal"], 3),
        "score_range":         round(scores["range"], 3),
        "sl":                  mr_sl,
        "tp":                  round(range_high, 2),
        "rr":                  round(mr_rr, 2),
        "rs4w":                rs4w,
        "vol_tier":            _vol_tier(float(row["Volume"]), float(row["mr_avg_vol20"]),
                                         float(row.get("avg_vol_pre5", float("nan")))),
    }


# ============================================================
# SWING FILTER — Professional Swing Scanner (per swing_scanner_rules_pro_v_2.md)
# ============================================================

def compute_swing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators required by the Swing Filter spec (Section 5)."""
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]
    open_ = d["Open"]

    d["sw_ma20"] = close.rolling(20).mean()
    d["sw_ma50"] = close.rolling(50).mean()
    d["sw_vol_ma20"] = volume.rolling(20).mean()
    d["sw_vol_ma5"] = volume.rolling(5).mean()
    d["sw_ret_5d"] = close.pct_change(5)
    d["sw_ret_2d"] = close.pct_change(2)
    d["sw_volatility10"] = close.pct_change().rolling(10).std()
    d["sw_distance_ma20"] = close / d["sw_ma20"]
    d["sw_volume_spike"] = volume / d["sw_vol_ma20"]
    d["sw_value"] = close * volume
    d["sw_high_20d"] = high.rolling(20).max()
    candle_range = (high - low).replace(0, 1e-9)
    d["sw_body_ratio"] = (close - open_).abs() / candle_range
    d["sw_range_pct"] = candle_range / close.replace(0, 1e-9)

    # RSI(14) — Wilder Smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["sw_rsi"] = 100 - (100 / (1 + rs))

    # Momentum Acceleration
    d["sw_mom_accel"] = (d["sw_ret_2d"] - (d["sw_ret_5d"] / 2)).clip(-0.30, 0.30)

    # Pivot high: shift(1).rolling(10).max — previous 10 bars' max high
    d["sw_pivot_high"] = high.shift(1).rolling(10).max()

    # Buildup components
    d["sw_tightness"] = close.rolling(8).std() / close.replace(0, 1e-9)
    d["sw_vol_dryup"] = d["sw_vol_ma5"] < d["sw_vol_ma20"] * 0.80
    d["sw_near_ma20"] = close.between(d["sw_ma20"] * 0.99, d["sw_ma20"] * 1.02)
    d["sw_higher_low"] = low > low.shift(3)
    d["sw_small_body"] = d["sw_body_ratio"].rolling(5).mean() < 0.50
    d["sw_range_contract"] = d["sw_range_pct"].rolling(5).mean() < d["sw_range_pct"].rolling(15).mean()

    down_day = close < open_
    distribution_day = down_day & (volume > d["sw_vol_ma20"] * 1.20)
    d["sw_no_distribution"] = distribution_day.rolling(5).sum() <= 1

    # Buildup score
    d["sw_buildup_score"] = (
        (d["sw_tightness"] < 0.025).astype(int)
        + d["sw_vol_dryup"].astype(int)
        + d["sw_near_ma20"].astype(int)
        + d["sw_higher_low"].astype(int)
        + d["sw_small_body"].astype(int)
        + d["sw_range_contract"].astype(int)
        + d["sw_no_distribution"].astype(int)
    )

    # Buildup definition
    d["sw_is_buildup"] = (
        (d["sw_tightness"] < 0.025)
        & d["sw_near_ma20"]
        & d["sw_no_distribution"]
        & (d["sw_buildup_score"] >= 4)
    )

    # Buildup persistence
    d["sw_buildup_days"] = d["sw_is_buildup"].astype(float).rolling(5).sum()

    # ATR(10) for adaptive TP/SL
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["sw_atr10"] = tr.rolling(10).mean()

    # Recent low for tighter SL
    d["sw_low_5d"] = low.rolling(5).min()

    return d


def swing_market_regime_ok(vnindex_df: pd.DataFrame | None) -> bool:
    """Section 6: Market Regime Gate. All conditions on latest bar."""
    if vnindex_df is None or len(vnindex_df) < 51:
        return False
    try:
        col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
        vn_close = vnindex_df[col].dropna()
        if len(vn_close) < 51:
            return False
        vn_ma20 = vn_close.rolling(20).mean()
        vn_ma50 = vn_close.rolling(50).mean()
        vn_ret_3d = vn_close.pct_change(3)
        last_close = float(vn_close.iloc[-1])
        last_ma20 = float(vn_ma20.iloc[-1])
        last_ma50 = float(vn_ma50.iloc[-1])
        last_ret3 = float(vn_ret_3d.iloc[-1])
        return (
            last_ma20 > last_ma50
            and last_close > last_ma50 * 0.97
            and last_ret3 > -0.03
        )
    except Exception:
        return False


def _swing_rs_vs_vni(df: pd.DataFrame, vnindex_df: pd.DataFrame | None) -> float | None:
    """Section 7: 20-day relative strength vs VNINDEX."""
    if vnindex_df is None or len(vnindex_df) < 21 or len(df) < 21:
        return None
    try:
        stock_perf = float(df["Close"].iloc[-1] / df["Close"].iloc[-21])
        col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
        idx_perf = float(vnindex_df[col].iloc[-1] / vnindex_df[col].iloc[-21])
        if idx_perf == 0:
            return None
        return round(stock_perf / idx_perf, 4)
    except Exception:
        return None


def scan_swing_filter(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Swing Filter scan for a single ticker.
    Applies hard filters (Section 8), buildup (Section 9),
    entry trigger (Section 10), and exit levels (Section 11).
    Returns signal dict or None.
    """
    if df is None or len(df) < 60:
        return None

    d = compute_swing_indicators(df)
    row = d.iloc[-1]

    required = [
        "sw_ma20", "sw_ma50", "sw_vol_ma20", "sw_vol_ma5",
        "sw_rsi", "sw_tightness", "sw_buildup_score",
        "sw_buildup_days", "sw_pivot_high", "sw_value",
        "sw_mom_accel", "sw_is_buildup", "sw_atr10", "sw_low_5d",
    ]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    volume = float(row["Volume"])
    ma20 = float(row["sw_ma20"])
    ma50 = float(row["sw_ma50"])
    rsi = float(row["sw_rsi"])
    value = float(row["sw_value"])
    vol_ma20 = float(row["sw_vol_ma20"])
    pivot_high = float(row["sw_pivot_high"])
    mom_accel = float(row["sw_mom_accel"])
    tightness = float(row["sw_tightness"])
    buildup_score = int(row["sw_buildup_score"])
    buildup_days = float(row["sw_buildup_days"])
    is_buildup = bool(row["sw_is_buildup"])
    atr10 = float(row["sw_atr10"])
    low_5d = float(row["sw_low_5d"])

    # Section 7: Relative strength
    rs_vs_vni = _swing_rs_vs_vni(df, vnindex_df)

    # Section 8: Per-Stock Hard Filters (fail-fast order)
    if value <= 1e10:
        return None
    if not (close > ma50):
        return None
    if not (ma20 > ma50):
        return None
    if not (close > ma50 * 0.97):
        return None
    if close > ma20 * 1.05:
        return None
    if close > ma50 * 1.15:
        return None
    if rsi >= 72:
        return None
    if rs_vs_vni is not None and rs_vs_vni <= 1.0:
        return None
    if not is_buildup:
        return None
    if buildup_days < 3:
        return None
    # FIX 3: Require buildup_score >= 5 (buildup=4 had 0% WR)
    if buildup_score < 5:
        return None

    # Section 10: Entry Trigger
    price_break = close > pivot_high
    vol_expand = volume > vol_ma20 * 1.5
    candle_range = high - low
    strong_bar = ((close - low) / max(candle_range, 1e-9)) > 0.5
    mom_ok = mom_accel > 0

    entry_confirmed = price_break and vol_expand and strong_bar
    # FIX 4: Require mom_accel > 0 as mandatory (accelerating momentum)
    if not mom_ok:
        return None

    if not entry_confirmed:
        return None

    trigger_score = int(price_break) + int(vol_expand) + int(strong_bar) + int(mom_ok)

    # Section 11: Entry / Exit levels
    # SL: spec base (ma50*0.97) capped at max 5% from entry
    entry = round(close * 1.001, 2)   # 0.1% slippage
    sl_base = ma50 * 0.97
    sl_cap = entry * 0.95
    stop_loss = round(max(sl_base, sl_cap), 2)
    # TP: fixed 7% / 12% (reliable across different volatility regimes)
    target_1 = round(entry * 1.07, 2)
    target_2 = round(entry * 1.12, 2)
    rr_ratio = round((target_1 - entry) / max(entry - stop_loss, 1e-9), 2)

    if rr_ratio < 1.5:
        return None

    # Quality tier — target A=100% WR, B=>50% WR with R:R >= 2
    risk_pct = (entry - stop_loss) / entry * 100
    higher_low = bool(row["sw_higher_low"])
    range_contract = bool(row["sw_range_contract"])
    vol_dryup = bool(row["sw_vol_dryup"])
    # Tier A: tight risk + range contraction + volume dryup + higher low (strongest buildup)
    if (is_buildup and risk_pct < 2.5 and range_contract
            and vol_dryup and higher_low and rr_ratio >= 2.0):
        sw_tier = "A"
    # Tier B: buildup + low risk + contraction + R:R >= 2
    elif (is_buildup and risk_pct < 4.0 and range_contract and rr_ratio >= 2.0):
        sw_tier = "B"
    else:
        return None

    # Collect all data for cross-sectional scoring later
    return {
        "signal": "SWING_FILTER",
        "date": df.index[-1],
        "close": round(close, 2),
        "sw_tier": sw_tier,
        "rsi": round(rsi, 1),
        "tightness": round(tightness, 4),
        "near_ma20": bool(row["sw_near_ma20"]),
        "buildup_score": buildup_score,
        "buildup_days": int(buildup_days),
        "vol_dryup": bool(row["sw_vol_dryup"]),
        "range_contract": bool(row["sw_range_contract"]),
        "no_distribution": bool(row["sw_no_distribution"]),
        "rs_vs_vni_20": rs_vs_vni,
        "price_break": price_break,
        "vol_expand": vol_expand,
        "strong_bar": strong_bar,
        "mom_accel": round(mom_accel, 4),
        "trigger_score": trigger_score,
        "entry_confirmed": entry_confirmed,
        "sl": stop_loss,
        "tp": target_1,
        "tp2": target_2,
        "rr": rr_ratio,
        "atr10": round(atr10, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "value": round(value, 0),
        "volume_spike": round(float(row["sw_volume_spike"]), 2),
        "distance_ma20": round(float(row["sw_distance_ma20"]), 4),
        "pivot_high": round(pivot_high, 2),
        "volume": int(volume),
        "avg_vol20": int(vol_ma20),
        # raw features for cross-sectional scoring
        "_ret_5d": float(row["sw_ret_5d"]) if not pd.isna(row["sw_ret_5d"]) else 0,
        "_volume_spike": float(row["sw_volume_spike"]) if not pd.isna(row["sw_volume_spike"]) else 0,
        "_distance_ma20": float(row["sw_distance_ma20"]) if not pd.isna(row["sw_distance_ma20"]) else 1,
        "_tightness": tightness,
        "_buildup_score": buildup_score,
        "_trigger_score": trigger_score,
        "_rs_vs_vni_20": rs_vs_vni if rs_vs_vni is not None else 1.0,
    }


def _swing_cross_sectional_score(candidates: list[dict]) -> None:
    """
    Section 12: Cross-sectional percentile ranking and final score.
    Mutates dicts in-place, adding 'score' field.
    """
    if not candidates:
        return
    from scipy.stats import rankdata

    n = len(candidates)
    if n == 1:
        candidates[0]["score"] = 0.50
        return

    def pct_rank(values):
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        return rankdata(arr) / len(arr)

    ret_5d = [c["_ret_5d"] for c in candidates]
    vol_spike = [c["_volume_spike"] for c in candidates]
    dist_ma20 = [-(c["_distance_ma20"] - 1.0) if c["_distance_ma20"] > 1.0 else 0 for c in candidates]
    tightness = [-c["_tightness"] for c in candidates]
    buildup = [c["_buildup_score"] for c in candidates]
    trigger = [c["_trigger_score"] for c in candidates]
    rs = [c["_rs_vs_vni_20"] for c in candidates]

    r_momentum = pct_rank(ret_5d)
    r_vol_spike = pct_rank(vol_spike)
    r_ma20_dist = pct_rank(dist_ma20)
    r_tightness = pct_rank(tightness)
    r_buildup = pct_rank(buildup)
    r_trigger = pct_rank(trigger)
    r_rs = pct_rank(rs)

    for i, c in enumerate(candidates):
        c["score"] = round(
            0.20 * r_momentum[i]
            + 0.10 * r_vol_spike[i]
            + 0.10 * r_ma20_dist[i]
            + 0.15 * r_tightness[i]
            + 0.20 * r_buildup[i]
            + 0.15 * r_trigger[i]
            + 0.10 * r_rs[i],
            3,
        )


def run_swing_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
    bypass_market_gate: bool = False,
) -> list[dict]:
    """
    Run Swing Filter across all symbols.
    Returns top 10 candidates sorted by cross-sectional score.
    """
    # Section 6: Market regime gate
    if not bypass_market_gate and not swing_market_regime_ok(vnindex_df):
        return []

    candidates: list[dict] = []
    total = len(symbols)
    done = 0

    def _one(sym: str) -> dict | None:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty or len(df) < 60:
            return None
        sig = scan_swing_filter(df, vnindex_df)
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
                    candidates.append(res)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    # Section 12: Cross-sectional scoring
    _swing_cross_sectional_score(candidates)

    # Section 13: Top 10 by score
    candidates.sort(key=lambda c: -c.get("score", 0))
    return candidates[:10]


# ── Swing Filter results table ─────────────────────────────────────
def _render_swing_results(rows: list[dict], use_cache: bool, key: str = "sw_table") -> None:
    if not rows:
        st.info("Khong co tin hieu Swing Filter.")
        return

    st.caption(
        "Swing Filter — Tier A/B quality | "
        "Constructive buildup + breakout confirmation | Sorted by Score"
    )

    table_rows = []
    for r in rows:
        rs = r.get("rs_vs_vni_20")
        rs_str = f"{rs:.3f}" if rs is not None else "—"
        table_rows.append({
            "Ma": r["symbol"],
            "Tier": r.get("sw_tier", ""),
            "Score": f"{r.get('score', 0):.3f}",
            "Gia": r["close"],
            "RSI": r.get("rsi", ""),
            "Buildup": f"{r.get('buildup_score', '')}/7 ({r.get('buildup_days', '')}d)",
            "Trigger": f"{r.get('trigger_score', '')}/4",
            "Tightness": f"{r.get('tightness', 0):.4f}",
            "RS20": rs_str,
            "Vol Spike": f"{r.get('volume_spike', 0):.1f}x",
            "SL": r.get("sl", ""),
            "TP1": r.get("tp", ""),
            "TP2": r.get("tp2", ""),
            "R:R": r.get("rr", ""),
            "Nganh": r.get("sector", ""),
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
            st.session_state["sw_sel"] = rows[sel_rows[0]]
            st.session_state.pop("sig_sel", None)

            st.session_state.pop("mr_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# PRICE ACTION SCANNER — Breakout & Pullback (v2.1)
# per price_action_scanner_breakout_pullback_v2.md
# ============================================================

def compute_pa_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators for the Price Action scanner (Sections 4-6)."""
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    open_ = d["Open"]
    volume = d["Volume"]

    # Section 4: Core indicators
    d["pa_ma20"] = close.rolling(20).mean()
    d["pa_ma50"] = close.rolling(50).mean()
    d["pa_vol_ma20"] = volume.rolling(20).mean()
    d["pa_vol_ma5"] = volume.rolling(5).mean()
    d["pa_value"] = close * volume
    d["pa_value_avg5"] = d["pa_value"].rolling(5).mean()

    d["pa_ma20_slope"] = d["pa_ma20"] - d["pa_ma20"].shift(5)
    d["pa_ma50_slope"] = d["pa_ma50"] - d["pa_ma50"].shift(10)

    bar_range = (high - low).replace(0, 1e-9)
    d["pa_bar_range"] = bar_range
    d["pa_range_pct"] = bar_range / close.replace(0, 1e-9)
    d["pa_body_ratio"] = (close - open_).abs() / bar_range
    d["pa_upper_tail"] = (high - close) / bar_range

    d["pa_pivot_10"] = high.shift(1).rolling(10).max()
    d["pa_pivot_20"] = high.shift(1).rolling(20).max()
    d["pa_low_3"] = low.shift(1).rolling(3).min()
    d["pa_low_5"] = low.shift(1).rolling(5).min()

    d["pa_ma20_distance"] = close / d["pa_ma20"].replace(0, 1e-9)
    d["pa_range_avg20"] = d["pa_range_pct"].rolling(20).mean()
    d["pa_range_expansion"] = d["pa_range_pct"] / d["pa_range_avg20"].replace(0, 1e-9)

    d["pa_ret_5d"] = close.pct_change(5)
    d["pa_ret_2d"] = close.pct_change(2)

    # RSI(14) — Wilder Smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["pa_rsi"] = 100 - (100 / (1 + rs))

    # Momentum Acceleration
    d["pa_mom_accel"] = (d["pa_ret_2d"] - (d["pa_ret_5d"] / 2)).clip(-0.30, 0.30)

    # Barrier Cluster Detection (Volman)
    def _barrier_touches(w):
        if len(w) == 0:
            return 0
        mx = w.max()
        return ((w >= mx * 0.985) & (w <= mx)).sum()

    d["pa_barrier_touches"] = high.shift(1).rolling(20).apply(_barrier_touches, raw=False)
    d["pa_strong_barrier"] = d["pa_barrier_touches"] >= 3

    # Squeeze Detection (Volman)
    d["pa_support_rising"] = d["pa_low_5"] > d["pa_low_5"].shift(5)
    d["pa_resist_flat"] = d["pa_pivot_20"] <= d["pa_pivot_20"].shift(5) * 1.005
    d["pa_is_squeeze"] = d["pa_support_rising"] & d["pa_resist_flat"]

    # Section 5: Trend filter
    d["pa_trend_filter"] = (
        (close > d["pa_ma20"])
        & (d["pa_ma20"] > d["pa_ma50"])
        & (d["pa_ma20_slope"] > 0)
        & (d["pa_ma50_slope"] >= 0)
    )

    # Section 6: Buildup
    d["pa_tightness"] = close.rolling(8).std() / close.replace(0, 1e-9)
    d["pa_tight_closes"] = close.rolling(5).std() / close.replace(0, 1e-9)
    d["pa_vol_dryup"] = d["pa_vol_ma5"] < d["pa_vol_ma20"] * 0.80
    d["pa_near_ma20"] = close.between(d["pa_ma20"] * 0.99, d["pa_ma20"] * 1.02)
    d["pa_higher_low"] = low > low.shift(3)
    d["pa_small_body"] = d["pa_body_ratio"].rolling(5).mean() < 0.55
    d["pa_range_contract"] = d["pa_range_pct"].rolling(5).mean() < d["pa_range_pct"].rolling(15).mean()
    d["pa_below_pivot"] = close <= d["pa_pivot_20"] * 1.02

    down_day = close < open_
    distribution_day = down_day & (volume > d["pa_vol_ma20"] * 1.20)
    d["pa_no_distribution"] = distribution_day.rolling(5).sum() <= 1

    d["pa_buildup_score"] = (
        (d["pa_tightness"] < 0.025).astype(int)
        + (d["pa_tight_closes"] < 0.020).astype(int)
        + d["pa_vol_dryup"].astype(int)
        + d["pa_near_ma20"].astype(int)
        + d["pa_higher_low"].astype(int)
        + d["pa_small_body"].astype(int)
        + d["pa_range_contract"].astype(int)
        + d["pa_below_pivot"].astype(int)
        + d["pa_no_distribution"].astype(int)
        + d["pa_strong_barrier"].astype(int)
        + d["pa_is_squeeze"].astype(int)
    )

    d["pa_is_buildup"] = (
        d["pa_near_ma20"]
        & d["pa_below_pivot"]
        & d["pa_no_distribution"]
        & (d["pa_tightness"] < 0.025)
        & (d["pa_buildup_score"] >= 5)
    )

    d["pa_buildup_days"] = d["pa_is_buildup"].astype(float).rolling(5).sum()

    # Limit-up detection (HOSE ±7%)
    ref_price = close.shift(1)
    ceil_price = ref_price * 1.07
    d["pa_hit_limit_up"] = close >= ceil_price * 0.998
    d["pa_half_day"] = volume < d["pa_vol_ma20"] * 0.50

    # Pullback structure
    d["pa_prior_push"] = close.shift(3) > close.shift(8)
    d["pa_pullback_depth"] = (close.rolling(5).max() - low) / close.replace(0, 1e-9)

    return d


def pa_market_regime_ok(vnindex_df: pd.DataFrame | None) -> bool:
    """Section 5.5: VNINDEX market regime gate."""
    if vnindex_df is None or len(vnindex_df) < 51:
        return False
    try:
        col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
        vn_close = vnindex_df[col].dropna()
        if len(vn_close) < 51:
            return False
        vn_ma20 = vn_close.rolling(20).mean()
        vn_ma50 = vn_close.rolling(50).mean()
        vn_ret3d = vn_close.pct_change(3)
        return (
            float(vn_close.iloc[-1]) > float(vn_ma50.iloc[-1]) * 0.97
            and float(vn_ma20.iloc[-1]) > float(vn_ma50.iloc[-1])
            and float(vn_ret3d.iloc[-1]) > -0.03
        )
    except Exception:
        return False


def _pa_rs_vs_vni(df: pd.DataFrame, vnindex_df: pd.DataFrame | None) -> float | None:
    """Section 4: 20-day relative strength vs VNINDEX."""
    if vnindex_df is None or len(vnindex_df) < 21 or len(df) < 21:
        return None
    try:
        stock_perf = float(df["Close"].iloc[-1] / df["Close"].iloc[-21])
        col = "Close" if "Close" in vnindex_df.columns else vnindex_df.columns[3]
        idx_perf = float(vnindex_df[col].iloc[-1] / vnindex_df[col].iloc[-21])
        if idx_perf == 0:
            return None
        return round(stock_perf / idx_perf, 4)
    except Exception:
        return None


def scan_pa(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Price Action scan for a single ticker.
    Returns breakout or pullback signal, or None.
    """
    if df is None or len(df) < 60:
        return None

    d = compute_pa_indicators(df)
    row = d.iloc[-1]

    required = [
        "pa_ma20", "pa_ma50", "pa_vol_ma20", "pa_rsi", "pa_tightness",
        "pa_buildup_score", "pa_buildup_days", "pa_pivot_20", "pa_value_avg5",
        "pa_mom_accel", "pa_is_buildup", "pa_trend_filter",
        "pa_ma20_slope", "pa_ma50_slope", "pa_range_expansion",
    ]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    volume = float(row["Volume"])
    ma20 = float(row["pa_ma20"])
    ma50 = float(row["pa_ma50"])
    rsi = float(row["pa_rsi"])
    value_avg5 = float(row["pa_value_avg5"])
    vol_ma20 = float(row["pa_vol_ma20"])
    pivot_20 = float(row["pa_pivot_20"])
    mom_accel = float(row["pa_mom_accel"])
    tightness = float(row["pa_tightness"])
    buildup_score = int(row["pa_buildup_score"])
    buildup_days = float(row["pa_buildup_days"])
    is_buildup = bool(row["pa_is_buildup"])
    trend_ok = bool(row["pa_trend_filter"])
    bar_range = float(row["pa_bar_range"])
    range_expansion = float(row["pa_range_expansion"])
    upper_tail = float(row["pa_upper_tail"])
    strong_barrier = bool(row["pa_strong_barrier"])
    is_squeeze = bool(row["pa_is_squeeze"])

    rs_vs_vni = _pa_rs_vs_vni(df, vnindex_df)

    # Section 7: Hard filters
    if not trend_ok:
        return None
    if value_avg5 <= 1e10:
        return None
    if not is_buildup:
        return None
    if buildup_days < 3:
        return None
    if close > ma20 * 1.05:
        return None
    if close > ma50 * 1.15:
        return None
    if rsi >= 72:
        return None
    if rs_vs_vni is not None and rs_vs_vni <= 1.0:
        return None
    # Limit-up / half-day rejection
    if bool(row.get("pa_hit_limit_up", False)):
        return None
    if bool(row.get("pa_half_day", False)):
        return None

    # ── Try Setup A: Breakout After Buildup (Section 8) ──
    price_break = close > pivot_20
    break_distance = close / max(pivot_20, 1e-9)
    clean_break = break_distance <= 1.03
    # FIX 1: Relax vol threshold 1.5x→1.3x (VN ATC/block trades inflate avg;
    #         1.5x was too strict — only 1 breakout in 2yr backtest)
    vol_expand = volume > vol_ma20 * 1.30
    strong_close = ((close - low) / max(bar_range, 1e-9)) > 0.60
    small_upper = upper_tail < 0.35
    # FIX 5: Relax range_expansion 2.0→2.5 (VN stocks gap frequently on news)
    acceptable_bar = range_expansion <= 2.50

    setup_breakout = (
        price_break and clean_break and vol_expand
        and strong_close and small_upper and acceptable_bar
    )

    # ── Try Setup B: Pullback Continuation (Section 9) ──
    prior_push = bool(row.get("pa_prior_push", False))
    pullback_depth = float(row.get("pa_pullback_depth", 0))
    pullback_exists = pullback_depth > 0.02
    # down days in last 4
    down_days = 0
    for i in range(1, min(5, len(d))):
        if i < len(d) - 1 and float(d.iloc[-i]["Close"]) < float(d.iloc[-i - 1]["Close"]):
            down_days += 1
    down_days_ok = down_days >= 1
    pullback_to_ma20 = low <= ma20 * 1.02
    close_holds_ma20 = close >= ma20 * 0.99
    shallow_pullback = close >= ma50
    no_deep_damage = low >= ma50 * 0.98

    bull_reclaim = close > float(d.iloc[-2]["High"]) if len(d) >= 2 else False
    pb_strong_close = ((close - low) / max(bar_range, 1e-9)) > 0.55
    reversal_bar = (close > open_) and bull_reclaim and pb_strong_close
    # FIX 3: Pullback needs at least avg volume to confirm demand is real
    vol_ok = volume >= vol_ma20 * 1.00

    setup_pullback = (
        prior_push and pullback_exists and down_days_ok
        and pullback_to_ma20 and close_holds_ma20
        and shallow_pullback and no_deep_damage
        and reversal_bar and vol_ok
        # FIX 4: Require buildup_score >= 6 for pullback (weaker pullbacks lose)
        and buildup_score >= 6
    )

    # Section 10: Label
    if setup_breakout:
        setup_type = "PA_BREAKOUT"
    elif setup_pullback:
        setup_type = "PA_PULLBACK"
    else:
        return None

    # Section 8/9: Trigger score (6 points max)
    trigger_score = (
        int(price_break) + int(vol_expand) + int(strong_close)
        + int(mom_accel > 0) + int(strong_barrier) + int(is_squeeze)
    )
    # For pullback: bonus if volume > 1.3x avg (strong demand at support)
    pb_vol_bonus = setup_type == "PA_PULLBACK" and volume > vol_ma20 * 1.30
    if pb_vol_bonus:
        trigger_score = min(trigger_score + 1, 6)

    # Section 12: Exit levels
    # T+2.5 note: 58% of losses hit Day 1-2 (unsellable in VN).
    # Use wider SL to survive early noise — position size accordingly.
    entry = round(close * 1.001, 2)   # 0.1% slippage
    if setup_type == "PA_BREAKOUT":
        # SL = max(signal low, MA20 - 3%) — gives room for Day 1-2 noise
        stop_loss = round(max(low, ma20 * 0.97), 2)
        target_1 = round(entry * 1.07, 2)
        target_2 = round(entry * 1.12, 2)
    else:  # pullback
        # SL = max(signal low, MA50 - 2%) for pullback
        stop_loss = round(max(low, ma50 * 0.98), 2)
        target_1 = round(entry * 1.07, 2)
        target_2 = round(entry * 1.10, 2)

    # FIX: Require trigger_score >= 3 (trigger=2 had 0% WR)
    if trigger_score < 3:
        return None

    rr_ratio = round((target_1 - entry) / max(entry - stop_loss, 1e-9), 2)
    if rr_ratio < 1.5:
        return None

    volume_spike = volume / max(vol_ma20, 1e-9)
    barrier_touches = int(row.get("pa_barrier_touches", 0))

    # Quality tier from backtest analysis (R:R >= 2 win rate)
    risk_pct = (entry - stop_loss) / entry * 100
    if tightness < 0.012 and volume_spike >= 2.0 and not is_squeeze:
        pa_tier = "A"   # 100% WR in backtest (tight + vol spike + no squeeze)
    elif (tightness < 0.012 and risk_pct < 3.0
          and not is_squeeze and setup_type == "PA_BREAKOUT"):
        pa_tier = "B"   # 67% WR in backtest
    else:
        return None

    return {
        "signal": setup_type,
        "date": df.index[-1],
        "close": round(close, 2),
        "setup_type": setup_type,
        "pa_tier": pa_tier,
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma20_slope": round(float(row["pa_ma20_slope"]), 4),
        "rsi": round(rsi, 1),
        "rs_vs_vni": rs_vs_vni,
        "buildup_score": buildup_score,
        "buildup_days": int(buildup_days),
        "tightness": round(tightness, 4),
        "strong_barrier": strong_barrier,
        "barrier_touches": barrier_touches,
        "is_squeeze": is_squeeze,
        "trigger_score": trigger_score,
        "price_break": price_break,
        "vol_expand": vol_expand,
        "mom_accel": round(mom_accel, 4),
        "sl": stop_loss,
        "tp": target_1,
        "tp2": target_2,
        "rr": rr_ratio,
        "volume": int(volume),
        "avg_vol20": int(vol_ma20),
        "volume_spike": round(volume_spike, 2),
        "pivot_20": round(pivot_20, 2),
        "range_expansion": round(range_expansion, 2),
        "upper_tail": round(upper_tail, 2),
        # raw features for cross-sectional scoring
        "_ret_5d": float(row["pa_ret_5d"]) if not pd.isna(row["pa_ret_5d"]) else 0,
        "_volume_spike": volume_spike,
        "_buildup_score": buildup_score,
        "_tightness": tightness,
        "_close_quality": (close - low) / max(bar_range, 1e-9),
        "_extension_penalty": max(float(row["pa_ma20_distance"]) - 1.0, 0),
        "_rs_vs_vni": rs_vs_vni if rs_vs_vni is not None else 1.0,
        "_trigger_score": trigger_score,
    }


def _pa_cross_sectional_score(candidates: list[dict]) -> None:
    """Section 11: Cross-sectional percentile ranking. Mutates dicts in-place."""
    if not candidates:
        return
    from scipy.stats import rankdata

    n = len(candidates)
    if n == 1:
        candidates[0]["score"] = 0.50
        return

    def pct_rank(values):
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        return rankdata(arr) / len(arr)

    r_momentum = pct_rank([c["_ret_5d"] for c in candidates])
    r_vol_spike = pct_rank([c["_volume_spike"] for c in candidates])
    r_buildup = pct_rank([c["_buildup_score"] for c in candidates])
    r_tightness = pct_rank([-c["_tightness"] for c in candidates])
    r_close_q = pct_rank([c["_close_quality"] for c in candidates])
    r_ext = pct_rank([-c["_extension_penalty"] for c in candidates])
    r_rs = pct_rank([c["_rs_vs_vni"] for c in candidates])
    r_trigger = pct_rank([c["_trigger_score"] for c in candidates])

    for i, c in enumerate(candidates):
        c["score"] = round(
            0.20 * r_momentum[i]
            + 0.10 * r_vol_spike[i]
            + 0.15 * r_buildup[i]
            + 0.15 * r_tightness[i]
            + 0.10 * r_close_q[i]
            + 0.10 * r_ext[i]
            + 0.10 * r_rs[i]
            + 0.10 * r_trigger[i],
            3,
        )


def run_pa_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
    bypass_market_gate: bool = False,
) -> list[dict]:
    """Run Price Action scanner across all symbols. Returns top 10 by score."""
    if not bypass_market_gate and not pa_market_regime_ok(vnindex_df):
        return []

    candidates: list[dict] = []
    total = len(symbols)
    done = 0

    def _one(sym: str) -> dict | None:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or df.empty or len(df) < 60:
            return None
        sig = scan_pa(df, vnindex_df)
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
                    candidates.append(res)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    _pa_cross_sectional_score(candidates)

    # Sector cap: max 2 per sector
    sector_count: dict[str, int] = {}
    result: list[dict] = []
    for c in sorted(candidates, key=lambda x: -x.get("score", 0)):
        sec = c.get("sector", "")
        cnt = sector_count.get(sec, 0)
        if cnt < 2:
            result.append(c)
            sector_count[sec] = cnt + 1
        if len(result) >= 10:
            break

    return result


def _render_pa_results(rows: list[dict], use_cache: bool, key: str = "pa_table") -> None:
    if not rows:
        st.info("Khong co tin hieu Price Action.")
        return

    st.caption(
        "Price Action — Tier A/B quality | "
        "Breakout after buildup & Pullback to MA20 | Sorted by Score"
    )

    table_rows = []
    for r in rows:
        rs = r.get("rs_vs_vni")
        rs_str = f"{rs:.3f}" if rs is not None else "—"
        volman_flags = []
        if r.get("strong_barrier"):
            volman_flags.append(f"Barrier({r.get('barrier_touches', 0)})")
        if r.get("is_squeeze"):
            volman_flags.append("Squeeze")
        table_rows.append({
            "Ma": r["symbol"],
            "Type": r["signal"].replace("PA_", ""),
            "Tier": r.get("pa_tier", ""),
            "Score": f"{r.get('score', 0):.3f}",
            "Gia": r["close"],
            "RSI": r.get("rsi", ""),
            "Buildup": f"{r.get('buildup_score', '')}/11 ({r.get('buildup_days', '')}d)",
            "Trigger": f"{r.get('trigger_score', '')}/6",
            "Tightness": f"{r.get('tightness', 0):.4f}",
            "Volman": " ".join(volman_flags) if volman_flags else "—",
            "RS20": rs_str,
            "Vol": f"{r.get('volume_spike', 0):.1f}x",
            "SL": r.get("sl", ""),
            "TP1": r.get("tp", ""),
            "R:R": r.get("rr", ""),
            "Nganh": r.get("sector", ""),
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
            st.session_state["pa_sel"] = rows[sel_rows[0]]
            st.session_state.pop("sig_sel", None)

            st.session_state.pop("mr_sel", None)
            st.session_state.pop("sw_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


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
            "Tier":        r.get("mr_tier", ""),
            "Score":       f"{r['final_score']:.2f}",
            "Giá":         r["close"],
            "SL":          r.get("sl", ""),
            "TP":          r.get("tp", ""),
            "R:R":         r.get("rr", ""),
            "Pos%":        f"{r.get('position_in_range', '')}%",
            "Dist%":       f"{r.get('dist_support_pct', '')}%",
            "Range%":      f"{r.get('range_size_pct', '')}%",
            "RSI3":        r.get("rsi3", ""),
            "Reversal":    f"{reversal_icons.get(rev, '')} {rev}",
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
            st.session_state.pop("sig_sel", None)
            st.session_state.pop("pa_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)



# ============================================================
# CLIMAX SCANNER — Sell Climax + False Break Support + Reversal
# ============================================================

def compute_climax_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators for the Climax reversal scanner."""
    d = df.copy()
    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    open_ = d["Open"]
    volume = d["Volume"]

    d["cx_ma20"] = close.rolling(20).mean()
    d["cx_ma20_prev5"] = d["cx_ma20"].shift(5)
    d["cx_ma20_slope_down"] = d["cx_ma20"] < d["cx_ma20_prev5"]
    d["cx_vol_ma20"] = volume.rolling(20).mean()

    # ATR(14) for climax candle sizing
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["cx_atr14"] = tr.rolling(14).mean()

    # Candle anatomy
    candle_range = (high - low).replace(0, 1e-9)
    d["cx_range"] = candle_range
    body = (close - open_).abs()
    d["cx_body"] = body
    d["cx_body_ratio"] = body / candle_range
    d["cx_upper_wick"] = high - close.where(close >= open_, open_)
    d["cx_lower_wick"] = close.where(close < open_, open_) - low
    d["cx_upper_wick_ratio"] = d["cx_upper_wick"] / candle_range
    d["cx_lower_wick_ratio"] = d["cx_lower_wick"] / candle_range

    # Bearish wide-range candle
    d["cx_bearish_wide"] = (
        (close < open_)
        & (candle_range > 1.5 * d["cx_atr14"])
        & (d["cx_body_ratio"] >= 0.65)
    )

    # Support: lowest low of previous 15 bars
    d["cx_support"] = low.shift(1).rolling(15).min()

    # Swing high for decline measurement
    d["cx_swing_high_10"] = high.shift(1).rolling(10).max()
    d["cx_decline_pct"] = (d["cx_swing_high_10"] - close) / d["cx_swing_high_10"].replace(0, 1e-9)

    # Red candle count in last 7 bars
    d["cx_red_count_7"] = (close < open_).astype(int).rolling(7).sum()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss_s = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss_s.replace(0, 1e-9)
    d["cx_rsi"] = 100 - (100 / (1 + rs))

    # Value
    d["cx_value_avg5"] = (close * volume).rolling(5).mean()

    return d


def scan_climax(df: pd.DataFrame, vnindex_df=None) -> dict | None:
    """
    Sell Climax + False Break Support + Reversal Buy scanner.
    Returns signal dict or None. Status = PENDING (needs confirmation).
    """
    if df is None or len(df) < 60:
        return None

    d = compute_climax_indicators(df)
    row = d.iloc[-1]

    required = [
        "cx_ma20", "cx_ma20_slope_down", "cx_vol_ma20", "cx_atr14",
        "cx_support", "cx_decline_pct", "cx_rsi", "cx_range",
    ]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    volume = float(row["Volume"])
    ma20 = float(row["cx_ma20"])
    vol_ma20 = float(row["cx_vol_ma20"])
    atr14 = float(row["cx_atr14"])
    support = float(row["cx_support"])
    decline_pct = float(row["cx_decline_pct"])
    rsi = float(row["cx_rsi"])
    candle_range = float(row["cx_range"])
    body = float(row["cx_body"])
    body_ratio = float(row["cx_body_ratio"])
    upper_wick = float(row["cx_upper_wick"])
    lower_wick = float(row["cx_lower_wick"])
    red_count = int(row["cx_red_count_7"])

    # ── A. Downtrend context ──
    if close >= ma20:
        return None
    if not bool(row["cx_ma20_slope_down"]):
        return None
    if decline_pct < 0.06:  # need >= 6% decline
        return None
    if red_count < 4:  # need >= 60% red in 7 bars
        return None

    # ── B. Sell climax: check last 3 bars for strong bearish candles ──
    climax_count = 0
    climax_vol_ok = False
    for j in range(1, min(4, len(d))):
        prev = d.iloc[-j]
        if bool(prev.get("cx_bearish_wide", False)):
            climax_count += 1
            if float(prev["Volume"]) >= float(prev["cx_vol_ma20"]) * 1.3:
                climax_vol_ok = True
    if climax_count < 1:
        return None

    # ── C. False break of support ──
    buffer = 0.005  # 0.5%
    broke_support = low < support * (1 - buffer)
    close_recovered = close >= support
    if not (broke_support and close_recovered):
        return None

    # ── D. Reversal candle detection ──
    if close <= open_:  # must be bullish
        return None

    reversal_type = None

    # Check bullish marubozu
    if (body_ratio >= 0.8
            and upper_wick <= 0.1 * candle_range
            and lower_wick <= 0.15 * candle_range):
        reversal_type = "MARUBOZU"

    # Check bullish hammer / pin bar
    if reversal_type is None:
        if (lower_wick >= 2 * body
                and upper_wick <= 0.3 * max(body, 1e-9)
                and close >= low + 0.66 * candle_range):
            reversal_type = "HAMMER"

    # Check strong bullish engulfing (close near high, large range)
    if reversal_type is None:
        if (candle_range > 1.0 * atr14
                and body_ratio >= 0.65
                and close >= high - 0.15 * candle_range):
            reversal_type = "ENGULFING"

    if reversal_type is None:
        return None

    # ── E. Volume check ──
    vol_spike = volume / max(vol_ma20, 1e-9)
    # Reversal or climax must have elevated volume
    if vol_spike < 1.0 and not climax_vol_ok:
        return None

    # ── SL / TP / R:R ──
    stop_loss = round(low * 0.998, 2)  # just below reversal candle low
    entry = round(close * 1.001, 2)   # 0.1% slippage
    risk = entry - stop_loss
    if risk <= 0:
        return None

    tp_ma20 = round(ma20, 2)  # mean reversion target
    tp_rr2 = round(entry + 2 * risk, 2)  # R:R = 2 target
    target = max(tp_ma20, tp_rr2)  # take the higher target
    rr_ratio = round((target - entry) / max(risk, 1e-9), 2)

    if rr_ratio < 2.0:
        return None

    # ── Quality tier from backtest analysis (R:R >= 2 win rate) ──
    risk_pct = (entry - stop_loss) / entry * 100
    # Tier A: deep decline + tight risk + pin bar/hammer reversal
    # Tier B: oversold + climax vol confirmed + HAMMER/MARUBOZU only + risk < 5%
    if (decline_pct >= 0.08 and risk_pct < 2.0
            and reversal_type in ("HAMMER", "MARUBOZU")):
        cx_tier = "A"
    elif (rsi < 35 and climax_vol_ok and risk_pct < 5.0
            and reversal_type in ("HAMMER", "MARUBOZU")):
        cx_tier = "B"
    else:
        return None

    return {
        "signal": "CLIMAX_REVERSAL",
        "date": df.index[-1],
        "close": round(close, 2),
        "status": "PENDING",
        "cx_tier": cx_tier,
        "reversal_type": reversal_type,
        "rsi": round(rsi, 1),
        "decline_pct": round(decline_pct * 100, 1),
        "support": round(support, 2),
        "red_count_7": red_count,
        "climax_count": climax_count,
        "climax_vol_ok": climax_vol_ok,
        "vol_spike": round(vol_spike, 2),
        "sl": stop_loss,
        "tp": target,
        "tp_ma20": tp_ma20,
        "rr": rr_ratio,
        "ma20": round(ma20, 2),
        "atr14": round(atr14, 2),
        "volume": int(volume),
        "avg_vol20": int(vol_ma20),
    }


def run_climax_scan(
    symbols: dict[str, str],
    use_cache: bool = True,
    vnindex_df=None,
    progress_cb=None,
) -> list[dict]:
    """Scan all symbols for Climax Reversal signals."""
    candidates: list[dict] = []
    total = len(symbols)
    done = 0

    def _one(sym: str, sector: str) -> dict | None:
        try:
            df_price = load_price_data(sym, use_cache=use_cache)
            if df_price is None or len(df_price) < 60:
                return None
            sig = scan_climax(df_price, vnindex_df)
            if sig:
                sig["symbol"] = sym.replace(".VN", "")
                sig["sector"] = sector
            return sig
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, sym, sec): sym for sym, sec in symbols.items()}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    candidates.append(res)
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    # Sort by decline depth (deeper = more extreme = potentially better reversal)
    candidates.sort(key=lambda r: -r.get("decline_pct", 0))
    return candidates[:10]


def _render_climax_results(rows: list[dict], use_cache: bool, key: str = "cx_table") -> None:
    if not rows:
        st.info("Khong co tin hieu Climax Reversal.")
        return

    st.caption(
        "Sell Climax — False break support + Reversal candle | "
        "Status PENDING = cho xac nhan"
    )

    table_rows = []
    for r in rows:
        table_rows.append({
            "Ma": r["symbol"],
            "Tier": r.get("cx_tier", ""),
            "Status": r.get("status", "PENDING"),
            "Reversal": r.get("reversal_type", ""),
            "Gia": r["close"],
            "RSI": r.get("rsi", ""),
            "Decline": f"{r.get('decline_pct', 0):.1f}%",
            "Support": r.get("support", ""),
            "Climax": f"{r.get('climax_count', 0)} bar",
            "Vol": f"{r.get('vol_spike', 0):.1f}x",
            "SL": r.get("sl", ""),
            "TP": r.get("tp", ""),
            "R:R": r.get("rr", ""),
            "Nganh": r.get("sector", ""),
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
            st.session_state["cx_sel"] = rows[sel_rows[0]]
            st.session_state.pop("sig_sel", None)

            st.session_state.pop("pa_sel", None)
            st.session_state.pop("sw_sel", None)
            st.session_state.pop("mr_sel", None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# PIN BAR 4H — Intraday Pin Bar at Context (last 2 days)
# ============================================================

def _resample_1h_to_4h(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Resample 1H OHLCV to 4H bars, drop non-trading windows."""
    df_4h = raw.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "Close"])
    df_4h = df_4h[df_4h["Volume"] > 0]
    return df_4h if len(df_4h) >= 60 else None


def load_price_data_4h(symbol: str) -> pd.DataFrame | None:
    """Fetch 1H data and resample to 4H. yfinance first (no rate limit), vnstock3 fallback."""
    # yfinance first — no strict rate limit, good for parallel scan
    try:
        raw = yf.Ticker(symbol).history(period="60d", interval="1h")
        if raw is not None and not raw.empty:
            if raw.index.tz is not None:
                raw.index = raw.index.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
            result = _resample_1h_to_4h(raw)
            if result is not None:
                return result
    except Exception:
        pass

    # vnstock3 fallback
    if HAS_VNSTOCK:
        sym_clean = symbol.replace(".VN", "")
        for source in ("VCI", "TCBS"):
            try:
                stock = Vnstock().stock(symbol=sym_clean, source=source)
                end   = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                raw   = stock.quote.history(symbol=sym_clean, start=start,
                                            end=end, interval="1H")
                if raw is not None and not raw.empty and len(raw) >= 20:
                    raw = raw.rename(columns={
                        "close": "Close", "open": "Open",
                        "high": "High", "low": "Low", "volume": "Volume"})
                    date_col = next((c for c in ["time", "date"] if c in raw.columns), None)
                    if date_col:
                        raw.index = pd.to_datetime(raw[date_col])
                    result = _resample_1h_to_4h(raw)
                    if result is not None:
                        return result
            except Exception:
                continue
    return None


def _check_d1_trend(sym: str, use_cache: bool = True) -> bool | None:
    """Check if D1 trend is up: close > MA50 and MA50 rising."""
    try:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or len(df) < 55:
            return None
        ma50 = df["Close"].rolling(50).mean()
        ma50_prev5 = ma50.shift(5)
        c = df["Close"].iloc[-1]
        m50 = ma50.iloc[-1]
        m50p = ma50_prev5.iloc[-1]
        if pd.isna(m50) or pd.isna(m50p):
            return None
        return bool(c > m50 and m50 > m50p)
    except Exception:
        return None


def run_pinbar_4h_scan(
    symbols: dict[str, str],
    vnindex_df=None,
    progress_cb=None,
    lookback_bars: int = 4,
) -> list[dict]:
    """Scan 4H candles with MTF trend alignment from D1."""
    candidates: list[dict] = []
    total = len(symbols)
    done  = 0

    def _one(sym: str, sector: str) -> list[dict]:
        try:
            # Load D1 trend for MTF alignment
            d1_trend = _check_d1_trend(sym)
            df_4h = load_price_data_4h(sym)
            if df_4h is None or len(df_4h) < 60:
                return []
            df_4h = compute_indicators(df_4h)
            hits: list[dict] = []
            for offset in range(lookback_bars):
                sub = df_4h.iloc[:len(df_4h) - offset] if offset > 0 else df_4h
                if len(sub) < 60:
                    break
                sig = scan_pinbar(sub, vnindex_df, d1_trend_up=d1_trend)
                if sig:
                    sig["symbol"]    = sym.replace(".VN", "")
                    sig["sector"]    = sector
                    sig["timeframe"] = "4H"
                    hits.append(sig)
            return hits
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, sym, sec): sym for sym, sec in symbols.items()}
        for fut in as_completed(futures):
            try:
                candidates.extend(fut.result())
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    candidates.sort(key=lambda r: (
        0 if r.get("pin_tier") == "A" else 1,
        -r.get("pin_score", 0),
        -r.get("rr", 0),
    ))
    return candidates


def _render_pinbar4h_results(rows: list[dict], use_cache: bool,
                              key: str = "pb4h_table") -> None:
    if not rows:
        st.info("Khong co Pin Bar 4H nao.")
        return
    st.caption(
        "Pin Bar 4H — nen 4 gio tai support (MA20/MA50/MA200/Swing) | "
        "PENDING = cho next candle close > high"
    )
    table_rows = []
    for r in rows:
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        wr       = r.get("wick_ratio")
        date_str = str(r.get("date", ""))[:16]
        score    = r.get("pin_score", "")
        rsi      = r.get("rsi14")
        rsi_str  = f"{rsi:.0f}" if rsi is not None else ""
        # Quality tags
        qtags = []
        if r.get("mtf_trend"):   qtags.append("MTF")
        pb = r.get("pullback", 0)
        if pb >= 2:              qtags.append(f"PB{pb}")
        if r.get("vol_quiet"):   qtags.append("DryUp")
        if r.get("body_pos_upper"): qtags.append("TopBody")
        table_rows.append({
            "Ma":      r["symbol"],
            "Tier":    r.get("pin_tier", ""),
            "Score":   f"{score}/13" if score != "" else "",
            "Signal":  r.get("signal", ""),
            "Date":    date_str,
            "Close":   r.get("close", ""),
            "Context": r.get("context", ""),
            "Wick":    f"{wr:.0%}" if wr is not None else "",
            "RSI":     rsi_str,
            "SL":      r.get("sl", ""),
            "TP":      r.get("tp", ""),
            "R:R":     r.get("rr", ""),
            "Volume":  f"{vol_icon} {r.get('vol_tier', '')}",
            "Quality": " ".join(qtags) if qtags else "",
            "Nganh":   r.get("sector", ""),
        })
    df_display = pd.DataFrame(table_rows)
    try:
        selected = st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key=key,
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["pb4h_sel"] = rows[sel_rows[0]]
            for k in ("sig_sel", "pa_sel", "sw_sel", "mr_sel", "cx_sel"):
                st.session_state.pop(k, None)
    except TypeError:
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================
# PIN BAR v2 — Confirmed buy-only (D1 + 4H, per pinbar_scanner_v2.md)
# ============================================================

def _check_d1_trend_v2(sym: str, use_cache: bool = True) -> bool | None:
    """HTF filter for 4H v2: D1 close > MA50 OR MA50 rising (weaker than v1 AND)."""
    try:
        df = load_price_data(sym, use_cache=use_cache)
        if df is None or len(df) < 55:
            return None
        ma50 = df["Close"].rolling(50).mean()
        ma50_prev5 = ma50.shift(5)
        c = df["Close"].iloc[-1]
        m50 = ma50.iloc[-1]
        m50p = ma50_prev5.iloc[-1]
        if pd.isna(m50) or pd.isna(m50p):
            return None
        return bool(c > m50 or m50 > m50p)
    except Exception:
        return None


def scan_pinbar_v2(df: pd.DataFrame, vnindex_df=None,
                    d1_trend_up: bool | None = None) -> dict | None:
    """
    Buy-only bullish pin bar at support — v2 (signal-only, no confirmation gate).

    Delegates shape/context/volume/scoring/RR to scan_pinbar.  v2 differs from
    the v1 chain only in its orchestration: it's run in a dedicated scanner
    combining D1 and 4H, with a weaker 4H HTF filter (see run_pinbar_v2_scan).
    """
    return scan_pinbar(df, vnindex_df, d1_trend_up=d1_trend_up)


def run_pinbar_v2_scan(symbols: dict[str, str], vnindex_df=None,
                       progress_cb=None) -> list[dict]:
    """
    Buy-only confirmed pin bar scan across D1 and 4H.

    Priority rules (per pinbar_scanner_v2.md):
      - D1 hits are primary.
      - 4H hits require HTF trend filter (D1 close>MA50 OR MA50 rising).
      - If both timeframes hit for the same symbol, D1 = primary, 4H = alt
        (earlier/refined entry).
    """
    candidates: list[dict] = []
    total = len(symbols)
    done  = 0

    def _one(sym: str, sector: str) -> list[dict]:
        out: list[dict] = []
        d1_trend = _check_d1_trend_v2(sym)

        # ── D1 scan ──
        try:
            df_d1 = load_price_data(sym, use_cache=True)
            if df_d1 is not None and len(df_d1) >= 60:
                df_d1 = compute_indicators(df_d1)
                sig = scan_pinbar_v2(df_d1, vnindex_df, d1_trend_up=d1_trend)
                if sig:
                    sig["symbol"]    = sym.replace(".VN", "")
                    sig["sector"]    = sector
                    sig["timeframe"] = "D1"
                    sig["priority"]  = "primary"
                    out.append(sig)
        except Exception:
            pass

        # ── 4H scan (gated by HTF filter) ──
        if d1_trend is True:
            try:
                df_4h = load_price_data_4h(sym)
                if df_4h is not None and len(df_4h) >= 60:
                    df_4h = compute_indicators(df_4h)
                    sig = scan_pinbar_v2(df_4h, vnindex_df, d1_trend_up=d1_trend)
                    if sig:
                        sig["symbol"]    = sym.replace(".VN", "")
                        sig["sector"]    = sector
                        sig["timeframe"] = "4H"
                        has_d1 = any(r["timeframe"] == "D1" for r in out)
                        sig["priority"]  = "alt" if has_d1 else "primary"
                        out.append(sig)
            except Exception:
                pass
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_one, sym, sec): sym for sym, sec in symbols.items()}
        for fut in as_completed(futures):
            try:
                candidates.extend(fut.result())
            except Exception:
                pass
            done += 1
            if progress_cb:
                progress_cb(done, total)

    candidates.sort(key=lambda r: (
        0 if r.get("priority") == "primary" else 1,
        0 if r.get("timeframe")  == "D1"      else 1,
        0 if r.get("pin_tier")   == "A"       else 1,
        -r.get("pin_score", 0),
        -r.get("rr", 0),
    ))
    return candidates


def _render_pinbar_v2_results(rows: list[dict], use_cache: bool,
                               key: str = "pbv2_table") -> None:
    if not rows:
        st.info("Khong co Pin Bar v2 nao.")
        return
    st.caption(
        "Pin Bar v2 — buy-only pin bar tai support (D1 + 4H) | "
        "D1 = primary, 4H = refined entry | HTF filter: D1 close>MA50 OR MA50 rising"
    )
    table_rows = []
    for r in rows:
        vol_icon = {"TIER1": "🔥", "TIER2": "📈"}.get(r.get("vol_tier", ""), "")
        wr       = r.get("wick_ratio")
        date_str = str(r.get("date", ""))[:16]
        score    = r.get("pin_score", "")
        rsi      = r.get("rsi14")
        rsi_str  = f"{rsi:.0f}" if rsi is not None else ""
        table_rows.append({
            "Ma":       r["symbol"],
            "TF":       r.get("timeframe", ""),
            "Prio":     r.get("priority", ""),
            "Tier":     r.get("pin_tier", ""),
            "Score":    f"{score}/13" if score != "" else "",
            "Signal":   r.get("signal", ""),
            "Date":     date_str,
            "Close":    r.get("close", ""),
            "Context":  r.get("context", ""),
            "Wick":     f"{wr:.0%}" if wr is not None else "",
            "RSI":      rsi_str,
            "SL":       r.get("sl", ""),
            "TP":       r.get("tp", ""),
            "R:R":      r.get("rr", ""),
            "Volume":   f"{vol_icon} {r.get('vol_tier', '')}",
            "Nganh":    r.get("sector", ""),
        })
    df_display = pd.DataFrame(table_rows)
    try:
        selected = st.dataframe(
            df_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key=key,
        )
        sel_rows = (selected.get("selection", {}) or {}).get("rows", [])
        if sel_rows:
            st.session_state["pbv2_sel"] = rows[sel_rows[0]]
            for k in ("sig_sel", "pa_sel", "sw_sel", "mr_sel", "cx_sel", "pb4h_sel"):
                st.session_state.pop(k, None)
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
    st.caption("Breakout Momentum & Trend Filter & Swing Filter | Scan sau khi nến ngày đóng cửa")

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
        st.header("Scan Options")
        bypass_vni = st.checkbox("Bypass VNINDEX filter", value=False,
                                  help="Bo qua dieu kien thi truong VNINDEX — scan bat ke xu huong index")
        st.session_state["bypass_vni"] = bypass_vni

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
        sigs, mkt_down = run_scan(VN30_STOCKS,  use_cache=use_cache, vnindex_df=vnindex_df, progress_cb=_cb30)
        st.session_state["scan_results"]       = sigs
        st.session_state["scan_universe"]      = "VN30"
        st.session_state["market_downtrend"]   = mkt_down
        prog.empty()

    if do_vn100:
        prog = st.progress(0, text="Scanning VN100… 0 / 100")
        def _cb100(done, total):
            prog.progress(done / total, text=f"Scanning VN100… {done} / {total}")
        sigs, mkt_down = run_scan(VN100_STOCKS, use_cache=use_cache, vnindex_df=vnindex_df, progress_cb=_cb100)
        st.session_state["scan_results"]       = sigs
        st.session_state["scan_universe"]      = "VN100"
        st.session_state["market_downtrend"]   = mkt_down
        prog.empty()

    # ── Market downtrend banner
    if st.session_state.get("market_downtrend"):
        st.warning(
            "⚠️ Thị trường downtrend (VNIndex < MA50) — "
            "Breakout signals đã bị tắt. Chỉ hiển thị Trend Filter & Pin Bar."
        )

    # ── Results
    results  = st.session_state.get("scan_results", [])
    universe = st.session_state.get("scan_universe", "")

    if results:
        _PINBAR_TYPES = {"PINBAR_MA200", "PINBAR_MA50", "PINBAR_MA20", "PINBAR_SWING"}
        n_bo  = sum(1 for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY"))
        n_nr7 = sum(1 for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY"))
        n_gap = sum(1 for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY"))
        n_pb  = sum(1 for r in results if r["signal"] in _PINBAR_TYPES)
        n_tf  = sum(1 for r in results if r["signal"] in ("TF_MA20", "TF_MA50"))
        n_rs  = sum(1 for r in results if (r.get("rs4w") or 0) >= 1.05)
        st.subheader(
            f"Kết quả {universe} — {len(results)} tín hiệu "
            f"(🚀{n_bo} · 🔩{n_nr7} · ⚡{n_gap} · 📍{n_pb} · 🎯{n_tf} · 🟢RS{n_rs})"
        )

        tab_all, tab_bo, tab_nr7, tab_gap, tab_pb, tab_tf = st.tabs([
            f"Tất cả ({len(results)})",
            f"🚀 Breakout ({n_bo})",
            f"🔩 NR7 ({n_nr7})",
            f"⚡ Gap ({n_gap})",
            f"📍 Pin Bar ({n_pb})",
            f"🎯 Trend Filter ({n_tf})",
        ])
        with tab_all:
            _render_results(results, use_cache, key="tab_all")
        with tab_bo:
            _render_results([r for r in results if r["signal"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY")], use_cache, key="tab_bo")
        with tab_nr7:
            _render_results([r for r in results if r["signal"] in ("NR7_STRONG", "NR7_EARLY")], use_cache, key="tab_nr7")
        with tab_gap:
            _render_results([r for r in results if r["signal"] in ("GAP_STRONG", "GAP_EARLY")], use_cache, key="tab_gap")
        with tab_pb:
            _render_results([r for r in results if r["signal"] in _PINBAR_TYPES], use_cache, key="tab_pb")
        with tab_tf:
            _render_results([r for r in results if r["signal"] in ("TF_MA20", "TF_MA50")], use_cache, key="tab_tf")

    # ── Chart panel — rendered OUTSIDE tabs so rerun never hides it ──
    sig_sel = st.session_state.get("sig_sel")

    if sig_sel:
        st.divider()
        st.subheader(f"Chart — {sig_sel['symbol']}")
        show_chart(sig_sel["symbol"], sig=sig_sel, use_cache=use_cache)

    if not results and not do_vn30 and not do_vn100 and not sig_sel:
        st.info("Nhấn Scan VN30 hoặc Scan VN100 để bắt đầu.")

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

    # ══════════════════════════════════════════════════════════════
    # SWING FILTER SCAN
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🎯 Swing Filter Scan")
    st.caption("Constructive buildup near support + breakout confirmation + cross-sectional scoring")
    if st.session_state.get("bypass_vni"):
        st.warning("⚠️ VNINDEX filter bypassed — ket qua khong duoc bao ve boi market regime gate")

    col_sw30, col_sw100 = st.columns(2)
    with col_sw30:
        do_sw30 = st.button("Scan Swing VN30", use_container_width=True)
    with col_sw100:
        do_sw100 = st.button("Scan Swing VN100", use_container_width=True)

    bypass_vni = st.session_state.get("bypass_vni", False)

    if do_sw30:
        prog = st.progress(0, text="Scanning Swing VN30... 0 / 30")
        def _sw_cb30(done, total):
            prog.progress(done / total, text=f"Scanning Swing VN30... {done} / {total}")
        sw_sigs = run_swing_scan(VN30_STOCKS, use_cache=use_cache,
                                  vnindex_df=vnindex_df, progress_cb=_sw_cb30,
                                  bypass_market_gate=bypass_vni)
        st.session_state["sw_results"] = sw_sigs
        st.session_state["sw_universe"] = "VN30"
        prog.empty()

    if do_sw100:
        prog = st.progress(0, text="Scanning Swing VN100... 0 / 100")
        def _sw_cb100(done, total):
            prog.progress(done / total, text=f"Scanning Swing VN100... {done} / {total}")
        sw_sigs = run_swing_scan(VN100_STOCKS, use_cache=use_cache,
                                  vnindex_df=vnindex_df, progress_cb=_sw_cb100,
                                  bypass_market_gate=bypass_vni)
        st.session_state["sw_results"] = sw_sigs
        st.session_state["sw_universe"] = "VN100"
        prog.empty()

    sw_results = st.session_state.get("sw_results", [])
    sw_universe = st.session_state.get("sw_universe", "")

    if sw_results:
        st.subheader(f"Swing Filter {sw_universe} — {len(sw_results)} candidates")
        _render_swing_results(sw_results, use_cache, key="sw_table_main")
    elif not do_sw30 and not do_sw100:
        st.caption("Nhan Scan Swing de bat dau.")

    # Swing chart panel — outside tabs
    sw_sel = st.session_state.get("sw_sel")
    if sw_sel:
        st.divider()
        st.subheader(
            f"Chart — {sw_sel['symbol']} | "
            f"Swing Filter Score {sw_sel.get('score', 0):.3f} | "
            f"Buildup {sw_sel.get('buildup_score', '')}/7 | "
            f"Trigger {sw_sel.get('trigger_score', '')}/4"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", sw_sel.get("close", ""))
        c1.metric("MA20", sw_sel.get("ma20", ""))
        c2.metric("MA50", sw_sel.get("ma50", ""))
        c2.metric("RSI", sw_sel.get("rsi", ""))
        c3.metric("SL", sw_sel.get("sl", ""))
        c3.metric("TP1", sw_sel.get("tp", ""))
        c4.metric("TP2", sw_sel.get("tp2", ""))
        c4.metric("R:R", sw_sel.get("rr", ""))
        rs = sw_sel.get("rs_vs_vni_20")
        c5.metric("RS vs VNI", f"{rs:.3f}" if rs else "—")
        c5.metric("Tightness", f"{sw_sel.get('tightness', 0):.4f}")
        show_chart(sw_sel["symbol"], sig=sw_sel, use_cache=use_cache)

    # ══════════════════════════════════════════════════════════════
    # PRICE ACTION SCAN
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📐 Price Action — Breakout & Pullback")
    st.caption("Volman-style buildup+barrier detection | Breakout after buildup & Pullback to MA20")
    if st.session_state.get("bypass_vni"):
        st.warning("⚠️ VNINDEX filter bypassed — ket qua khong duoc bao ve boi market regime gate")

    col_pa30, col_pa100 = st.columns(2)
    with col_pa30:
        do_pa30 = st.button("Scan PA VN30", use_container_width=True)
    with col_pa100:
        do_pa100 = st.button("Scan PA VN100", use_container_width=True)

    if do_pa30:
        prog = st.progress(0, text="Scanning PA VN30... 0 / 30")
        def _pa_cb30(done, total):
            prog.progress(done / total, text=f"Scanning PA VN30... {done} / {total}")
        pa_sigs = run_pa_scan(VN30_STOCKS, use_cache=use_cache,
                               vnindex_df=vnindex_df, progress_cb=_pa_cb30,
                               bypass_market_gate=bypass_vni)
        st.session_state["pa_results"] = pa_sigs
        st.session_state["pa_universe"] = "VN30"
        prog.empty()

    if do_pa100:
        prog = st.progress(0, text="Scanning PA VN100... 0 / 100")
        def _pa_cb100(done, total):
            prog.progress(done / total, text=f"Scanning PA VN100... {done} / {total}")
        pa_sigs = run_pa_scan(VN100_STOCKS, use_cache=use_cache,
                               vnindex_df=vnindex_df, progress_cb=_pa_cb100,
                               bypass_market_gate=bypass_vni)
        st.session_state["pa_results"] = pa_sigs
        st.session_state["pa_universe"] = "VN100"
        prog.empty()

    pa_results = st.session_state.get("pa_results", [])
    pa_universe = st.session_state.get("pa_universe", "")

    if pa_results:
        n_bo = sum(1 for r in pa_results if r["signal"] == "PA_BREAKOUT")
        n_pb = sum(1 for r in pa_results if r["signal"] == "PA_PULLBACK")
        st.subheader(
            f"PA {pa_universe} — {len(pa_results)} candidates "
            f"(Breakout {n_bo} / Pullback {n_pb})"
        )
        _render_pa_results(pa_results, use_cache, key="pa_table_main")
    elif not do_pa30 and not do_pa100:
        st.caption("Nhan Scan PA de bat dau.")

    # PA chart panel
    pa_sel = st.session_state.get("pa_sel")
    if pa_sel:
        st.divider()
        volman = []
        if pa_sel.get("strong_barrier"):
            volman.append(f"Barrier({pa_sel.get('barrier_touches', 0)})")
        if pa_sel.get("is_squeeze"):
            volman.append("Squeeze")
        st.subheader(
            f"Chart — {pa_sel['symbol']} | "
            f"{pa_sel['signal'].replace('PA_', '')} | "
            f"Score {pa_sel.get('score', 0):.3f} | "
            f"Buildup {pa_sel.get('buildup_score', '')}/11"
            + (f" | {' '.join(volman)}" if volman else "")
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", pa_sel.get("close", ""))
        c1.metric("MA20", pa_sel.get("ma20", ""))
        c2.metric("MA50", pa_sel.get("ma50", ""))
        c2.metric("RSI", pa_sel.get("rsi", ""))
        c3.metric("SL", pa_sel.get("sl", ""))
        c3.metric("TP1", pa_sel.get("tp", ""))
        c4.metric("TP2", pa_sel.get("tp2", ""))
        c4.metric("R:R", pa_sel.get("rr", ""))
        rs = pa_sel.get("rs_vs_vni")
        c5.metric("RS vs VNI", f"{rs:.3f}" if rs else "—")
        c5.metric("Trigger", f"{pa_sel.get('trigger_score', '')}/6")
        show_chart(pa_sel["symbol"], sig=pa_sel, use_cache=use_cache)

    # ── Section E: Climax Reversal Scan ──────────────────────────
    st.divider()
    st.subheader("🔻 Climax Reversal — Sell Climax + False Break + Reversal")
    st.caption(
        "Tim co phieu bi ban thao manh, thung support gia, "
        "xuat hien nen dao chieu (PENDING = cho xac nhan)"
    )

    col_cx30, col_cx100 = st.columns(2)
    with col_cx30:
        do_cx30 = st.button("Scan Climax VN30", use_container_width=True)
    with col_cx100:
        do_cx100 = st.button("Scan Climax VN100", use_container_width=True)

    if do_cx30:
        prog = st.progress(0, text="Scanning Climax VN30... 0 / 30")
        def _cx_cb30(done, total):
            prog.progress(done / total, text=f"Scanning Climax VN30... {done} / {total}")
        cx_sigs = run_climax_scan(VN30_STOCKS, use_cache=use_cache,
                                   vnindex_df=vnindex_df, progress_cb=_cx_cb30)
        st.session_state["cx_results"] = cx_sigs
        st.session_state["cx_universe"] = "VN30"
        prog.empty()

    if do_cx100:
        prog = st.progress(0, text="Scanning Climax VN100... 0 / 100")
        def _cx_cb100(done, total):
            prog.progress(done / total, text=f"Scanning Climax VN100... {done} / {total}")
        cx_sigs = run_climax_scan(VN100_STOCKS, use_cache=use_cache,
                                   vnindex_df=vnindex_df, progress_cb=_cx_cb100)
        st.session_state["cx_results"] = cx_sigs
        st.session_state["cx_universe"] = "VN100"
        prog.empty()

    cx_results = st.session_state.get("cx_results", [])
    cx_universe = st.session_state.get("cx_universe", "")

    if cx_results:
        st.subheader(
            f"Climax {cx_universe} — {len(cx_results)} candidates"
        )
        _render_climax_results(cx_results, use_cache, key="cx_table_main")
    elif not do_cx30 and not do_cx100:
        st.caption("Nhan Scan Climax de bat dau.")

    # Climax chart panel
    cx_sel = st.session_state.get("cx_sel")
    if cx_sel:
        st.divider()
        st.subheader(
            f"Chart — {cx_sel['symbol']} | "
            f"{cx_sel.get('reversal_type', '')} | "
            f"Tier {cx_sel.get('cx_tier', '')} | "
            f"Decline {cx_sel.get('decline_pct', 0):.1f}%"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", cx_sel.get("close", ""))
        c1.metric("Support", cx_sel.get("support", ""))
        c2.metric("RSI", cx_sel.get("rsi", ""))
        c2.metric("Decline", f"{cx_sel.get('decline_pct', 0):.1f}%")
        c3.metric("SL", cx_sel.get("sl", ""))
        c3.metric("TP", cx_sel.get("tp", ""))
        c4.metric("R:R", cx_sel.get("rr", ""))
        c4.metric("Vol", f"{cx_sel.get('vol_spike', 0):.1f}x")
        c5.metric("Status", cx_sel.get("status", "PENDING"))
        c5.metric("Climax bars", cx_sel.get("climax_count", ""))
        show_chart(cx_sel["symbol"], sig=cx_sel, use_cache=use_cache)

    # ══════════════════════════════════════════════════════════════
    # PIN BAR 4H — Intraday (last 2 days)
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📍 Pin Bar 4H — Nen 4 gio gan day (2 ngay)")
    st.caption(
        "Scan pin bar tren khung 4H — wick dai tai MA20/MA50/MA200/Swing Low | "
        "Lookback ~4 nen 4H (2 ngay giao dich)"
    )

    col_pb4h_30, col_pb4h_100 = st.columns(2)
    with col_pb4h_30:
        do_pb4h_30 = st.button("Scan PB4H VN30", use_container_width=True)
    with col_pb4h_100:
        do_pb4h_100 = st.button("Scan PB4H VN100", use_container_width=True)

    if do_pb4h_30:
        prog = st.progress(0, text="Scanning Pin Bar 4H VN30... 0 / 30")
        def _pb4h_cb30(done, total):
            prog.progress(done / total, text=f"Scanning Pin Bar 4H VN30... {done} / {total}")
        pb4h_sigs = run_pinbar_4h_scan(VN30_STOCKS, vnindex_df=vnindex_df,
                                        progress_cb=_pb4h_cb30, lookback_bars=4)
        st.session_state["pb4h_results"] = pb4h_sigs
        st.session_state["pb4h_universe"] = "VN30"
        prog.empty()

    if do_pb4h_100:
        prog = st.progress(0, text="Scanning Pin Bar 4H VN100... 0 / 100")
        def _pb4h_cb100(done, total):
            prog.progress(done / total, text=f"Scanning Pin Bar 4H VN100... {done} / {total}")
        pb4h_sigs = run_pinbar_4h_scan(VN100_STOCKS, vnindex_df=vnindex_df,
                                        progress_cb=_pb4h_cb100, lookback_bars=4)
        st.session_state["pb4h_results"] = pb4h_sigs
        st.session_state["pb4h_universe"] = "VN100"
        prog.empty()

    pb4h_results  = st.session_state.get("pb4h_results", [])
    pb4h_universe = st.session_state.get("pb4h_universe", "")

    if pb4h_results:
        st.subheader(f"Pin Bar 4H {pb4h_universe} — {len(pb4h_results)} tin hieu")
        _render_pinbar4h_results(pb4h_results, use_cache, key="pb4h_table_main")
    elif not do_pb4h_30 and not do_pb4h_100:
        st.caption("Nhan Scan PB4H de bat dau.")

    # PB4H chart panel
    pb4h_sel = st.session_state.get("pb4h_sel")
    if pb4h_sel:
        st.divider()
        score = pb4h_sel.get("pin_score", "")
        st.subheader(
            f"Chart — {pb4h_sel['symbol']} | "
            f"{pb4h_sel.get('signal', '')} | "
            f"Tier {pb4h_sel.get('pin_tier', '')} ({score}/13) | "
            f"Context {pb4h_sel.get('context', '')}"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", pb4h_sel.get("close", ""))
        c1.metric("MA20", pb4h_sel.get("ma20", ""))
        c2.metric("MA50", pb4h_sel.get("ma50", ""))
        c2.metric("R:R", pb4h_sel.get("rr", ""))
        c3.metric("SL", pb4h_sel.get("sl", ""))
        c3.metric("TP", pb4h_sel.get("tp", ""))
        c4.metric("Wick", f"{pb4h_sel.get('wick_ratio', 0):.0%}")
        rsi = pb4h_sel.get("rsi14")
        c4.metric("RSI", f"{rsi:.0f}" if rsi else "—")
        c5.metric("Vol", pb4h_sel.get("vol_tier", ""))
        c5.metric("Score", pb4h_sel.get("score_detail", ""))
        show_chart(pb4h_sel["symbol"], sig=pb4h_sel, use_cache=use_cache)

    # ══════════════════════════════════════════════════════════════
    # PIN BAR v2 — Buy-only pin bar signal (D1 + 4H)
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🎯 Pin Bar v2 — Buy-only signal (D1 + 4H)")
    st.caption(
        "Buy-only pin bar tai support — signal khi nen hinh thanh (khong yeu cau confirm) | "
        "D1 = primary, 4H = refined/earlier entry | HTF filter 4H: D1 close>MA50 OR MA50 rising"
    )

    col_pbv2_30, col_pbv2_100 = st.columns(2)
    with col_pbv2_30:
        do_pbv2_30 = st.button("Scan PBv2 VN30", use_container_width=True)
    with col_pbv2_100:
        do_pbv2_100 = st.button("Scan PBv2 VN100", use_container_width=True)

    if do_pbv2_30:
        prog = st.progress(0, text="Scanning Pin Bar v2 VN30... 0 / 30")
        def _pbv2_cb30(done, total):
            prog.progress(done / total, text=f"Scanning Pin Bar v2 VN30... {done} / {total}")
        pbv2_sigs = run_pinbar_v2_scan(VN30_STOCKS, vnindex_df=vnindex_df,
                                        progress_cb=_pbv2_cb30)
        st.session_state["pbv2_results"]  = pbv2_sigs
        st.session_state["pbv2_universe"] = "VN30"
        prog.empty()

    if do_pbv2_100:
        prog = st.progress(0, text="Scanning Pin Bar v2 VN100... 0 / 100")
        def _pbv2_cb100(done, total):
            prog.progress(done / total, text=f"Scanning Pin Bar v2 VN100... {done} / {total}")
        pbv2_sigs = run_pinbar_v2_scan(VN100_STOCKS, vnindex_df=vnindex_df,
                                        progress_cb=_pbv2_cb100)
        st.session_state["pbv2_results"]  = pbv2_sigs
        st.session_state["pbv2_universe"] = "VN100"
        prog.empty()

    pbv2_results  = st.session_state.get("pbv2_results", [])
    pbv2_universe = st.session_state.get("pbv2_universe", "")

    if pbv2_results:
        st.subheader(f"Pin Bar v2 {pbv2_universe} — {len(pbv2_results)} tin hieu")
        _render_pinbar_v2_results(pbv2_results, use_cache, key="pbv2_table_main")
    elif not do_pbv2_30 and not do_pbv2_100:
        st.caption("Nhan Scan PBv2 de bat dau.")

    # PBv2 chart panel
    pbv2_sel = st.session_state.get("pbv2_sel")
    if pbv2_sel:
        st.divider()
        score = pbv2_sel.get("pin_score", "")
        st.subheader(
            f"Chart — {pbv2_sel['symbol']} [{pbv2_sel.get('timeframe','')}] | "
            f"{pbv2_sel.get('signal', '')} | "
            f"Tier {pbv2_sel.get('pin_tier', '')} ({score}/13) | "
            f"Context {pbv2_sel.get('context', '')}"
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close",    pbv2_sel.get("close", ""))
        c1.metric("MA20",     pbv2_sel.get("ma20", ""))
        c2.metric("MA50",     pbv2_sel.get("ma50", ""))
        c2.metric("R:R",      pbv2_sel.get("rr", ""))
        c3.metric("SL",       pbv2_sel.get("sl", ""))
        c3.metric("TP",       pbv2_sel.get("tp", ""))
        c4.metric("Wick",     f"{pbv2_sel.get('wick_ratio', 0):.0%}")
        rsi = pbv2_sel.get("rsi14")
        c4.metric("RSI",      f"{rsi:.0f}" if rsi else "—")
        c5.metric("Vol",      pbv2_sel.get("vol_tier", ""))
        c5.metric("Priority", pbv2_sel.get("priority", ""))
        show_chart(pbv2_sel["symbol"], sig=pbv2_sel, use_cache=use_cache)


if __name__ == "__main__":
    main()
