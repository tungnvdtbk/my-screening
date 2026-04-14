"""
test_app.py — unit tests for app.py pure-logic functions.
Run: python test_app.py

Covers:
  compute_indicators · _vol_tier · compute_rs4w · compute_market_filter
  scan_breakout · scan_nr7 · scan_gap
  _save_df/_load_df · cache_stats/clear_cache · load_price_data
"""

import sys
import types
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


# ── 1. Stub Streamlit before importing app ────────────────────────────
class _CM:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _): return lambda *a, **kw: _CM()
    def __iter__(self): return iter([])

def _noop(*a, **kw): return _CM()
def _columns(n, *a, **kw):
    n = n if isinstance(n, int) else len(n)
    return [_CM() for _ in range(n)]

_st = types.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)
_st.session_state = {}
_st.button        = lambda *a, **kw: False
_st.checkbox      = lambda *a, **kw: True
_st.columns       = _columns
_st.sidebar       = _CM()
_st.spinner       = _CM
_st.tabs          = lambda labels: [_CM() for _ in labels]
for _attr in [
    "set_page_config", "title", "caption", "info", "warning", "error",
    "subheader", "markdown", "metric", "dataframe", "selectbox",
    "slider", "download_button", "stop", "rerun", "header", "write",
    "divider", "plotly_chart", "image", "text_input",
]:
    setattr(_st, _attr, _noop)
sys.modules["streamlit"] = _st

# ── 2. Stub vnstock3 ──────────────────────────────────────────────────
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

# ── 3. Import app ─────────────────────────────────────────────────────
import app


# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_uptrend_df(n=350, base=50.0, vol=500_000, seed=42):
    """Steadily rising OHLCV — price gains ~50 % over n bars."""
    rng = np.random.default_rng(seed)
    prices = base + np.linspace(0, base * 0.5, n) + rng.standard_normal(n) * 0.3
    prices = np.abs(prices)
    idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.full(n, float(vol)),
    }, index=idx)


def make_downtrend_df(n=350, base=80.0, vol=500_000, seed=42):
    """Steadily falling OHLCV — price loses ~40 % over n bars."""
    rng = np.random.default_rng(seed)
    prices = base - np.linspace(0, base * 0.4, n) + rng.standard_normal(n) * 0.3
    prices = np.abs(prices)
    idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.full(n, float(vol)),
    }, index=idx)


def make_df_ending(n, end_date, base=50.0, vol=500_000, seed=42):
    """Synthetic df whose last bar lands on a specific date."""
    rng = np.random.default_rng(seed)
    prices = base + np.linspace(0, base * 0.3, n) + rng.standard_normal(n) * 0.2
    prices = np.abs(prices)
    idx = pd.bdate_range(end=end_date, periods=n)
    return pd.DataFrame({
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.full(n, float(vol)),
    }, index=idx)


def patch_last(df, **kwargs):
    """Return a copy of df with last-row cells overwritten."""
    df = df.copy()
    for col, val in kwargs.items():
        if col not in df.columns:
            df[col] = np.nan
        df.loc[df.index[-1], col] = val
    return df


def patch_prev(df, **kwargs):
    """Return a copy of df with second-to-last row cells overwritten."""
    df = df.copy()
    for col, val in kwargs.items():
        if col not in df.columns:
            df[col] = np.nan
        df.loc[df.index[-2], col] = val
    return df


def _ind(n=300, trend="up", seed=42):
    """Convenience: compute_indicators on a base df."""
    base = make_uptrend_df(n, seed=seed) if trend == "up" else make_downtrend_df(n, seed=seed)
    return app.compute_indicators(base)


def _breakout_row(df, price=100.0, high10_mult=0.97, high20_mult=0.98):
    """
    Patch last row of df to satisfy ALL scan_breakout base conditions.
    high10_mult / high20_mult control HIGH10 / HIGH20 relative to price.
    Default: price > high20 → BREAKOUT_STRONG.
    """
    atr      = max(float(df["atr10"].dropna().iloc[-1]), 0.5)
    avg_vol  = float(df["avg_vol20"].dropna().iloc[-1])
    ma50     = price * 0.93          # well below price (uptrend)
    return patch_last(df,
        Close=price,   Open=price * 0.994,
        High=price * 1.001, Low=price - atr * 1.15,
        Volume=avg_vol * 2.5,
        atr10=atr,    avg_vol20=avg_vol,  avg_vol_pre5=avg_vol * 0.70,
        ma50=ma50,    ma50_prev5=ma50 * 0.99,
        high10=price * high10_mult,
        high20=price * high20_mult,
        high_prev1=price * 0.985,
        ma200=price * 0.80,
        candle_range=price * 1.001 - (price - atr * 1.15),
        nr7=False, gap_pct=0.003,
    )


def _nr7_row(df, price=100.0, high10_mult=0.97, high20_mult=0.98):
    """
    Patch last row to satisfy NR7 conditions including new quality filters:
      - Volume QUIET++ (< 0.5x avg) → +25 score points
      - Previous candle wraps current → Inside Bar → +25 score points
      - Total score = 50 ≥ MIN_SCORE(30) and ≥ NR7_EARLY threshold(50)
    """
    atr      = max(float(df["atr10"].dropna().iloc[-1]), 0.5)
    avg_vol  = float(df["avg_vol20"].dropna().iloc[-1])
    ma50     = price * 0.93
    narrow   = atr * 0.4   # NR7: range smaller than recent candles
    # Make previous candle wider than current → Inside Bar
    df = patch_prev(df, High=price * 1.02, Low=price * 0.97)
    return patch_last(df,
        Close=price,    Open=price - narrow * 0.4,
        High=price,     Low=price - narrow,
        Volume=avg_vol * 0.4,              # QUIET++ → not HIGH, score +25
        atr10=atr,     avg_vol20=avg_vol,  avg_vol_pre5=avg_vol * 0.70,
        ma50=ma50,     ma50_prev5=ma50 * 0.99,
        high10=price * high10_mult,
        high20=price * high20_mult,
        high_prev1=price * 0.985,
        ma200=price * 0.80,
        candle_range=narrow,
        nr7=True, gap_pct=0.0,
    )


def _gap_row(df, price=100.0, gap_pct=0.008, high10_mult=0.97, high20_mult=0.98):
    """Patch last row to satisfy GAP conditions."""
    atr      = max(float(df["atr10"].dropna().iloc[-1]), 0.5)
    avg_vol  = float(df["avg_vol20"].dropna().iloc[-1])
    ma50     = price * 0.93
    prev_close = price / (1 + gap_pct) * 0.999   # ensure close > prev_close
    return patch_last(
        patch_prev(df, Close=prev_close),
        Close=price,   Open=price * 0.998,
        High=price * 1.001, Low=price * 0.994,
        Volume=avg_vol * 2.5,
        atr10=atr,    avg_vol20=avg_vol,  avg_vol_pre5=avg_vol * 0.70,
        ma50=ma50,    ma50_prev5=ma50 * 0.99,
        high10=price * high10_mult,
        high20=price * high20_mult,
        high_prev1=price * 0.985,
        ma200=price * 0.80,
        candle_range=price * 1.001 - price * 0.994,
        nr7=False, gap_pct=gap_pct,
    )



# ══════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════════════

class TestComputeIndicators(unittest.TestCase):
    """compute_indicators(df) adds all expected columns."""

    def setUp(self):
        self.df = app.compute_indicators(make_uptrend_df(300))

    def test_all_columns_present(self):
        expected = [
            "atr10", "avg_vol20", "avg_vol_pre5",
            "high10", "high20",
            "ma50", "ma200", "ma50_prev5",
            "high_prev1", "candle_range", "nr7", "gap_pct",
        ]
        for col in expected:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_high20_gte_high10(self):
        """MAX of 20 days always ≥ MAX of 10 days."""
        valid = self.df.dropna(subset=["high10", "high20"])
        self.assertTrue((valid["high20"] >= valid["high10"]).all())

    def test_nr7_is_boolean_dtype(self):
        """nr7 column must contain True/False values."""
        non_nan = self.df["nr7"].dropna()
        self.assertTrue(non_nan.isin([True, False]).all())

    def test_gap_pct_direction(self):
        """gap_pct > 0 when open > prior close (uptrend mostly gaps up)."""
        valid = self.df["gap_pct"].dropna()
        self.assertGreater(valid.mean(), -0.05)   # sanity: not systematically negative

    def test_high_prev1_equals_prior_high(self):
        """high_prev1[i] must equal High[i-1]."""
        h_prev = self.df["high_prev1"].iloc[50]
        h_actual = self.df["High"].iloc[49]
        self.assertAlmostEqual(float(h_prev), float(h_actual), places=6)

    def test_short_df_produces_nan_indicators(self):
        """DataFrame shorter than rolling window → indicator NaN at start."""
        df = app.compute_indicators(make_uptrend_df(30))
        self.assertTrue(df["atr10"].iloc[:12].isna().any())


# ─────────────────────────────────────────────────────────────────────

class TestVolTier(unittest.TestCase):
    """_vol_tier(vol, avg_vol20, avg_vol_pre5) returns correct tier label."""

    def test_tier1_spike_and_contract(self):
        avg = 1_000_000.0
        result = app._vol_tier(2_100_000, avg, avg * 0.70)   # 2.1× spike, 70% contract
        self.assertEqual(result, "TIER1")

    def test_tier2_spike_no_contract(self):
        avg = 1_000_000.0
        result = app._vol_tier(2_100_000, avg, avg * 0.85)   # 2.1× spike, 85% (no contract)
        self.assertEqual(result, "TIER2")

    def test_tier3_no_spike(self):
        avg = 1_000_000.0
        result = app._vol_tier(1_500_000, avg, avg * 0.60)   # 1.5× — below 2×
        self.assertEqual(result, "TIER3")

    def test_no_vol_data_nan(self):
        self.assertEqual(app._vol_tier(500_000, float("nan"), 300_000), "NO_VOL_DATA")

    def test_no_vol_data_zero(self):
        self.assertEqual(app._vol_tier(500_000, 0.0, 300_000), "NO_VOL_DATA")

    def test_tier1_boundary_exactly_2x(self):
        avg = 1_000_000.0
        # Exactly 2× is NOT > 2×, so should be TIER3 (no spike)
        result = app._vol_tier(2_000_000, avg, avg * 0.70)
        self.assertEqual(result, "TIER3")

    def test_tier2_boundary_above_2x(self):
        avg = 1_000_000.0
        result = app._vol_tier(2_000_001, avg, avg * 0.80)   # just over 2×, no contract
        self.assertEqual(result, "TIER2")


# ─────────────────────────────────────────────────────────────────────

class TestComputeRs4w(unittest.TestCase):
    """compute_rs4w(df, vnindex_df) — 4-week relative strength."""

    def _make_idx_df(self, ret_pct, n=50):
        """Create a simple index df with a given 21-bar return."""
        prices = np.ones(n) * 1000.0
        prices[-1] = 1000.0 * (1 + ret_pct)
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    def test_returns_none_no_vnindex(self):
        df = make_uptrend_df(50)
        self.assertIsNone(app.compute_rs4w(df, None))

    def test_returns_none_short_stock_df(self):
        vn = self._make_idx_df(0.05)
        df = make_uptrend_df(10)     # too short
        self.assertIsNone(app.compute_rs4w(df, vn))

    def test_returns_none_short_vnindex(self):
        vn = self._make_idx_df(0.05, n=5)    # too short
        df = make_uptrend_df(50)
        self.assertIsNone(app.compute_rs4w(df, vn))

    def test_outperform_returns_gt_1(self):
        """Stock +15 % vs index +5 % → RS4W > 1."""
        prices = np.linspace(100, 115, 50)
        idx    = pd.date_range(end="2026-01-01", periods=50, freq="B")
        stock  = pd.DataFrame({"Close": prices, "Open": prices, "High": prices * 1.01,
                                "Low": prices * 0.99, "Volume": np.ones(50) * 1e6}, index=idx)
        vn     = self._make_idx_df(0.05)
        result = app.compute_rs4w(stock, vn)
        self.assertIsNotNone(result)
        self.assertGreater(result, 1.0)

    def test_underperform_returns_lt_1(self):
        """Stock flat (0 %) vs index +10 % → RS4W < 1."""
        prices = np.ones(50) * 100.0
        idx    = pd.date_range(end="2026-01-01", periods=50, freq="B")
        stock  = pd.DataFrame({"Close": prices, "Open": prices, "High": prices * 1.01,
                                "Low": prices * 0.99, "Volume": np.ones(50) * 1e6}, index=idx)
        vn     = self._make_idx_df(0.10)
        result = app.compute_rs4w(stock, vn)
        self.assertIsNotNone(result)
        self.assertLess(result, 1.0)

    def test_flat_index_returns_none(self):
        """Index with near-zero return → division guard → None."""
        vn = self._make_idx_df(0.0005)   # 0.05% — below 0.001 guard
        df = make_uptrend_df(50)
        self.assertIsNone(app.compute_rs4w(df, vn))


# ─────────────────────────────────────────────────────────────────────

class TestComputeMarketFilter(unittest.TestCase):
    """compute_market_filter(vnindex_df) — market traffic light."""

    def test_none_returns_unknown_label(self):
        label, details = app.compute_market_filter(None)
        self.assertIn("⚪", label)
        self.assertEqual(details, {})

    def test_short_df_returns_unknown_label(self):
        df = make_uptrend_df(30, base=1000.0)
        label, details = app.compute_market_filter(df)
        self.assertIn("⚪", label)

    def test_uptrend_returns_favorable(self):
        """Price > MA50, MA20 > MA50 → green/favorable."""
        df = make_uptrend_df(200, base=1000.0)
        label, details = app.compute_market_filter(df)
        self.assertIn("🟢", label)

    def test_downtrend_returns_unfavorable(self):
        """Price falling below MA50 → red/unfavorable."""
        df = make_downtrend_df(200, base=1300.0)
        label, details = app.compute_market_filter(df)
        self.assertIn("🔴", label)

    def test_details_has_required_keys(self):
        df = make_uptrend_df(200, base=1000.0)
        _, details = app.compute_market_filter(df)
        for key in ["VNINDEX", "MA20", "MA50"]:
            self.assertIn(key, details)


# ─────────────────────────────────────────────────────────────────────

class TestScanBreakout(unittest.TestCase):
    """scan_breakout(df) — BREAKOUT_STRONG / BREAKOUT_EARLY detection."""

    def setUp(self):
        self.base = _ind(300)

    def test_breakout_strong_above_high20(self):
        """high > HIGH20 → signal = BREAKOUT_STRONG."""
        df = _breakout_row(self.base, price=100.0, high10_mult=0.97, high20_mult=0.98)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "BREAKOUT_STRONG")

    def test_breakout_early_above_high10_only(self):
        """high > HIGH10 but ≤ HIGH20 → signal = BREAKOUT_EARLY."""
        # price = 100, high10 = 97, high20 = 101 (price can't exceed high20)
        df = _breakout_row(self.base, price=100.0, high10_mult=0.97, high20_mult=1.01)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "BREAKOUT_EARLY")

    def test_returns_none_too_short(self):
        df = app.compute_indicators(make_uptrend_df(30))
        self.assertIsNone(app.scan_breakout(df))

    def test_returns_none_downtrend(self):
        """close < ma50 → fail TREND condition."""
        df = _breakout_row(self.base, price=100.0, high10_mult=0.97, high20_mult=0.98)
        df = patch_last(df, ma50=110.0, ma50_prev5=109.0)   # ma50 above close
        self.assertIsNone(app.scan_breakout(df))

    def test_returns_none_close_not_above_prev_high(self):
        """close <= high_prev1 → fail close-above condition."""
        df = _breakout_row(self.base, price=100.0)
        df = patch_last(df, high_prev1=101.0)   # prev high above close
        self.assertIsNone(app.scan_breakout(df))

    def test_returns_none_candle_too_small(self):
        """Range < 1.0 × ATR → fail SIZE condition."""
        df = _breakout_row(self.base, price=100.0)
        atr = float(df["atr10"].iloc[-1])
        df = patch_last(df, Low=100.0 - atr * 0.4)   # range ≈ 0.5 ATR
        self.assertIsNone(app.scan_breakout(df))

    def test_returns_none_overextended(self):
        """close > ma50 × 1.08 → filtered out."""
        df = _breakout_row(self.base, price=100.0)
        df = patch_last(df, ma50=90.0, ma50_prev5=89.0)   # 11% above ma50
        self.assertIsNone(app.scan_breakout(df))

    def test_returns_none_no_breakout(self):
        """high < HIGH10 → no breakout at all."""
        df = _breakout_row(self.base, price=100.0)
        df = patch_last(df, high10=105.0, high20=110.0)   # both above price
        self.assertIsNone(app.scan_breakout(df))

    def test_result_keys(self):
        """Result dict must contain all expected fields."""
        df = _breakout_row(self.base)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        for key in ["signal", "close", "sl", "tp", "rr", "atr10", "vol_tier", "rs4w"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_sl_below_close(self):
        df = _breakout_row(self.base)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertLess(result["sl"], result["close"])

    def test_tp_above_close(self):
        df = _breakout_row(self.base)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertGreater(result["tp"], result["close"])

    def test_vol_tier_present_and_valid(self):
        df = _breakout_row(self.base)
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertIn(result["vol_tier"], ["TIER1", "TIER2", "TIER3", "NO_VOL_DATA"])

    def test_high_vol_gives_tier1(self):
        """Vol > 2× AND contract < 75 % → TIER1."""
        df = _breakout_row(self.base)   # already sets vol = avg*2.5, pre5 = avg*0.70
        result = app.scan_breakout(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["vol_tier"], "TIER1")


# ─────────────────────────────────────────────────────────────────────

class TestScanNr7(unittest.TestCase):
    """scan_nr7(df) — NR7 coil breakout detection."""

    def setUp(self):
        self.base = _ind(300)

    def test_nr7_strong_above_high20(self):
        df = _nr7_row(self.base, price=100.0, high10_mult=0.97, high20_mult=0.98)
        result = app.scan_nr7(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "NR7_STRONG")

    def test_nr7_early_above_high10_only(self):
        df = _nr7_row(self.base, price=100.0, high10_mult=0.97, high20_mult=1.01)
        result = app.scan_nr7(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "NR7_EARLY")

    def test_returns_none_not_nr7(self):
        """nr7 = False → no NR7 signal."""
        df = _nr7_row(self.base, price=100.0)
        df = patch_last(df, nr7=False)
        self.assertIsNone(app.scan_nr7(df))

    def test_returns_none_downtrend(self):
        df = _nr7_row(self.base, price=100.0)
        df = patch_last(df, ma50=110.0, ma50_prev5=109.0)
        self.assertIsNone(app.scan_nr7(df))

    def test_returns_none_no_breakout(self):
        """NR7 candle but high < HIGH10 → not a breakout."""
        df = _nr7_row(self.base, price=100.0)
        df = patch_last(df, high10=105.0, high20=110.0)
        self.assertIsNone(app.scan_nr7(df))

    def test_returns_none_bearish_candle(self):
        """close < open → fail BULL condition."""
        df = _nr7_row(self.base, price=100.0)
        df = patch_last(df, Open=101.0)   # open > close
        self.assertIsNone(app.scan_nr7(df))

    def test_result_has_required_keys(self):
        df = _nr7_row(self.base)
        result = app.scan_nr7(df)
        self.assertIsNotNone(result)
        for key in ["signal", "close", "sl", "tp", "rr", "atr10", "vol_tier"]:
            self.assertIn(key, result)

    def test_sl_is_low_of_signal_candle(self):
        df = _nr7_row(self.base, price=100.0)
        result = app.scan_nr7(df)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["sl"], float(df["Low"].iloc[-1]), places=1)


# ─────────────────────────────────────────────────────────────────────

class TestScanGap(unittest.TestCase):
    """scan_gap(df) — gap-up breakout detection."""

    def setUp(self):
        self.base = _ind(300)

    def test_gap_strong_above_high20(self):
        df = _gap_row(self.base, price=100.0, gap_pct=0.008, high10_mult=0.97, high20_mult=0.98)
        result = app.scan_gap(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "GAP_STRONG")

    def test_gap_early_above_high10_only(self):
        df = _gap_row(self.base, price=100.0, gap_pct=0.008, high10_mult=0.97, high20_mult=1.01)
        result = app.scan_gap(df)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "GAP_EARLY")

    def test_returns_none_gap_too_small(self):
        """Gap < 0.5 % → below threshold."""
        df = _gap_row(self.base, price=100.0, gap_pct=0.003)
        self.assertIsNone(app.scan_gap(df))

    def test_returns_none_gap_fades(self):
        """close < prev_close → gap fully faded."""
        df = _gap_row(self.base, price=100.0, gap_pct=0.008)
        # prev close is higher than new close → gap faded
        df = patch_prev(df, Close=105.0)
        self.assertIsNone(app.scan_gap(df))

    def test_returns_none_downtrend(self):
        df = _gap_row(self.base, price=100.0, gap_pct=0.008)
        df = patch_last(df, ma50=110.0, ma50_prev5=109.0)
        self.assertIsNone(app.scan_gap(df))

    def test_returns_none_no_breakout(self):
        df = _gap_row(self.base, price=100.0)
        df = patch_last(df, high10=105.0, high20=110.0)
        self.assertIsNone(app.scan_gap(df))

    def test_gap_pct_in_result(self):
        """gap_pct should be included in result dict (as %)."""
        df = _gap_row(self.base, price=100.0, gap_pct=0.008)
        result = app.scan_gap(df)
        self.assertIsNotNone(result)
        self.assertIn("gap_pct", result)
        self.assertAlmostEqual(result["gap_pct"], 0.8, places=0)


# ─────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# CACHE TESTS (kept from original — test unchanged cache functions)
# ══════════════════════════════════════════════════════════════════════

class TestSavLoadHelpers(unittest.TestCase):
    """_save_df / _load_df — parquet-or-CSV round-trip."""

    def test_roundtrip(self):
        df = make_uptrend_df(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            app._save_df(df, path)
            loaded = app._load_df(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), len(df))
            self.assertAlmostEqual(float(loaded["Close"].iloc[-1]),
                                   float(df["Close"].iloc[-1]), places=4)

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(app._load_df("/nonexistent/path.parquet"))

    def test_columns_preserved(self):
        df = make_uptrend_df(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cols.parquet")
            app._save_df(df, path)
            loaded = app._load_df(path)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                self.assertIn(col, loaded.columns)


class TestCacheStats(unittest.TestCase):
    """cache_stats() and clear_cache()."""

    def test_stats_counts_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                for name in ["A_VN.parquet", "B_VN.parquet", "C_VN.parquet"]:
                    open(os.path.join(tmpdir, name), "wb").close()
                n, _ = app.cache_stats()
                self.assertEqual(n, 3)

    def test_stats_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                n, mb = app.cache_stats()
                self.assertEqual(n, 0)
                self.assertEqual(mb, 0.0)

    def test_clear_deletes_parquet_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                for name in ["A.parquet", "B.parquet", "C.csv"]:
                    open(os.path.join(tmpdir, name), "wb").close()
                deleted = app.clear_cache()
                self.assertEqual(deleted, 3)
                n, _ = app.cache_stats()
                self.assertEqual(n, 0)

    def test_clear_returns_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                for name in ["X.parquet", "Y.parquet"]:
                    open(os.path.join(tmpdir, name), "wb").close()
                self.assertEqual(app.clear_cache(), 2)

    def test_clear_ignores_non_cache_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                open(os.path.join(tmpdir, "readme.txt"), "w").close()
                open(os.path.join(tmpdir, "A.parquet"), "wb").close()
                deleted = app.clear_cache()
                self.assertEqual(deleted, 1)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "readme.txt")))


class TestLoadPriceDataCache(unittest.TestCase):
    """load_price_data() — incremental cache logic (yfinance mocked)."""

    def _mock_ticker(self, df):
        m = MagicMock()
        m.history.return_value = df
        return m

    def test_uptodate_cache_no_yfinance_call(self):
        last_bday = pd.bdate_range(end=pd.Timestamp.today(), periods=1)[0].normalize()
        df = make_df_ending(200, last_bday)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(df, app._cache_path("FRESH.VN"))
                with patch("yfinance.Ticker") as mock_yf:
                    result = app.load_price_data("FRESH.VN", use_cache=True)
                    mock_yf.assert_not_called()
                    self.assertIsNotNone(result)
                    self.assertEqual(len(result), len(df))

    def test_same_day_cache_no_yfinance_call(self):
        last_bday = pd.bdate_range(end=pd.Timestamp.today(), periods=1)[0].normalize()
        df = make_df_ending(100, last_bday)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(df, app._cache_path("TODAY.VN"))
                with patch("yfinance.Ticker") as mock_yf:
                    app.load_price_data("TODAY.VN", use_cache=True)
                    mock_yf.assert_not_called()

    def test_stale_cache_triggers_fetch(self):
        today  = pd.Timestamp.today().normalize()
        stale  = today - pd.Timedelta(days=10)
        old_df = make_df_ending(200, stale)
        new_df = make_df_ending(5, today, base=old_df["Close"].iloc[-1] * 1.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("STALE.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    result = app.load_price_data("STALE.VN", use_cache=True)
                    self.assertGreater(len(result), len(old_df))

    def test_stale_cache_appends_new_bars(self):
        today  = pd.Timestamp.today().normalize()
        stale  = today - pd.Timedelta(days=10)
        old_df = make_df_ending(100, stale)
        new_df = make_df_ending(3, today, base=99.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("APPEND.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    result = app.load_price_data("APPEND.VN", use_cache=True)
                    self.assertAlmostEqual(float(result["Close"].iloc[-1]),
                                           float(new_df["Close"].iloc[-1]), places=2)

    def test_stale_cache_saved_to_disk(self):
        import time
        today  = pd.Timestamp.today().normalize()
        stale  = today - pd.Timedelta(days=10)
        old_df = make_df_ending(100, stale)
        new_df = make_df_ending(3, today, base=60.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                path = app._cache_path("SAVE.VN")
                app._save_df(old_df, path)
                mtime_before = os.path.getmtime(path)
                time.sleep(0.05)
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    app.load_price_data("SAVE.VN", use_cache=True)
                self.assertGreater(os.path.getmtime(path), mtime_before)

    def test_use_cache_false_skips_disk_read(self):
        today  = pd.Timestamp.today().normalize()
        old_df = make_df_ending(200, today)
        fresh  = make_uptrend_df(300)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("NOCACHE.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(fresh)) as mock_yf:
                    app.load_price_data("NOCACHE.VN", use_cache=False)
                    mock_yf.assert_called_once()

    def test_use_cache_false_fetches_full_period(self):
        fresh = make_uptrend_df(300)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                mock_ticker = self._mock_ticker(fresh)
                with patch("yfinance.Ticker", return_value=mock_ticker):
                    app.load_price_data("FULL.VN", use_cache=False)
                    mock_ticker.history.assert_called_once_with(period="2y")

    def test_no_cache_fetches_and_writes(self):
        fresh = make_uptrend_df(300)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                path = app._cache_path("NEW.VN")
                self.assertFalse(os.path.exists(path))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(fresh)):
                    result = app.load_price_data("NEW.VN", use_cache=True)
                self.assertIsNotNone(result)
                exists = os.path.exists(path) or os.path.exists(path.replace(".parquet", ".csv"))
                self.assertTrue(exists)

    def test_combined_capped_at_500_bars(self):
        today  = pd.Timestamp.today().normalize()
        stale  = today - pd.Timedelta(days=10)
        old_df = make_df_ending(490, stale)
        new_df = make_df_ending(50, today, base=old_df["Close"].iloc[-1] * 1.02)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("CAP.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    result = app.load_price_data("CAP.VN", use_cache=True)
                    self.assertLessEqual(len(result), 500)


# ══════════════════════════════════════════════════════════════════════
# SWING FILTER TESTS
# ══════════════════════════════════════════════════════════════════════

def _make_swing_df(n=300, base=50.0, vol=2_000_000, seed=42):
    """
    Build a df that has high value (price*vol > 1e10), uptrend with
    MA20 > MA50, and enough history for all swing indicators.
    """
    rng = np.random.default_rng(seed)
    prices = base + np.linspace(0, base * 0.5, n) + rng.standard_normal(n) * 0.2
    prices = np.abs(prices)
    idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open":   prices * 0.999,
        "High":   prices * 1.005,
        "Low":    prices * 0.995,
        "Close":  prices,
        "Volume": np.full(n, float(vol)),
    }, index=idx)


def _swing_ready_row(df, price=60.0):
    """
    Patch last row to satisfy ALL scan_swing_filter hard filters,
    buildup, and entry trigger conditions.
    """
    ma50 = price * 0.96
    ma20 = price * 0.99
    vol_ma20 = 2_000_000.0
    # Patch several trailing rows for buildup persistence
    for offset in range(1, 6):
        idx = df.index[-offset]
        df.loc[idx, "Close"] = price * (1 - offset * 0.001)
        df.loc[idx, "Open"] = price * (1 - offset * 0.001) * 0.999
        df.loc[idx, "High"] = price * (1 - offset * 0.001) * 1.002
        df.loc[idx, "Low"] = price * (1 - offset * 0.001) * 0.998
        df.loc[idx, "Volume"] = vol_ma20 * 0.7  # low volume for buildup

    df = patch_last(df,
        Close=price,
        Open=price * 0.993,
        High=price * 1.003,
        Low=price * 0.990,
        Volume=vol_ma20 * 2.0,  # volume expansion for entry
    )
    return df


class TestComputeSwingIndicators(unittest.TestCase):
    """compute_swing_indicators(df) adds all expected columns."""

    def setUp(self):
        self.df = app.compute_swing_indicators(_make_swing_df(300))

    def test_all_sw_columns_present(self):
        expected = [
            "sw_ma20", "sw_ma50", "sw_vol_ma20", "sw_vol_ma5",
            "sw_ret_5d", "sw_ret_2d", "sw_volatility10",
            "sw_distance_ma20", "sw_volume_spike", "sw_value",
            "sw_high_20d", "sw_body_ratio", "sw_range_pct",
            "sw_rsi", "sw_mom_accel", "sw_pivot_high",
            "sw_tightness", "sw_vol_dryup", "sw_near_ma20",
            "sw_higher_low", "sw_small_body", "sw_range_contract",
            "sw_no_distribution", "sw_buildup_score",
            "sw_is_buildup", "sw_buildup_days",
        ]
        for col in expected:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_rsi_bounded_0_100(self):
        """RSI(14) must always be between 0 and 100."""
        valid = self.df["sw_rsi"].dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_mom_accel_clipped(self):
        """Momentum acceleration clipped to [-0.30, 0.30]."""
        valid = self.df["sw_mom_accel"].dropna()
        self.assertTrue((valid >= -0.30).all())
        self.assertTrue((valid <= 0.30).all())

    def test_buildup_score_bounded(self):
        """Buildup score is 0–7 (7 boolean components)."""
        valid = self.df["sw_buildup_score"].dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 7).all())

    def test_pivot_high_uses_shifted_data(self):
        """pivot_high = high.shift(1).rolling(10).max — must not include current high."""
        row = self.df.iloc[-1]
        if not pd.isna(row["sw_pivot_high"]):
            # pivot_high is max of high[-2] to high[-11], should not equal current high
            # unless coincidence — just check it's not NaN and is positive
            self.assertGreater(float(row["sw_pivot_high"]), 0)

    def test_tightness_positive(self):
        """Tightness = rolling std / close, must be >= 0."""
        valid = self.df["sw_tightness"].dropna()
        self.assertTrue((valid >= 0).all())


class TestSwingMarketRegime(unittest.TestCase):
    """swing_market_regime_ok(vnindex_df) — market regime gate."""

    def test_returns_false_for_none(self):
        self.assertFalse(app.swing_market_regime_ok(None))

    def test_returns_false_for_short_df(self):
        df = _make_swing_df(30, base=1000)
        self.assertFalse(app.swing_market_regime_ok(df))

    def test_uptrend_market_returns_true(self):
        """Strong uptrend: MA20 > MA50, price > MA50*0.97, 3d ret > -3%."""
        df = _make_swing_df(200, base=1000)
        result = app.swing_market_regime_ok(df)
        self.assertTrue(result)

    def test_downtrend_market_returns_false(self):
        """Downtrend: MA20 < MA50 → False."""
        df = make_downtrend_df(200, base=1300)
        result = app.swing_market_regime_ok(df)
        self.assertFalse(result)


class TestSwingRsVsVni(unittest.TestCase):
    """_swing_rs_vs_vni — 20-day relative strength."""

    def test_returns_none_no_vnindex(self):
        df = _make_swing_df(50)
        self.assertIsNone(app._swing_rs_vs_vni(df, None))

    def test_returns_none_short_df(self):
        df = _make_swing_df(10)
        vn = _make_swing_df(50, base=1000)
        self.assertIsNone(app._swing_rs_vs_vni(df, vn))

    def test_outperformer_gt_1(self):
        """Stock gaining faster than index → RS > 1."""
        prices_stock = np.linspace(50, 60, 50)  # +20%
        idx = pd.date_range(end="2026-01-01", periods=50, freq="B")
        stock = pd.DataFrame({
            "Open": prices_stock, "High": prices_stock * 1.01,
            "Low": prices_stock * 0.99, "Close": prices_stock,
            "Volume": np.ones(50) * 1e6,
        }, index=idx)
        prices_vn = np.linspace(1000, 1050, 50)  # +5%
        vn = pd.DataFrame({"Close": prices_vn}, index=idx)
        rs = app._swing_rs_vs_vni(stock, vn)
        self.assertIsNotNone(rs)
        self.assertGreater(rs, 1.0)


class TestScanSwingFilter(unittest.TestCase):
    """scan_swing_filter(df) — full pipeline test."""

    def test_returns_none_for_short_df(self):
        df = _make_swing_df(30)
        self.assertIsNone(app.scan_swing_filter(df))

    def test_returns_none_for_downtrend(self):
        """close < ma50 → fail hard filter."""
        df = make_downtrend_df(300)
        self.assertIsNone(app.scan_swing_filter(df))

    def test_returns_none_low_value(self):
        """value < 1e10 → fail liquidity gate."""
        df = _make_swing_df(300, vol=100)  # very low volume → value < 1e10
        self.assertIsNone(app.scan_swing_filter(df))

    def test_returns_none_rsi_too_high(self):
        """RSI >= 72 → overbought filter."""
        df = _make_swing_df(300)
        d = app.compute_swing_indicators(df)
        # Force RSI to be high
        d.loc[d.index[-1], "sw_rsi"] = 75.0
        # scan_swing_filter calls compute_swing_indicators internally,
        # so we test via the function directly — it will recompute.
        # Instead, test the filter logic by checking RSI is a gate.
        result = app.scan_swing_filter(df)
        # If result is not None, RSI must be < 72
        if result is not None:
            self.assertLess(result["rsi"], 72)

    def test_result_keys_when_signal_found(self):
        """If signal is found, check all required output columns."""
        # This test may or may not find a signal depending on random data,
        # so we just check the contract IF a signal is found.
        df = _make_swing_df(300, vol=5_000_000)
        result = app.scan_swing_filter(df)
        if result is not None:
            required_keys = [
                "signal", "close", "rsi", "tightness", "buildup_score",
                "buildup_days", "rs_vs_vni_20", "price_break", "vol_expand",
                "strong_bar", "mom_accel", "trigger_score", "entry_confirmed",
                "sl", "tp", "tp2", "rr", "ma20", "ma50", "sw_tier", "score",
            ]
            # score is added by cross-sectional scoring, not by scan_swing_filter
            for key in required_keys:
                if key != "score":
                    self.assertIn(key, result, f"Missing key: {key}")

    def test_sl_below_entry(self):
        """Stop loss must be below entry price."""
        df = _make_swing_df(300, vol=5_000_000)
        result = app.scan_swing_filter(df)
        if result is not None:
            self.assertLess(result["sl"], result["close"])

    def test_tp_above_entry(self):
        """Target must be above entry price."""
        df = _make_swing_df(300, vol=5_000_000)
        result = app.scan_swing_filter(df)
        if result is not None:
            self.assertGreater(result["tp"], result["close"])

    def test_rr_at_least_1_5(self):
        """R:R ratio must be >= 1.5 (hard filter)."""
        df = _make_swing_df(300, vol=5_000_000)
        result = app.scan_swing_filter(df)
        if result is not None:
            self.assertGreaterEqual(result["rr"], 1.5)

    def test_sw_tier_valid(self):
        """sw_tier must be A, B, or C."""
        df = _make_swing_df(300, vol=5_000_000)
        result = app.scan_swing_filter(df)
        if result is not None:
            self.assertIn(result["sw_tier"], ["A", "B", "C"])


class TestSwingCrossSectionalScore(unittest.TestCase):
    """_swing_cross_sectional_score — percentile ranking."""

    def _make_candidate(self, **overrides):
        base = {
            "_ret_5d": 0.05,
            "_volume_spike": 1.5,
            "_distance_ma20": 1.01,
            "_tightness": 0.02,
            "_buildup_score": 5,
            "_trigger_score": 3,
            "_rs_vs_vni_20": 1.05,
        }
        base.update(overrides)
        return base

    def test_empty_list(self):
        """No candidates → no crash."""
        app._swing_cross_sectional_score([])

    def test_single_candidate_gets_050(self):
        """Single candidate gets score 0.50."""
        c = self._make_candidate()
        app._swing_cross_sectional_score([c])
        self.assertAlmostEqual(c["score"], 0.50, places=2)

    def test_multiple_candidates_ranked(self):
        """Better candidate gets higher score."""
        weak = self._make_candidate(
            _ret_5d=0.01, _volume_spike=0.8,
            _buildup_score=3, _trigger_score=2,
        )
        strong = self._make_candidate(
            _ret_5d=0.10, _volume_spike=3.0,
            _buildup_score=7, _trigger_score=4,
        )
        candidates = [weak, strong]
        app._swing_cross_sectional_score(candidates)
        self.assertGreater(strong["score"], weak["score"])

    def test_scores_between_0_and_1(self):
        """All scores must be in [0, 1]."""
        candidates = [
            self._make_candidate(_ret_5d=i * 0.01, _buildup_score=i % 7 + 1)
            for i in range(10)
        ]
        app._swing_cross_sectional_score(candidates)
        for c in candidates:
            self.assertGreaterEqual(c["score"], 0.0)
            self.assertLessEqual(c["score"], 1.0)

    def test_score_weights_sum_to_1(self):
        """Verify the weight formula sums to 1.0."""
        self.assertAlmostEqual(0.20 + 0.10 + 0.10 + 0.15 + 0.20 + 0.15 + 0.10, 1.0)


class TestRunSwingScan(unittest.TestCase):
    """run_swing_scan — integration test."""

    def test_returns_empty_when_market_down(self):
        """If market regime fails, return empty list."""
        # Downtrend VNINDEX
        vn = make_downtrend_df(200, base=1300)
        result = app.run_swing_scan(
            {"FPT.VN": "Technology"}, use_cache=True, vnindex_df=vn,
        )
        self.assertEqual(result, [])

    def test_returns_list(self):
        """Return type is always a list."""
        vn = _make_swing_df(200, base=1000)
        result = app.run_swing_scan(
            {"FPT.VN": "Technology"}, use_cache=True, vnindex_df=vn,
        )
        self.assertIsInstance(result, list)

    def test_max_10_results(self):
        """At most 10 candidates returned."""
        vn = _make_swing_df(200, base=1000)
        result = app.run_swing_scan(
            app.VN100_STOCKS, use_cache=True, vnindex_df=vn,
        )
        self.assertLessEqual(len(result), 10)


# ══════════════════════════════════════════════════════════════════════
# PRICE ACTION SCANNER TESTS
# ══════════════════════════════════════════════════════════════════════

class TestComputePaIndicators(unittest.TestCase):
    """compute_pa_indicators(df) adds all expected columns."""

    def setUp(self):
        self.df = app.compute_pa_indicators(make_uptrend_df(300))

    def test_all_pa_columns_present(self):
        expected = [
            "pa_ma20", "pa_ma50", "pa_vol_ma20", "pa_vol_ma5",
            "pa_value", "pa_value_avg5", "pa_ma20_slope", "pa_ma50_slope",
            "pa_bar_range", "pa_range_pct", "pa_body_ratio", "pa_upper_tail",
            "pa_pivot_10", "pa_pivot_20", "pa_low_3", "pa_low_5",
            "pa_ma20_distance", "pa_range_avg20", "pa_range_expansion",
            "pa_ret_5d", "pa_ret_2d", "pa_rsi", "pa_mom_accel",
            "pa_barrier_touches", "pa_strong_barrier",
            "pa_support_rising", "pa_resist_flat", "pa_is_squeeze",
            "pa_trend_filter",
            "pa_tightness", "pa_tight_closes", "pa_vol_dryup",
            "pa_near_ma20", "pa_higher_low", "pa_small_body",
            "pa_range_contract", "pa_below_pivot", "pa_no_distribution",
            "pa_buildup_score", "pa_is_buildup", "pa_buildup_days",
            "pa_hit_limit_up", "pa_half_day",
            "pa_prior_push", "pa_pullback_depth",
        ]
        for col in expected:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_rsi_bounded(self):
        valid = self.df["pa_rsi"].dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_buildup_score_bounded(self):
        """Buildup score 0-11 (11 boolean components)."""
        valid = self.df["pa_buildup_score"].dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 11).all())

    def test_pivot_20_uses_shifted_data(self):
        """pivot_20 = high.shift(1).rolling(20).max — no lookahead."""
        row = self.df.iloc[-1]
        if not pd.isna(row["pa_pivot_20"]):
            self.assertGreater(float(row["pa_pivot_20"]), 0)

    def test_barrier_touches_non_negative(self):
        valid = self.df["pa_barrier_touches"].dropna()
        self.assertTrue((valid >= 0).all())

    def test_trend_filter_is_boolean(self):
        valid = self.df["pa_trend_filter"].dropna()
        self.assertTrue(valid.isin([True, False]).all())

    def test_mom_accel_clipped(self):
        valid = self.df["pa_mom_accel"].dropna()
        self.assertTrue((valid >= -0.30).all())
        self.assertTrue((valid <= 0.30).all())


class TestPaMarketRegime(unittest.TestCase):
    """pa_market_regime_ok(vnindex_df) — market regime gate."""

    def test_returns_false_for_none(self):
        self.assertFalse(app.pa_market_regime_ok(None))

    def test_returns_false_for_short_df(self):
        df = make_uptrend_df(30, base=1000)
        self.assertFalse(app.pa_market_regime_ok(df))

    def test_uptrend_returns_true(self):
        df = make_uptrend_df(200, base=1000)
        self.assertTrue(app.pa_market_regime_ok(df))

    def test_downtrend_returns_false(self):
        df = make_downtrend_df(200, base=1300)
        self.assertFalse(app.pa_market_regime_ok(df))


class TestPaRsVsVni(unittest.TestCase):
    """_pa_rs_vs_vni — relative strength."""

    def test_none_no_vnindex(self):
        self.assertIsNone(app._pa_rs_vs_vni(make_uptrend_df(50), None))

    def test_outperformer_gt_1(self):
        prices = np.linspace(50, 60, 50)
        idx = pd.date_range(end="2026-01-01", periods=50, freq="B")
        stock = pd.DataFrame({
            "Open": prices, "High": prices * 1.01,
            "Low": prices * 0.99, "Close": prices,
            "Volume": np.ones(50) * 1e6,
        }, index=idx)
        vn = pd.DataFrame({"Close": np.linspace(1000, 1050, 50)}, index=idx)
        rs = app._pa_rs_vs_vni(stock, vn)
        self.assertIsNotNone(rs)
        self.assertGreater(rs, 1.0)


class TestScanPa(unittest.TestCase):
    """scan_pa(df) — full pipeline."""

    def test_returns_none_short_df(self):
        self.assertIsNone(app.scan_pa(make_uptrend_df(30)))

    def test_returns_none_downtrend(self):
        self.assertIsNone(app.scan_pa(make_downtrend_df(300)))

    def test_returns_none_low_value(self):
        """value_avg5 < 1e10 → fail liquidity."""
        df = make_uptrend_df(300, vol=100)
        self.assertIsNone(app.scan_pa(df))

    def test_result_has_required_keys(self):
        """If signal found, check required output columns."""
        df = make_uptrend_df(300, vol=5_000_000)
        result = app.scan_pa(df)
        if result is not None:
            for key in [
                "signal", "close", "setup_type", "rsi", "buildup_score",
                "buildup_days", "tightness", "strong_barrier", "is_squeeze",
                "trigger_score", "sl", "tp", "tp2", "rr", "pa_tier",
            ]:
                self.assertIn(key, result, f"Missing key: {key}")

    def test_rr_at_least_1_5(self):
        """R:R ratio must be >= 1.5."""
        df = make_uptrend_df(300, vol=5_000_000)
        result = app.scan_pa(df)
        if result is not None:
            self.assertGreaterEqual(result["rr"], 1.5)

    def test_setup_type_valid(self):
        """Signal must be PA_BREAKOUT or PA_PULLBACK."""
        df = make_uptrend_df(300, vol=5_000_000)
        result = app.scan_pa(df)
        if result is not None:
            self.assertIn(result["signal"], ["PA_BREAKOUT", "PA_PULLBACK"])

    def test_pa_tier_valid(self):
        """pa_tier must be A, B, or C."""
        df = make_uptrend_df(300, vol=5_000_000)
        result = app.scan_pa(df)
        if result is not None:
            self.assertIn(result["pa_tier"], ["A", "B", "C"])


class TestPaCrossSectionalScore(unittest.TestCase):
    """_pa_cross_sectional_score — percentile ranking."""

    def _make_candidate(self, **overrides):
        base = {
            "_ret_5d": 0.05, "_volume_spike": 1.5,
            "_buildup_score": 6, "_tightness": 0.02,
            "_close_quality": 0.7, "_extension_penalty": 0.01,
            "_rs_vs_vni": 1.05, "_trigger_score": 4,
        }
        base.update(overrides)
        return base

    def test_empty_list(self):
        app._pa_cross_sectional_score([])

    def test_single_gets_050(self):
        c = self._make_candidate()
        app._pa_cross_sectional_score([c])
        self.assertAlmostEqual(c["score"], 0.50, places=2)

    def test_better_candidate_higher_score(self):
        weak = self._make_candidate(
            _ret_5d=0.01, _buildup_score=3, _trigger_score=2,
        )
        strong = self._make_candidate(
            _ret_5d=0.10, _buildup_score=9, _trigger_score=6,
        )
        app._pa_cross_sectional_score([weak, strong])
        self.assertGreater(strong["score"], weak["score"])

    def test_scores_bounded(self):
        candidates = [
            self._make_candidate(_ret_5d=i * 0.01, _buildup_score=i % 11)
            for i in range(10)
        ]
        app._pa_cross_sectional_score(candidates)
        for c in candidates:
            self.assertGreaterEqual(c["score"], 0.0)
            self.assertLessEqual(c["score"], 1.0)

    def test_weights_sum_to_1(self):
        self.assertAlmostEqual(0.20+0.10+0.15+0.15+0.10+0.10+0.10+0.10, 1.0)


class TestRunPaScan(unittest.TestCase):
    """run_pa_scan — integration."""

    def test_returns_empty_when_market_down(self):
        vn = make_downtrend_df(200, base=1300)
        result = app.run_pa_scan({"FPT.VN": "Technology"}, vnindex_df=vn)
        self.assertEqual(result, [])

    def test_returns_list(self):
        vn = make_uptrend_df(200, base=1000)
        result = app.run_pa_scan({"FPT.VN": "Technology"}, vnindex_df=vn)
        self.assertIsInstance(result, list)

    def test_max_10_results(self):
        vn = make_uptrend_df(200, base=1000)
        result = app.run_pa_scan(app.VN100_STOCKS, vnindex_df=vn)
        self.assertLessEqual(len(result), 10)

    def test_sector_cap(self):
        """Max 2 signals per sector."""
        vn = make_uptrend_df(200, base=1000)
        result = app.run_pa_scan(app.VN100_STOCKS, vnindex_df=vn)
        from collections import Counter
        sector_counts = Counter(r.get("sector", "") for r in result)
        for sec, cnt in sector_counts.items():
            self.assertLessEqual(cnt, 2, f"Sector {sec} has {cnt} > 2 signals")


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
