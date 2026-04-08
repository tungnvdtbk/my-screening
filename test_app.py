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

if __name__ == "__main__":
    unittest.main(verbosity=2)
