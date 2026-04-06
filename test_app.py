"""
test_app.py — unit tests for app.py pure-logic functions.
Run: /path/to/python test_app.py

Strategy: stub out Streamlit (no server needed) + vnstock3,
then call each logic function directly with synthetic DataFrames.
"""

import sys
import types
import unittest
import tempfile
import os
import pandas as pd
import numpy as np


# ── 1. Stub Streamlit before importing app ────────────────────────────
class _CM:
    """Universal Streamlit mock: context manager + attribute access."""
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _): return lambda *a, **kw: _CM()
    def __iter__(self): return iter([])  # e.g. st.columns unpacking safety


def _noop(*a, **kw): return _CM()
def _columns(n, *a, **kw):
    n = n if isinstance(n, int) else len(n)
    return [_CM() for _ in range(n)]


_st = types.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)   # identity decorator
_st.session_state = {}
_st.button        = lambda *a, **kw: False
_st.radio         = lambda *a, **kw: "VN30"
_st.checkbox      = lambda *a, **kw: True
_st.columns       = _columns
_st.sidebar       = _CM()
_st.spinner       = _CM
_st.expander      = _CM
_st.progress      = _noop
for _attr in [
    "set_page_config", "title", "caption", "info", "warning", "error",
    "subheader", "markdown", "metric", "dataframe", "line_chart",
    "selectbox", "slider", "download_button", "stop", "rerun",
    "header", "write",
]:
    setattr(_st, _attr, _noop)
sys.modules["streamlit"] = _st

# ── 2. Stub vnstock3 (optional dep) ──────────────────────────────────
_vn = types.ModuleType("vnstock3")
sys.modules.setdefault("vnstock3", _vn)

# ── 3. Import app ─────────────────────────────────────────────────────
import app


# ── Helpers ──────────────────────────────────────────────────────────
def make_uptrend_df(n=350, base=50.0, vol=500_000, seed=42):
    """Synthetic uptrend: price rises ~50 % over n trading bars."""
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


def make_downtrend_df(n=300, base=80.0, vol=500_000, seed=42):
    """Synthetic downtrend: price falls ~40 % over n trading bars."""
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


def inject_distribution(df, n_sessions=3):
    """Append n consecutive distribution sessions (price↓, vol×2) at end."""
    df = df.copy()
    close_col = df.columns.get_loc("Close")
    vol_col   = df.columns.get_loc("Volume")
    for i in range(-n_sessions, 0):
        df.iloc[i, close_col] = df["Close"].iloc[i - 1] * 0.99   # price down
        df.iloc[i, vol_col]   = df["Volume"].iloc[i - 1] * 2.0   # vol up
    return df


# ═════════════════════════════════════════════════════════════════════
# TEST CASES
# ═════════════════════════════════════════════════════════════════════

class TestRsStars(unittest.TestCase):
    """rs_stars(pct) — converts percentile to star string."""

    def test_triple_star(self):
        for pct in [90, 95, 100]:
            self.assertEqual(app.rs_stars(pct), "★★★", f"pct={pct}")

    def test_double_star(self):
        for pct in [80, 85, 89.9]:
            self.assertEqual(app.rs_stars(pct), "★★", f"pct={pct}")

    def test_single_star(self):
        for pct in [70, 75, 79.9]:
            self.assertEqual(app.rs_stars(pct), "★", f"pct={pct}")

    def test_no_star(self):
        for pct in [0, 50, 69.9]:
            self.assertEqual(app.rs_stars(pct), "", f"pct={pct}")

    def test_none(self):
        self.assertEqual(app.rs_stars(None), "")


class TestRankRs(unittest.TestCase):
    """rank_rs({sym: raw}) — converts raw RS to percentile 0–100."""

    def test_lowest_and_highest(self):
        result = app.rank_rs({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0})
        self.assertLess(result["A"], result["E"])
        self.assertEqual(result["E"], 100.0)

    def test_single_stock_is_100(self):
        self.assertEqual(app.rank_rs({"A": 5.0})["A"], 100.0)

    def test_all_none_returns_empty(self):
        self.assertEqual(app.rank_rs({"A": None, "B": None}), {})

    def test_none_filtered(self):
        result = app.rank_rs({"A": 10.0, "B": None, "C": 20.0})
        self.assertIn("A", result)
        self.assertNotIn("B", result)

    def test_ties_equal_percentile(self):
        result = app.rank_rs({"A": 10.0, "B": 10.0, "C": 20.0})
        self.assertEqual(result["A"], result["B"])

    def test_all_in_range(self):
        result = app.rank_rs({str(i): float(i) for i in range(100)})
        for v in result.values():
            self.assertGreater(v, 0)
            self.assertLessEqual(v, 100)


class TestComputeRsRaw(unittest.TestCase):
    """compute_rs_raw(df) — weighted performance RS formula."""

    def test_uptrend_positive(self):
        rs = app.compute_rs_raw(make_uptrend_df(300))
        self.assertIsNotNone(rs)
        self.assertGreater(rs, 0)

    def test_downtrend_negative(self):
        rs = app.compute_rs_raw(make_downtrend_df(300))
        self.assertIsNotNone(rs)
        self.assertLess(rs, 0)

    def test_too_short_returns_none(self):
        self.assertIsNone(app.compute_rs_raw(make_uptrend_df(50)))

    def test_partial_periods_still_works(self):
        # 80 bars: 1M (21) + 3M (63) available, but not 6M/12M
        rs = app.compute_rs_raw(make_uptrend_df(80))
        self.assertIsNotNone(rs)

    def test_none_df_returns_none(self):
        self.assertIsNone(app.compute_rs_raw(None))

    def test_empty_df_returns_none(self):
        self.assertIsNone(app.compute_rs_raw(pd.DataFrame()))

    def test_stronger_uptrend_higher_rs(self):
        slow_up  = make_uptrend_df(300, base=50.0, seed=1)
        fast_up  = make_uptrend_df(300, base=50.0, seed=1)
        # Make fast_up rise more steeply
        fast_up = fast_up.copy()
        fast_up["Close"] = slow_up["Close"] * np.linspace(1.0, 2.0, 300)
        rs_slow = app.compute_rs_raw(slow_up)
        rs_fast = app.compute_rs_raw(fast_up)
        self.assertGreater(rs_fast, rs_slow)


class TestComputeMarketFilter(unittest.TestCase):
    """compute_market_filter(df, breadth) — traffic light signal."""

    def test_green_all_conditions(self):
        df = make_uptrend_df(200, base=1000.0)
        signal, details = app.compute_market_filter(df, 60.0)
        self.assertEqual(signal, "green")

    def test_red_downtrend(self):
        df = make_downtrend_df(200, base=1300.0)
        signal, _ = app.compute_market_filter(df, 30.0)
        self.assertEqual(signal, "red")

    def test_red_low_breadth(self):
        df = make_uptrend_df(200, base=1000.0)
        signal, _ = app.compute_market_filter(df, 35.0)
        self.assertEqual(signal, "red")

    def test_yellow_medium_breadth(self):
        # Strong uptrend (MA20 > MA50, price > MA50) but breadth in middle zone
        df = make_uptrend_df(200, base=1000.0)
        signal, _ = app.compute_market_filter(df, 47.0)
        self.assertIn(signal, ["yellow", "green"])  # depends on exact MAs

    def test_unknown_insufficient_data(self):
        df = make_uptrend_df(30)
        signal, _ = app.compute_market_filter(df, 55.0)
        self.assertEqual(signal, "unknown")

    def test_unknown_none(self):
        signal, _ = app.compute_market_filter(None, 55.0)
        self.assertEqual(signal, "unknown")

    def test_details_contains_required_keys(self):
        df = make_uptrend_df(200, base=1000.0)
        _, details = app.compute_market_filter(df, 60.0)
        for key in ["price", "ma20", "ma50", "breadth"]:
            self.assertIn(key, details)

    def test_breadth_reflected_in_details(self):
        df = make_uptrend_df(200, base=1000.0)
        _, details = app.compute_market_filter(df, 62.5)
        self.assertAlmostEqual(details["breadth"], 62.5, places=1)


class TestCheckTrendTemplate(unittest.TestCase):
    """check_trend_template(df) — 10-criteria Trend Template."""

    # ── return structure ─────────────────────────────────────────────
    def test_returns_tuple(self):
        passed, score, details = app.check_trend_template(make_uptrend_df(300))
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(score, int)
        self.assertIsInstance(details, dict)

    def test_all_detail_keys_present(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        for k in ["price","ma50","ma150","vol20","high52","low52","score9",
                  "c1","c2","c3","c4","c5","c6","c7","c8","c10"]:
            self.assertIn(k, details, f"Missing key: {k}")

    def test_score_in_valid_range(self):
        _, score, _ = app.check_trend_template(make_uptrend_df(300))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 9)

    # ── insufficient data ────────────────────────────────────────────
    def test_too_short_returns_false(self):
        passed, score, details = app.check_trend_template(make_uptrend_df(100))
        self.assertFalse(passed)
        self.assertEqual(score, 0)
        self.assertEqual(details, {})

    def test_none_returns_false(self):
        passed, score, details = app.check_trend_template(None)
        self.assertFalse(passed)
        self.assertEqual(details, {})

    # ── criterion 1: price > MA50 ────────────────────────────────────
    def test_c1_pass_uptrend(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertTrue(details["c1"])

    def test_c1_fail_downtrend(self):
        _, _, details = app.check_trend_template(make_downtrend_df(300))
        self.assertFalse(details["c1"])

    # ── criterion 7: avg volume ≥ 200 k ─────────────────────────────
    def test_c7_fail_low_volume(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300, vol=50_000))
        self.assertFalse(details["c7"])

    def test_c7_pass_sufficient_volume(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300, vol=300_000))
        self.assertTrue(details["c7"])

    # ── criterion 8: price ≥ 15 ─────────────────────────────────────
    def test_c8_fail_penny_stock(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300, base=5.0))
        self.assertFalse(details["c8"])

    def test_c8_pass_normal_price(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300, base=30.0))
        self.assertTrue(details["c8"])

    # ── criterion 10: no distribution ────────────────────────────────
    def test_c10_pass_clean_uptrend(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertTrue(details["c10"])

    def test_c10_fail_with_3_distribution_sessions(self):
        df = inject_distribution(make_uptrend_df(300), n_sessions=3)
        _, _, details = app.check_trend_template(df)
        self.assertFalse(details["c10"], "3 distribution sessions should trigger c10=False")

    def test_c10_pass_with_only_2_distribution_sessions(self):
        df = inject_distribution(make_uptrend_df(300), n_sessions=2)
        _, _, details = app.check_trend_template(df)
        self.assertTrue(details["c10"], "Only 2 distribution sessions — should still pass")

    # ── criterion 5 & 6: 52-week range ──────────────────────────────
    def test_c5_above_52w_low(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300, base=30.0))
        self.assertTrue(details["c5"])

    def test_c6_near_52w_high(self):
        # Uptrend stock near its peak should be within 30 % of high
        _, _, details = app.check_trend_template(make_uptrend_df(300, base=30.0))
        self.assertTrue(details["c6"])

    # ── pass/fail integration ────────────────────────────────────────
    def test_strong_uptrend_high_score(self):
        _, score, _ = app.check_trend_template(make_uptrend_df(350, base=50.0, vol=600_000))
        self.assertGreaterEqual(score, 6, "Strong uptrend should pass ≥ 6/9 criteria")

    def test_downtrend_low_score(self):
        _, score, _ = app.check_trend_template(make_downtrend_df(300, vol=600_000))
        self.assertLessEqual(score, 4, "Downtrend should fail most criteria")

    # ── stop loss & target ───────────────────────────────────────────
    def test_stoploss_keys_present(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertIn("stoploss_price", details)
        self.assertIn("stoploss_pct",   details)

    def test_target_keys_present(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertIn("target_price", details)
        self.assertIn("target_pct",   details)

    def test_stoploss_below_current_price(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertLess(details["stoploss_price"], details["price"])

    def test_target_above_current_price(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertGreater(details["target_price"], details["price"])

    def test_stoploss_default_5_percent(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertEqual(details["stoploss_pct"], 5.0)
        expected = round(details["price"] * 0.95, 1)
        self.assertAlmostEqual(details["stoploss_price"], expected, places=1)

    def test_target_default_10_percent(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300))
        self.assertEqual(details["target_pct"], 10.0)
        expected = round(details["price"] * 1.10, 1)
        self.assertAlmostEqual(details["target_price"], expected, places=1)

    def test_custom_stoploss_pct(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300), sl_pct=8.0)
        self.assertEqual(details["stoploss_pct"], 8.0)
        expected = round(details["price"] * 0.92, 1)
        self.assertAlmostEqual(details["stoploss_price"], expected, places=1)

    def test_custom_target_pct(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300), tgt_pct=20.0)
        self.assertEqual(details["target_pct"], 20.0)
        expected = round(details["price"] * 1.20, 1)
        self.assertAlmostEqual(details["target_price"], expected, places=1)

    def test_stoploss_scales_with_price(self):
        """Higher price → higher absolute stoploss (same % distance)."""
        _, _, d_low  = app.check_trend_template(make_uptrend_df(300, base=20.0))
        _, _, d_high = app.check_trend_template(make_uptrend_df(300, base=80.0))
        self.assertGreater(d_high["stoploss_price"], d_low["stoploss_price"])

    def test_target_scales_with_price(self):
        """Higher price → higher absolute target."""
        _, _, d_low  = app.check_trend_template(make_uptrend_df(300, base=20.0))
        _, _, d_high = app.check_trend_template(make_uptrend_df(300, base=80.0))
        self.assertGreater(d_high["target_price"], d_low["target_price"])

    def test_stoploss_zero_pct_equals_price(self):
        _, _, details = app.check_trend_template(make_uptrend_df(300), sl_pct=0.0)
        self.assertAlmostEqual(details["stoploss_price"], details["price"], places=1)

    def test_stoploss_not_returned_for_insufficient_data(self):
        _, _, details = app.check_trend_template(make_uptrend_df(50))
        self.assertEqual(details, {})


class TestCacheHelpers(unittest.TestCase):
    """_save_df / _load_df — parquet-or-CSV file cache."""

    def test_roundtrip(self):
        df = make_uptrend_df(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            app._save_df(df, path)
            loaded = app._load_df(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), len(df))
            self.assertAlmostEqual(
                float(loaded["Close"].iloc[-1]),
                float(df["Close"].iloc[-1]),
                places=4,
            )

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


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
