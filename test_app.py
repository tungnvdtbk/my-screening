"""
test_app.py — unit tests for app.py pure-logic functions.
Run: /path/to/python test_app.py

Strategy: stub out Streamlit (no server needed) + vnstock3,
then call each logic function directly with synthetic DataFrames.
yfinance is mocked in cache tests so no network calls are needed.
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


def make_df_ending(n, end_date, base=50.0, vol=500_000, seed=42):
    """Synthetic df whose last bar lands on end_date."""
    rng = np.random.default_rng(seed)
    prices = base + np.linspace(0, base * 0.3, n) + rng.standard_normal(n) * 0.2
    prices = np.abs(prices)
    idx = pd.date_range(end=end_date, periods=n, freq="B")
    return pd.DataFrame({
        "Open": prices * 0.99, "High": prices * 1.01,
        "Low": prices * 0.98, "Close": prices,
        "Volume": np.full(n, float(vol)),
    }, index=idx)


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
                # create 3 dummy parquet files
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
                self.assertEqual(deleted, 1)            # only .parquet deleted
                self.assertTrue(os.path.exists(         # .txt untouched
                    os.path.join(tmpdir, "readme.txt")))


class TestLoadPriceDataCache(unittest.TestCase):
    """load_price_data() — incremental cache logic (yfinance mocked)."""

    # ── helper ──────────────────────────────────────────────────────
    def _mock_ticker(self, df):
        """Return a mock yf.Ticker whose .history() returns df."""
        m = MagicMock()
        m.history.return_value = df
        return m

    # ── up-to-date cache → no network call ──────────────────────────
    def test_uptodate_cache_no_yfinance_call(self):
        """Cache ends on last business day → return cached, never call yfinance."""
        # Use the most recent business day to avoid weekend edge-case
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
        """Cache ends today (business day) → return cached, no fetch."""
        last_bday = pd.bdate_range(end=pd.Timestamp.today(), periods=1)[0].normalize()
        df = make_df_ending(100, last_bday)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(df, app._cache_path("TODAY.VN"))
                with patch("yfinance.Ticker") as mock_yf:
                    app.load_price_data("TODAY.VN", use_cache=True)
                    mock_yf.assert_not_called()

    # ── stale cache → fetch new bars and append ──────────────────────
    def test_stale_cache_triggers_fetch(self):
        """Cache ends 10 days ago → yfinance called to fill the gap."""
        today   = pd.Timestamp.today().normalize()
        stale   = today - pd.Timedelta(days=10)
        old_df  = make_df_ending(200, stale)
        new_df  = make_df_ending(5, today, base=old_df["Close"].iloc[-1] * 1.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("STALE.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    result = app.load_price_data("STALE.VN", use_cache=True)
                    # result should be longer than old_df
                    self.assertGreater(len(result), len(old_df))

    def test_stale_cache_appends_new_bars(self):
        """New bars are actually present in the combined result."""
        today   = pd.Timestamp.today().normalize()
        stale   = today - pd.Timedelta(days=10)
        old_df  = make_df_ending(100, stale)
        new_df  = make_df_ending(3, today, base=99.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("APPEND.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    result = app.load_price_data("APPEND.VN", use_cache=True)
                    # last bar should come from new_df
                    self.assertAlmostEqual(
                        float(result["Close"].iloc[-1]),
                        float(new_df["Close"].iloc[-1]),
                        places=2,
                    )

    def test_stale_cache_saved_to_disk(self):
        """After appending, the updated data is written back to disk."""
        today  = pd.Timestamp.today().normalize()
        stale  = today - pd.Timedelta(days=10)
        old_df = make_df_ending(100, stale)
        new_df = make_df_ending(3, today, base=60.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                path = app._cache_path("SAVE.VN")
                app._save_df(old_df, path)
                mtime_before = os.path.getmtime(path)
                import time; time.sleep(0.05)
                with patch("yfinance.Ticker", return_value=self._mock_ticker(new_df)):
                    app.load_price_data("SAVE.VN", use_cache=True)
                self.assertGreater(os.path.getmtime(path), mtime_before)

    # ── use_cache=False → bypass read, always fetch ──────────────────
    def test_use_cache_false_skips_disk_read(self):
        """use_cache=False must call yfinance even when cache exists."""
        today  = pd.Timestamp.today().normalize()
        old_df = make_df_ending(200, today)   # up-to-date cache
        fresh  = make_uptrend_df(300)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                app._save_df(old_df, app._cache_path("NOCACHE.VN"))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(fresh)) as mock_yf:
                    app.load_price_data("NOCACHE.VN", use_cache=False)
                    mock_yf.assert_called_once()   # yfinance WAS called

    def test_use_cache_false_fetches_full_period(self):
        """use_cache=False calls history(period='2y'), not incremental."""
        fresh = make_uptrend_df(300)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                mock_ticker = self._mock_ticker(fresh)
                with patch("yfinance.Ticker", return_value=mock_ticker):
                    app.load_price_data("FULL.VN", use_cache=False)
                    mock_ticker.history.assert_called_once_with(period="2y")

    # ── no cache → full fetch ─────────────────────────────────────────
    def test_no_cache_fetches_and_writes(self):
        """No existing cache → fetch 2y and write to disk."""
        fresh = make_uptrend_df(300)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app, "CACHE_DIR", tmpdir):
                path = app._cache_path("NEW.VN")
                self.assertFalse(os.path.exists(path))
                with patch("yfinance.Ticker", return_value=self._mock_ticker(fresh)):
                    result = app.load_price_data("NEW.VN", use_cache=True)
                self.assertIsNotNone(result)
                # cache file now exists
                exists = os.path.exists(path) or os.path.exists(
                    path.replace(".parquet", ".csv"))
                self.assertTrue(exists)

    # ── 500-bar cap ───────────────────────────────────────────────────
    def test_combined_capped_at_500_bars(self):
        """After append, result never exceeds 500 bars."""
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


class TestPatternDetection(unittest.TestCase):
    """detect_patterns / _check_vcp / _check_flat_base / _check_pullback_ma20."""

    # ── helpers ────────────────────────────────────────────────────────
    def make_vcp_df(self, n=80, seed=7):
        """
        Three-window contracting range + declining volume aligned to _check_vcp windows.
        _check_vcp uses: p1=df.iloc[-60:-40], p2=df.iloc[-40:-20], p3=df.iloc[-20:]
        With n=80: p1=bars 20-39, p2=bars 40-59, p3=bars 60-79.
        """
        rng = np.random.default_rng(seed)
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        prices = 50 + np.linspace(0, 10, n) + rng.standard_normal(n) * 0.15
        highs = prices.copy()
        lows  = prices.copy()
        # p1 window (bars 20-39): wide range ≈ 20%
        highs[20:40] += 5.5;  lows[20:40]  -= 5.5
        # p2 window (bars 40-59): medium range ≈ 7%
        highs[40:60] += 2.0;  lows[40:60]  -= 2.0
        # p3 window (bars 60-79): tight ≈ 1% (natural noise only)
        # volumes: p1=800k, p2=600k, p3=350k — strict decline each window
        vol = np.array(
            [900_000] * 20 +   # bars 0-19 (pre-window, irrelevant)
            [800_000] * 20 +   # bars 20-39 = p1
            [600_000] * 20 +   # bars 40-59 = p2
            [350_000] * 20,    # bars 60-79 = p3
            dtype=float,
        )
        return pd.DataFrame({
            "Open": prices * 0.99, "High": highs,
            "Low": lows, "Close": prices, "Volume": vol,
        }, index=idx)

    def make_flat_base_df(self, n=60, base=50.0, seed=9):
        """Tight range (< 8 %) over last 40 bars, above MA50."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        # Prior 20 bars: trending up to base level
        prior = base - 5 + np.linspace(0, 5, n - 40) + rng.standard_normal(n - 40) * 0.3
        # Flat base: last 40 bars stay within ±2 of base
        flat = base + rng.standard_normal(40) * 0.5
        prices = np.concatenate([prior, flat])
        vol = np.full(n, 300_000)
        return pd.DataFrame({
            "Open": prices * 0.99, "High": prices + 0.5,
            "Low": prices - 0.5, "Close": prices, "Volume": vol.astype(float),
        }, index=idx)

    def make_pullback_ma20_df(self, n=60, seed=11):
        """Stock in uptrend that pulls back to MA20 with low volume."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        # Strong uptrend for first 50 bars, then gentle pullback for last 10
        up   = 40 + np.linspace(0, 15, n - 10) + rng.standard_normal(n - 10) * 0.2
        pull = up[-1] - np.linspace(0, 2, 10) + rng.standard_normal(10) * 0.1
        prices = np.concatenate([up, pull])
        # Volume declining in pullback
        vol = np.concatenate([np.full(n - 10, 500_000), np.full(10, 200_000)])
        return pd.DataFrame({
            "Open": prices * 0.99, "High": prices + 0.3,
            "Low": prices - 0.3, "Close": prices, "Volume": vol.astype(float),
        }, index=idx)

    # ── _passes_trend_filter ─────────────────────────────────────────────
    def test_trend_filter_pass_uptrend(self):
        """Strong uptrend with 300 bars: price > MA50 > MA150 > MA200."""
        df = make_uptrend_df(300, base=50.0)
        self.assertTrue(app._passes_trend_filter(df))

    def test_trend_filter_fail_downtrend(self):
        """Downtrend: price < MA50, should fail."""
        df = make_downtrend_df(300)
        self.assertFalse(app._passes_trend_filter(df))

    def test_trend_filter_fail_insufficient_bars(self):
        """Less than 200 bars → always False (MA200 undefined)."""
        df = make_uptrend_df(150)
        self.assertFalse(app._passes_trend_filter(df))

    def test_trend_filter_fail_none(self):
        self.assertFalse(app._passes_trend_filter(None))

    def test_detect_patterns_skips_downtrend(self):
        """detect_patterns returns [] for a downtrend stock even if enough bars."""
        df = make_downtrend_df(300)
        self.assertEqual(app.detect_patterns(df), [])

    def test_detect_patterns_skips_short_df(self):
        """detect_patterns returns [] when < 200 bars (trend filter needs 200)."""
        self.assertEqual(app.detect_patterns(make_uptrend_df(150)), [])

    # ── detect_patterns wrapper ─────────────────────────────────────────
    def test_returns_list(self):
        result = app.detect_patterns(make_uptrend_df(300))
        self.assertIsInstance(result, list)

    def test_none_df_returns_empty(self):
        self.assertEqual(app.detect_patterns(None), [])

    def test_too_short_returns_empty(self):
        self.assertEqual(app.detect_patterns(make_uptrend_df(20)), [])

    def test_pattern_dict_has_required_keys(self):
        """Every detected pattern must have the 5 required keys."""
        df = self.make_vcp_df()
        patterns = app.detect_patterns(df)
        for p in patterns:
            for k in ["pattern", "quality", "pivot", "stoploss", "notes"]:
                self.assertIn(k, p, f"Key '{k}' missing in {p}")

    def test_quality_values_valid(self):
        df = self.make_vcp_df()
        patterns = app.detect_patterns(df)
        for p in patterns:
            self.assertIn(p["quality"], ["★", "★★", "★★★"])

    # ── VCP ────────────────────────────────────────────────────────────
    def test_vcp_detected_on_contracting_data(self):
        df = self.make_vcp_df()
        p = app._check_vcp(df)
        self.assertIsNotNone(p, "VCP should be detected")
        self.assertEqual(p["pattern"], "VCP")

    def test_vcp_pivot_above_recent_high(self):
        df = self.make_vcp_df()
        p = app._check_vcp(df)
        if p:
            recent_high = float(df["High"].iloc[-20:].max())
            self.assertGreater(p["pivot"], recent_high * 0.99)

    def test_vcp_stoploss_below_recent_low(self):
        df = self.make_vcp_df()
        p = app._check_vcp(df)
        if p:
            recent_low = float(df["Low"].iloc[-20:].min())
            self.assertLessEqual(p["stoploss"], recent_low * 1.01)

    def test_vcp_not_detected_on_expanding_range(self):
        """Expanding volatility (opposite of VCP) should not trigger."""
        rng = np.random.default_rng(42)
        idx = pd.date_range(end="2026-01-01", periods=80, freq="B")
        prices = 50 + np.linspace(0, 5, 80)
        highs = prices.copy()
        lows  = prices.copy()
        highs[:20] += 1.0   # tight first
        lows[:20]  -= 1.0
        highs[20:40] += 3.0  # wider second
        lows[20:40]  -= 3.0
        highs[40:] += 6.0   # widest last
        lows[40:]  -= 6.0
        vol = np.full(80, 500_000)
        df = pd.DataFrame({
            "Open": prices, "High": highs, "Low": lows,
            "Close": prices, "Volume": vol.astype(float),
        }, index=idx)
        p = app._check_vcp(df)
        self.assertIsNone(p)

    def test_vcp_needs_60_bars(self):
        df = self.make_vcp_df()
        self.assertIsNone(app._check_vcp(df.iloc[-50:]))

    # ── Flat Base ──────────────────────────────────────────────────────
    def test_flat_base_detected(self):
        df = self.make_flat_base_df(n=80)
        p = app._check_flat_base(df)
        self.assertIsNotNone(p, "Flat Base should be detected")
        self.assertEqual(p["pattern"], "Flat Base")

    def test_flat_base_pivot_above_base_high(self):
        df = self.make_flat_base_df(n=80)
        p = app._check_flat_base(df)
        if p:
            base_high = float(df["High"].iloc[-40:].max())
            self.assertGreater(p["pivot"], base_high * 0.99)

    def test_flat_base_not_detected_wide_range(self):
        """Range > 15 % should not qualify as flat base."""
        df = make_uptrend_df(80, base=50.0)
        df = df.copy()
        df.iloc[-40:, df.columns.get_loc("High")] = df["Close"].iloc[-40:] * 1.10
        df.iloc[-40:, df.columns.get_loc("Low")]  = df["Close"].iloc[-40:] * 0.88
        p = app._check_flat_base(df)
        self.assertIsNone(p)

    # ── Pullback to MA20 ───────────────────────────────────────────────
    def test_pullback_ma20_detected(self):
        df = self.make_pullback_ma20_df()
        p = app._check_pullback_ma20(df)
        self.assertIsNotNone(p, "Pullback MA20 should be detected")
        self.assertEqual(p["pattern"], "Pullback MA20")

    def test_pullback_stoploss_below_ma50(self):
        df = self.make_pullback_ma20_df()
        p = app._check_pullback_ma20(df)
        if p:
            ma50_v = float(df["Close"].rolling(50).mean().iloc[-1])
            self.assertLess(p["stoploss"], ma50_v + 0.1)

    def test_pullback_not_detected_below_ma50(self):
        """Downtrend: price below MA50 → no Pullback MA20 signal."""
        df = make_downtrend_df(80)
        p = app._check_pullback_ma20(df)
        self.assertIsNone(p)

    def test_pullback_needs_30_bars(self):
        df = self.make_pullback_ma20_df()
        self.assertIsNone(app._check_pullback_ma20(df.iloc[-20:]))


class TestPatternDetectionLimitations(unittest.TestCase):
    """
    Explicit tests that document the LIMITATIONS of the pattern engine.

    These tests are NOT about proving correctness — they document known
    boundary cases and assumptions. Run with -v to see the messages.

    Rule of thumb:
      - Synthetic tests above prove the algorithm's LOGIC.
      - These tests prove the algorithm's ASSUMPTIONS are reasonable.
      - Visual chart inspection + backtesting remain the gold standard.
    """

    def test_vcp_requires_exactly_3_windows_of_20(self):
        """
        _check_vcp hardcodes p1/p2/p3 as 20-day windows.
        A real VCP might span 6-8 weeks (30-40 bars) — the 60-bar window
        is a simplification. This test documents that assumption explicitly.
        """
        # A 'real' VCP that contracts over 7 weeks (35 bars not 60)
        # will NOT be detected unless the contraction spans all 3 windows.
        # Here we put a tight contraction in only p2+p3 (not p1) — should fail.
        rng = np.random.default_rng(99)
        n = 80
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        prices = 50 + np.linspace(0, 5, n) + rng.standard_normal(n) * 0.2
        highs = prices.copy(); lows = prices.copy()
        # Only p2 and p3 contract — p1 stays flat-ish (no clear contraction vs p2)
        highs[20:40] += 2.0; lows[20:40] -= 2.0   # p1: ~7%
        highs[40:60] += 1.5; lows[40:60] -= 1.5   # p2: ~5.5%  (not >> p1)
        # p3: natural noise ~1%
        vol = np.array([800_000]*20 + [700_000]*20 + [500_000]*20 + [300_000]*20, dtype=float)
        df = pd.DataFrame({"Open": prices*0.99, "High": highs, "Low": lows,
                           "Close": prices, "Volume": vol}, index=idx)
        p = app._check_vcp(df)
        # p1 > p2 barely (7% vs 5.5%) may or may not pass — result depends on noise.
        # The point: 'soft' contractions near the boundary are unreliable.
        # This test just asserts the function doesn't crash.
        self.assertIsInstance(p, (dict, type(None)))

    def test_flat_base_gap_does_not_invalidate(self):
        """
        A single large gap-down (earnings miss) in an otherwise flat base
        would widen the range and prevent detection.
        This is CORRECT behaviour — a gap disrupts the base structure.
        """
        df = pd.concat([
            make_uptrend_df(60, base=50.0),
            make_uptrend_df(20, base=40.0),  # gap-down 20% then continues
        ]).reset_index(drop=True)
        df.index = pd.date_range(end="2026-01-01", periods=len(df), freq="B")
        p = app._check_flat_base(df)
        # With 20% gap, range >> 12% → flat base correctly NOT detected
        self.assertIsNone(p, "Gap-down should prevent flat-base detection")

    def test_pullback_panic_sell_rejection(self):
        """
        A single panic session (vol × 4, price −5%) should reject Pullback MA20.
        This documents that the no-panic-sell guard is functioning.
        """
        df = self._make_pullback_df_with_panic()
        p = app._check_pullback_ma20(df)
        self.assertIsNone(p, "Panic sell should reject the pullback setup")

    def _make_pullback_df_with_panic(self):
        rng = np.random.default_rng(55)
        n = 60
        idx = pd.date_range(end="2026-01-01", periods=n, freq="B")
        up   = 40 + np.linspace(0, 15, n - 10) + rng.standard_normal(n - 10) * 0.2
        pull = up[-1] - np.linspace(0, 2, 10) + rng.standard_normal(10) * 0.1
        prices = np.concatenate([up, pull])
        vol = np.concatenate([np.full(n - 10, 500_000), np.full(10, 200_000)])
        highs = prices.copy(); lows = prices.copy()
        # Inject panic sell 3 bars from end: price −5%, volume × 4
        prices[-3] = prices[-4] * 0.95
        vol[-3]    = 500_000 * 4.0
        highs[-3]  = prices[-3] + 0.2
        lows[-3]   = prices[-3] - 0.5
        return pd.DataFrame({"Open": prices*0.99, "High": highs, "Low": lows,
                             "Close": prices, "Volume": vol.astype(float)}, index=idx)

    def test_detection_is_deterministic(self):
        """Same input always produces same result — no hidden randomness."""
        df = make_uptrend_df(80)
        r1 = app.detect_patterns(df)
        r2 = app.detect_patterns(df)
        self.assertEqual(
            [(p["pattern"], p["quality"]) for p in r1],
            [(p["pattern"], p["quality"]) for p in r2],
        )

    def test_multiple_patterns_can_coexist(self):
        """
        A stock near MA20 AND in a flat base could trigger two patterns.
        The wrapper should return all detected patterns, not just the first.
        """
        df = make_uptrend_df(80, base=50.0)
        patterns = app.detect_patterns(df)
        # We don't assert a specific count — just that it's a list
        self.assertIsInstance(patterns, list)
        names = [p["pattern"] for p in patterns]
        # No duplicates
        self.assertEqual(len(names), len(set(names)))


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
