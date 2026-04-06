"""
test_backtest.py — Backtest pattern detection on real market data.

Downloads 3 years of daily OHLCV for well-known US/commodity symbols,
slides a detection window across the history, measures hit rate + returns,
then prints calibration suggestions.

Run (needs network):
    python test_backtest.py

Output: per-symbol, per-pattern hit rate and expected return.
Thresholds are printed at the end if adjustments are recommended.
"""

import sys
import types
import warnings
warnings.filterwarnings("ignore")

# ── Stub Streamlit ────────────────────────────────────────────────────
class _CM:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _): return lambda *a, **kw: _CM()

_st = types.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)
_st.session_state = {}
_st.button        = lambda *a, **kw: False
_st.radio         = lambda *a, **kw: "VN30"
_st.checkbox      = lambda *a, **kw: True
_st.columns       = lambda n, **kw: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
_st.sidebar       = _CM()
_st.spinner       = _CM
_st.expander      = _CM
_st.progress      = lambda *a, **kw: _CM()
for _a in ["set_page_config","title","caption","info","warning","error",
           "subheader","markdown","metric","dataframe","line_chart",
           "selectbox","slider","download_button","stop","rerun","header","write"]:
    setattr(_st, _a, lambda *a, **kw: None)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

import yfinance as yf
import pandas as pd
import numpy as np
import app

# ── Config ────────────────────────────────────────────────────────────
SYMBOLS = {
    "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "IBM":   "IBM",
    "GOOGL": "Google/Alphabet",
    "GC=F":  "Gold",
    "^GSPC": "S&P 500",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon",
}

LOOKAHEAD   = 15   # trading sessions to wait for breakout after signal
COOLDOWN    = 10   # min sessions between signals on same symbol
HISTORY     = "3y"
MIN_SIGNALS = 3    # skip symbols with too few signals


# ── Backtest engine ───────────────────────────────────────────────────
def fetch(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=HISTORY)
        if df is None or df.empty or len(df) < 120:
            return None
        # strip tz
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception as e:
        print(f"  [fetch error] {ticker}: {e}")
        return None


def backtest_one(df: pd.DataFrame, check_fn, symbol: str) -> list[dict]:
    """
    Slide a window over df.  At each bar where a pattern is detected,
    record entry, pivot, and whether price exceeded pivot within LOOKAHEAD.
    """
    results = []
    last_signal_idx = -COOLDOWN

    for i in range(100, len(df) - LOOKAHEAD):
        if i - last_signal_idx < COOLDOWN:
            continue

        window = df.iloc[max(0, i - 250): i]   # max 250 bars of history
        if len(window) < 80:
            continue

        pattern = check_fn(window)
        if pattern is None:
            continue

        last_signal_idx = i
        entry = float(df["Close"].iloc[i])
        pivot = pattern["pivot"]
        future = df["Close"].iloc[i + 1: i + LOOKAHEAD + 1]

        breakout_day = None
        exit_price   = float(future.iloc[-1]) if len(future) else entry
        for j, p in enumerate(future):
            if float(p) >= pivot:
                breakout_day  = j + 1
                exit_price    = float(p)
                break

        ret = (exit_price - entry) / entry * 100

        results.append({
            "symbol":          symbol,
            "date":            str(df.index[i])[:10],
            "pattern":         pattern["pattern"],
            "quality":         pattern["quality"],
            "entry":           round(entry, 2),
            "pivot":           pivot,
            "exit":            round(exit_price, 2),
            "return_pct":      round(ret, 2),
            "hit":             breakout_day is not None,
            "days_to_breakout": breakout_day,
        })

    return results


def summarise(rows: list[dict], label: str) -> dict:
    if not rows:
        return {}
    total  = len(rows)
    hits   = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]
    hit_rate   = len(hits)   / total * 100
    avg_ret    = sum(r["return_pct"] for r in rows) / total
    avg_hit    = sum(r["return_pct"] for r in hits)   / len(hits)   if hits   else 0.0
    avg_miss   = sum(r["return_pct"] for r in misses) / len(misses) if misses else 0.0
    avg_days   = sum(r["days_to_breakout"] for r in hits) / len(hits) if hits else 0.0
    return {
        "label":     label,
        "total":     total,
        "hit_rate":  round(hit_rate, 1),
        "avg_ret":   round(avg_ret, 2),
        "avg_hit":   round(avg_hit, 2),
        "avg_miss":  round(avg_miss, 2),
        "avg_days":  round(avg_days, 1),
    }


# ── Run ───────────────────────────────────────────────────────────────
all_rows: dict[str, list[dict]] = {"VCP": [], "Flat Base": [], "Pullback MA20": []}

print(f"\n{'='*70}")
print(f"  PATTERN BACKTEST  —  {HISTORY} history  |  lookahead {LOOKAHEAD} sessions")
print(f"{'='*70}\n")

for ticker, name in SYMBOLS.items():
    print(f"  Fetching {name} ({ticker})…", end=" ", flush=True)
    df = fetch(ticker)
    if df is None:
        print("SKIP (no data)")
        continue
    print(f"{len(df)} bars")

    for check_fn in [app._check_vcp, app._check_flat_base, app._check_pullback_ma20]:
        rows = backtest_one(df, check_fn, ticker)
        if rows:
            pname = rows[0]["pattern"]
            all_rows[pname].extend(rows)

print()

# ── Per-pattern summary ───────────────────────────────────────────────
CHECKERS = {
    "VCP":          app._check_vcp,
    "Flat Base":    app._check_flat_base,
    "Pullback MA20": app._check_pullback_ma20,
}

for pname, rows in all_rows.items():
    if len(rows) < MIN_SIGNALS:
        print(f"[{pname}] — too few signals ({len(rows)}), skipping\n")
        continue

    # Overall summary
    s = summarise(rows, pname)
    print(f"{'─'*60}")
    print(f"  {pname}")
    print(f"  Signals: {s['total']}  |  Hit rate: {s['hit_rate']}%  "
          f"|  Avg return: {s['avg_ret']:+.2f}%")
    print(f"  On hit:  {s['avg_hit']:+.2f}%  |  On miss: {s['avg_miss']:+.2f}%  "
          f"|  Avg days to breakout: {s['avg_days']}")

    # Per-symbol breakdown
    by_sym = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)
    print(f"  {'Symbol':<8} {'Signals':>7} {'Hit%':>7} {'AvgRet':>8}")
    for sym, sym_rows in sorted(by_sym.items()):
        ss = summarise(sym_rows, sym)
        flag = "  ✓" if ss["hit_rate"] >= 50 else ("  ?" if ss["hit_rate"] >= 35 else "  ✗")
        print(f"  {sym:<8} {ss['total']:>7} {ss['hit_rate']:>6.1f}% {ss['avg_ret']:>+7.2f}%{flag}")

    # Quality breakdown
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quality"], []).append(r)
    if len(by_q) > 1:
        print(f"  Quality breakdown:")
        for q in ["★★★", "★★", "★"]:
            if q in by_q:
                qs = summarise(by_q[q], q)
                print(f"    {q}  signals={qs['total']}  hit={qs['hit_rate']}%  "
                      f"ret={qs['avg_ret']:+.2f}%")
    print()

# ── Calibration report ────────────────────────────────────────────────
print(f"{'='*70}")
print("  CALIBRATION REPORT")
print(f"{'='*70}")

adjustments = []

for pname, rows in all_rows.items():
    if len(rows) < MIN_SIGNALS:
        continue
    s = summarise(rows, pname)

    if s["hit_rate"] < 35:
        adjustments.append(
            f"  ⚠  {pname}: hit rate {s['hit_rate']}% is LOW — "
            "consider RELAXING thresholds (wider range tolerance or looser vol)"
        )
    elif s["hit_rate"] > 70 and s["total"] < 10:
        adjustments.append(
            f"  ℹ  {pname}: hit rate {s['hit_rate']}% is high but only {s['total']} signals — "
            "thresholds may be TOO TIGHT (missing valid patterns)"
        )
    elif s["hit_rate"] >= 45:
        adjustments.append(
            f"  ✓  {pname}: hit rate {s['hit_rate']}% — thresholds look reasonable"
        )

    # Check if ★★★ outperforms ★
    by_q = {}
    for r in rows:
        by_q.setdefault(r["quality"], []).append(r)
    if "★★★" in by_q and "★" in by_q:
        top = summarise(by_q["★★★"], "★★★")
        bot = summarise(by_q["★"], "★")
        if top["hit_rate"] > bot["hit_rate"] + 10:
            adjustments.append(
                f"  ✓  {pname}: ★★★ ({top['hit_rate']}%) beats ★ ({bot['hit_rate']}%) — quality filter is effective"
            )
        elif top["hit_rate"] < bot["hit_rate"]:
            adjustments.append(
                f"  ⚠  {pname}: ★★★ ({top['hit_rate']}%) WORSE than ★ ({bot['hit_rate']}%) — quality criteria need review"
            )

for line in adjustments:
    print(line)

print(f"\n  NOTE: Lookahead={LOOKAHEAD} sessions, Cooldown={COOLDOWN} sessions")
print("  Symbols tested:", ", ".join(SYMBOLS.keys()))
print()

# ── Raw signal dump (top 10 by return) ───────────────────────────────
all_flat = [r for rows in all_rows.values() for r in rows]
if all_flat:
    all_flat.sort(key=lambda r: r["return_pct"], reverse=True)
    print(f"{'─'*60}")
    print("  TOP 10 SIGNALS BY RETURN")
    print(f"  {'Date':<12} {'Sym':<8} {'Pattern':<14} {'Q':>3} {'Entry':>8} "
          f"{'Pivot':>8} {'Exit':>8} {'Ret%':>7} {'Hit'}")
    for r in all_flat[:10]:
        print(f"  {r['date']:<12} {r['symbol']:<8} {r['pattern']:<14} "
              f"{r['quality']:>3} {r['entry']:>8.2f} {r['pivot']:>8.2f} "
              f"{r['exit']:>8.2f} {r['return_pct']:>+6.2f}% "
              f"{'✓' if r['hit'] else '✗'}")

print()
