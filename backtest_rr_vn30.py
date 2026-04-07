"""
backtest_rr_vn30.py — Risk/Reward backtest for pullback patterns across VN30.

Trade logic
-----------
  Entry  : Close at signal bar
  Target : entry × 1.15  (+15%)
  Stop   : entry × 0.95  (−5%)
  Max hold: 40 bars
  Win    : High touches target BEFORE Low touches stop
  Loss   : Low touches stop first

Tests four filter configurations side-by-side to quantify each filter's
impact on win rate.

Output
------
  data/backtest/rr_details.csv   — every signal row
  data/backtest/rr_summary.csv   — win rate by pattern × config
  data/backtest/rr_report.txt    — printed report saved to file
"""

import sys, types, os, warnings
warnings.filterwarnings("ignore")

# ── stub streamlit ────────────────────────────────────────────────────────────
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
           "subheader","markdown","metric","dataframe","line_chart","plotly_chart",
           "selectbox","slider","download_button","stop","rerun","header","write"]:
    setattr(_st, _a, lambda *a, **kw: None)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

import yfinance as yf
import pandas as pd
import numpy as np
import app
from chart_patterns.pullback_flag     import find_pullback_flag
from chart_patterns.pullback_pennant  import find_pullback_pennant
from chart_patterns.pullback_triangle import find_pullback_triangle

# ── constants ─────────────────────────────────────────────────────────────────
RISK_PCT    = 0.05     # 5% stop loss
REWARD_PCT  = 0.15     # 15% take profit
MAX_HOLD    = 40       # max bars to wait
HISTORY     = "2y"
OUT_DIR     = "./data/backtest"
os.makedirs(OUT_DIR, exist_ok=True)

# ── filter configurations to compare ─────────────────────────────────────────
BASE_FLAG     = dict(lookback=20, min_points=2, pole_lookback=15,
                     min_pole_gain=0.03, r_max=0.85, r_min=0.85)
BASE_PENNANT  = dict(lookback=20, min_points=2, pole_lookback=15,
                     min_pole_gain=0.03, r_max=0.85, r_min=0.85)
BASE_TRIANGLE = dict(lookback=25, min_points=2, prior_lookback=15,
                     min_prior_gain=0.03, rlimit=0.85, triangle_type="symmetrical")

CONFIGS = {
    "baseline": {
        "Flag Pullback":     BASE_FLAG,
        "Pennant Pullback":  BASE_PENNANT,
        "Triangle Pullback": BASE_TRIANGLE,
    },
    "trend_filter": {      # require price > MA50 > MA150
        "Flag Pullback":     BASE_FLAG,
        "Pennant Pullback":  BASE_PENNANT,
        "Triangle Pullback": BASE_TRIANGLE,
    },
    "vol_filter": {        # consolidation vol < pole vol × 0.7
        "Flag Pullback":     BASE_FLAG,
        "Pennant Pullback":  BASE_PENNANT,
        "Triangle Pullback": BASE_TRIANGLE,
    },
    "combined": {          # trend + vol + tighter pole gain
        "Flag Pullback":     {**BASE_FLAG,     "min_pole_gain": 0.05, "r_max": 0.90, "r_min": 0.90},
        "Pennant Pullback":  {**BASE_PENNANT,  "min_pole_gain": 0.05, "r_max": 0.90, "r_min": 0.90},
        "Triangle Pullback": {**BASE_TRIANGLE, "min_prior_gain": 0.05, "rlimit": 0.90},
    },
}

DETECTORS = [
    ("Flag Pullback",     find_pullback_flag,
     "pullback_flag_point",     "pullback_flag_pole_gain"),
    ("Pennant Pullback",  find_pullback_pennant,
     "pullback_pennant_point",  "pullback_pennant_pole_gain"),
    ("Triangle Pullback", find_pullback_triangle,
     "pullback_triangle_point", "pullback_triangle_prior_gain"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def fetch(sym: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(sym).history(period=HISTORY)
        if df is None or df.empty or len(df) < 100:
            return None
        if df.index.tz:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def passes_trend(df: pd.DataFrame, bar: int) -> bool:
    """price > MA50 > MA150 at signal bar."""
    close = df["Close"]
    if bar < 150:
        return False
    price  = float(close.iloc[bar])
    ma50   = float(close.iloc[max(0, bar-50):bar+1].mean())
    ma150  = float(close.iloc[max(0, bar-150):bar+1].mean())
    return price > ma50 > ma150


def passes_vol(df: pd.DataFrame, bar: int, lookback: int, pole_lookback: int) -> bool:
    """Consolidation avg vol < pole avg vol × 0.7."""
    pole_start = bar - lookback - pole_lookback
    pole_end   = bar - lookback
    cons_start = bar - lookback
    if pole_start < 0:
        return False
    pole_vol = float(df["Volume"].iloc[pole_start:pole_end].mean())
    cons_vol = float(df["Volume"].iloc[cons_start:bar + 1].mean())
    if pole_vol <= 0:
        return False
    return cons_vol < pole_vol * 0.7


def simulate_trade(df: pd.DataFrame, signal_bar: int) -> dict:
    """Scan forward from signal_bar; determine WIN / LOSS / OPEN."""
    entry  = float(df["Close"].iloc[signal_bar])
    target = entry * (1 + REWARD_PCT)
    stop   = entry * (1 - RISK_PCT)

    outcome = "OPEN"
    days    = MAX_HOLD
    exit_px = float(df["Close"].iloc[min(signal_bar + MAX_HOLD, len(df) - 1)])
    mae     = 0.0    # max adverse excursion (%)

    for j in range(signal_bar + 1, min(signal_bar + MAX_HOLD + 1, len(df))):
        h = float(df["High"].iloc[j])
        l = float(df["Low"].iloc[j])
        adverse = (entry - l) / entry * 100
        mae = max(mae, adverse)

        if h >= target:
            outcome = "WIN"
            days    = j - signal_bar
            exit_px = target
            break
        if l <= stop:
            outcome = "LOSS"
            days    = j - signal_bar
            exit_px = stop
            break

    ret_pct = (exit_px - entry) / entry * 100
    return {
        "entry":   round(entry,  2),
        "target":  round(target, 2),
        "stop":    round(stop,   2),
        "outcome": outcome,
        "days":    days,
        "ret_pct": round(ret_pct, 2),
        "mae":     round(mae, 2),
    }


def run_config(df: pd.DataFrame, sym: str, pname: str,
               detector_fn, point_col: str, gain_col: str,
               params: dict, config_name: str) -> list[dict]:
    """Run one detector with given params; apply config-level filters; return rows."""
    lookback     = params.get("lookback", 20)
    pole_lookback = params.get("pole_lookback", params.get("prior_lookback", 15))

    df_w  = df.reset_index()
    try:
        result = detector_fn(df_w.copy(), **params)
    except Exception as e:
        return []

    signals = result[result[point_col] > 0]
    rows = []

    for _, row in signals.iterrows():
        bar       = int(row[point_col])
        pole_gain = float(row[gain_col])

        # skip if not enough forward bars
        if bar + 5 >= len(df):
            continue

        # config-level filters
        if config_name in ("trend_filter", "combined"):
            if not passes_trend(df, bar):
                continue
        if config_name in ("vol_filter", "combined"):
            if not passes_vol(df, bar, lookback, pole_lookback):
                continue

        trade = simulate_trade(df, bar)
        trade.update({
            "symbol":   sym.replace(".VN", ""),
            "sector":   app.VN30_STOCKS.get(sym, ""),
            "pattern":  pname,
            "config":   config_name,
            "date":     str(df.index[bar])[:10],
            "pole_gain": round(pole_gain * 100, 2),
        })
        rows.append(trade)

    return rows


def summarise(rows: list[dict]) -> dict:
    decided = [r for r in rows if r["outcome"] != "OPEN"]
    if not decided:
        return {"signals": len(rows), "decided": 0, "win_rate": None,
                "avg_days": None, "avg_ret": None, "avg_mae": None}
    wins = [r for r in decided if r["outcome"] == "WIN"]
    return {
        "signals":  len(rows),
        "decided":  len(decided),
        "win_rate": round(len(wins) / len(decided) * 100, 1),
        "avg_days": round(np.mean([r["days"] for r in decided]), 1),
        "avg_ret":  round(np.mean([r["ret_pct"] for r in decided]), 2),
        "avg_mae":  round(np.mean([r["mae"] for r in decided]), 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────
all_rows: list[dict] = []

print(f"\n{'='*72}")
print(f"  VN30 R:R BACKTEST  |  Risk {RISK_PCT*100:.0f}%  Reward {REWARD_PCT*100:.0f}%"
      f"  Max hold {MAX_HOLD}d  |  {HISTORY} history")
print(f"{'='*72}\n")

for sym, sector in app.VN30_STOCKS.items():
    print(f"  {sym.replace('.VN',''):<6}…", end=" ", flush=True)
    df = fetch(sym)
    if df is None:
        print("skip")
        continue
    count = 0
    for pname, fn, pcol, gcol in DETECTORS:
        for cname, cfg_params in CONFIGS.items():
            params = cfg_params[pname]
            rows = run_config(df, sym, pname, fn, pcol, gcol, params, cname)
            all_rows.extend(rows)
            count += len(rows)
    print(f"{len(df)}b  {count} signals")

# ── save details ──────────────────────────────────────────────────────────────
df_det = pd.DataFrame(all_rows)
det_path = os.path.join(OUT_DIR, "rr_details.csv")
df_det.to_csv(det_path, index=False)
print(f"\n  Details → {det_path}  ({len(all_rows)} rows)\n")

# ── summary table ─────────────────────────────────────────────────────────────
summary_rows = []
lines = []
lines.append(f"\n{'='*72}")
lines.append(f"  WIN RATE BY PATTERN × CONFIG  "
             f"(Win = +{REWARD_PCT*100:.0f}% before −{RISK_PCT*100:.0f}%)")
lines.append(f"{'='*72}")

for pname in ["Flag Pullback", "Pennant Pullback", "Triangle Pullback"]:
    lines.append(f"\n  {pname}")
    lines.append(f"  {'Config':<16} {'Signals':>8} {'Decided':>8} "
                 f"{'Win%':>7} {'AvgRet':>8} {'AvgMAE':>8} {'AvgDays':>8}")
    lines.append(f"  {'-'*65}")

    for cname in CONFIGS:
        rows = [r for r in all_rows if r["pattern"] == pname and r["config"] == cname]
        s    = summarise(rows)
        wr   = f"{s['win_rate']:.1f}%" if s["win_rate"] is not None else "n/a"
        ar   = f"{s['avg_ret']:+.2f}%" if s["avg_ret"]  is not None else "n/a"
        mae  = f"{s['avg_mae']:.2f}%"  if s["avg_mae"]  is not None else "n/a"
        days = f"{s['avg_days']:.1f}"  if s["avg_days"] is not None else "n/a"
        flag = ("✓" if (s["win_rate"] or 0) >= 50
                else "?" if (s["win_rate"] or 0) >= 35 else "✗")
        lines.append(f"  {cname:<16} {s['signals']:>8} {s['decided']:>8} "
                     f"{wr:>7} {ar:>8} {mae:>8} {days:>8}  {flag}")
        summary_rows.append({
            "pattern": pname, "config": cname, **s,
        })

lines.append(f"\n{'='*72}")
lines.append("  IMPROVEMENT SUMMARY (combined vs baseline)")
lines.append(f"{'='*72}")
for pname in ["Flag Pullback", "Pennant Pullback", "Triangle Pullback"]:
    base  = summarise([r for r in all_rows if r["pattern"] == pname and r["config"] == "baseline"])
    comb  = summarise([r for r in all_rows if r["pattern"] == pname and r["config"] == "combined"])
    if base["win_rate"] and comb["win_rate"]:
        delta = comb["win_rate"] - base["win_rate"]
        sig_delta = comb["signals"] - base["signals"]
        lines.append(f"  {pname:<20}  win rate {base['win_rate']:.1f}% → {comb['win_rate']:.1f}%"
                     f"  ({delta:+.1f}pp)   signals {base['signals']} → {comb['signals']}"
                     f"  ({sig_delta:+d})")

# breakeven win rate for 1:3 R:R
be = 1 / (1 + REWARD_PCT / RISK_PCT) * 100
lines.append(f"\n  Breakeven win rate at {RISK_PCT*100:.0f}%/{REWARD_PCT*100:.0f}% R:R = {be:.1f}%")
lines.append(f"  (any config above {be:.1f}% is theoretically profitable)\n")

report = "\n".join(lines)
print(report)

rpt_path = os.path.join(OUT_DIR, "rr_report.txt")
with open(rpt_path, "w") as f:
    f.write(report)

df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(OUT_DIR, "rr_summary.csv"), index=False)
print(f"  Report  → {rpt_path}")
print(f"  Summary → {os.path.join(OUT_DIR, 'rr_summary.csv')}\n")
