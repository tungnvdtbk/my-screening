"""
backtest_pullback_vn30.py — Backtest pullback patterns (flag, pennant, triangle)
across all 30 VN30 symbols.

Run:
    python backtest_pullback_vn30.py

Output:
    data/backtest/vn30_pullback_details.csv   — every detected signal + forward returns
    data/backtest/vn30_pullback_summary.csv   — per-pattern summary stats
    data/backtest/evidence_*.png              — candlestick charts for top signals
"""

import sys, types, os, warnings
warnings.filterwarnings("ignore")

# ── stub streamlit so app.py can be imported ─────────────────────────────────
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

# chart_patterns project root lives at /chart_patterns; its __init__.py is there,
# so we add the PARENT dir ("/") so that `chart_patterns.chart_patterns.*` resolves.
for _cp_parent in ["/", "/app"]:
    if _cp_parent not in sys.path and os.path.isdir(os.path.join(_cp_parent, "chart_patterns")):
        sys.path.insert(0, _cp_parent)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import app
from chart_patterns.chart_patterns.pullback_flag     import find_pullback_flag
from chart_patterns.chart_patterns.pullback_pennant  import find_pullback_pennant
from chart_patterns.chart_patterns.pullback_triangle import find_pullback_triangle

# ── config ────────────────────────────────────────────────────────────────────
LOOKAHEAD       = 20        # bars to measure forward returns
HISTORY         = "2y"
OUT_DIR         = "./data/backtest"
os.makedirs(OUT_DIR, exist_ok=True)

FLAG_PARAMS     = dict(lookback=20, min_points=2, pole_lookback=15,
                       min_pole_gain=0.03, r_max=0.85, r_min=0.85)
PENNANT_PARAMS  = dict(lookback=20, min_points=2, pole_lookback=15,
                       min_pole_gain=0.03, r_max=0.85, r_min=0.85)
TRIANGLE_PARAMS = dict(lookback=25, min_points=2, prior_lookback=15,
                       min_prior_gain=0.03, rlimit=0.85, triangle_type="symmetrical")

# chart colours (dark theme)
BG, CARD, TEXT, MUTED = "#0a0e17", "#131829", "#e2e8f0", "#8892a8"
MA_COLORS = {"MA20": "#60a5fa", "MA50": "#34d399"}

# ── helpers ───────────────────────────────────────────────────────────────────

def fetch(sym: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(sym).history(period=HISTORY)
        if df is None or df.empty or len(df) < 80:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception as e:
        print(f"    [fetch error] {sym}: {e}")
        return None


def run_detector(df_yf: pd.DataFrame, detector_fn, point_col: str,
                 gain_col: str, params: dict) -> list[dict]:
    """Run one pullback detector on the full price history; return signal list."""
    df_work = df_yf.reset_index()          # numeric index; Date becomes column
    try:
        df_det = detector_fn(df_work.copy(), **params)
    except Exception as e:
        print(f"    [detector error] {e}")
        return []

    signals = df_det[df_det[point_col] > 0]
    rows = []
    for _, row in signals.iterrows():
        idx = int(row[point_col])
        if idx + 1 >= len(df_yf):          # no forward bars
            continue

        entry = float(df_yf["Close"].iloc[idx])
        future = df_yf["Close"].iloc[idx + 1: idx + LOOKAHEAD + 1].values

        ret_5  = (future[4]  / entry - 1) * 100 if len(future) >  4 else np.nan
        ret_10 = (future[9]  / entry - 1) * 100 if len(future) >  9 else np.nan
        ret_20 = (future[19] / entry - 1) * 100 if len(future) > 19 else np.nan
        max_ret = (max(future) / entry - 1) * 100 if len(future) else np.nan

        rows.append({
            "detect_idx":   idx,
            "date":         str(df_yf.index[idx])[:10],
            "entry":        round(entry, 2),
            "pole_gain":    round(float(row[gain_col]) * 100, 2),
            "ret_5":        round(ret_5,  2) if not np.isnan(ret_5)  else None,
            "ret_10":       round(ret_10, 2) if not np.isnan(ret_10) else None,
            "ret_20":       round(ret_20, 2) if not np.isnan(ret_20) else None,
            "max_ret_20":   round(max_ret, 2) if not np.isnan(max_ret) else None,
            "win_20":       bool(ret_20 > 0)  if not np.isnan(ret_20) else None,
            # trendline data for charting
            "_slmax":       row.get(point_col.replace("_point", "_slmax"), np.nan),
            "_slmin":       row.get(point_col.replace("_point", "_slmin"), np.nan),
            "_intercmax":   row.get(point_col.replace("_point", "_intercmax"), np.nan),
            "_intercmin":   row.get(point_col.replace("_point", "_intercmin"), np.nan),
        })
    return rows


def draw_candle(ax, x, o, h, l, c, width=0.6):
    col = "#34d399" if c >= o else "#ef4444"
    ax.plot([x, x], [l, h], color=col, lw=0.7, zorder=2)
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, min(o, c)), width, max(abs(c - o), 1e-6),
        boxstyle="square,pad=0", facecolor=col, edgecolor=col, lw=0, zorder=3)
    ax.add_patch(rect)


def draw_evidence(sym: str, df_yf: pd.DataFrame, sig: dict,
                  pattern_name: str, out_path: str) -> None:
    """Draw a candlestick chart showing the detected pattern + 20-bar forward."""
    idx   = sig["detect_idx"]
    start = max(0, idx - 45)
    end   = min(len(df_yf), idx + LOOKAHEAD + 1)
    view  = df_yf.iloc[start:end].copy()
    n     = len(view)
    det_x = idx - start           # detection bar in local coords

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = plt.GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1], hspace=0.04)
    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    for a in [ax, axv]:
        a.set_facecolor(CARD)
        a.tick_params(colors=MUTED, labelsize=7)
        for s in a.spines.values():
            s.set_edgecolor("#2a3050")

    xs = list(range(n))

    # candlesticks (pre-signal: normal, post-signal: muted)
    for i in xs:
        o = float(view["Open"].iloc[i])
        h = float(view["High"].iloc[i])
        l = float(view["Low"].iloc[i])
        c = float(view["Close"].iloc[i])
        if i <= det_x:
            draw_candle(ax, i, o, h, l, c)
        else:
            col = "#34d39955" if c >= o else "#ef444455"
            ax.plot([i, i], [l, h], color=col, lw=0.7, zorder=2)
            rect = mpatches.FancyBboxPatch(
                (i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-6),
                boxstyle="square,pad=0", facecolor=col, edgecolor=col, lw=0, zorder=3)
            ax.add_patch(rect)

    # MA lines
    close = view["Close"]
    for w, col in [(20, MA_COLORS["MA20"]), (50, MA_COLORS["MA50"])]:
        ma = close.rolling(w, min_periods=max(2, w // 2)).mean()
        vals = [(i, float(v)) for i, v in enumerate(ma) if not pd.isna(v)]
        if vals:
            xi, yi = zip(*vals)
            ax.plot(xi, yi, color=col, lw=0.9, alpha=0.8, label=f"MA{w}")

    # trendlines (over detection window)
    slmax, slmin   = sig["_slmax"],   sig["_slmin"]
    imax, imin     = sig["_intercmax"], sig["_intercmin"]
    if not any(np.isnan(v) for v in [slmax, slmin, imax, imin]):
        # shift from reset-index coords to local coords
        x0 = start
        xs_tl = np.array([det_x - 22, det_x + 2])
        ax.plot(xs_tl, slmax * (xs_tl + x0) + imax,
                color="#ef4444", lw=1.3, linestyle="--", alpha=0.9, label="High TL", zorder=4)
        ax.plot(xs_tl, slmin * (xs_tl + x0) + imin,
                color="#f59e0b", lw=1.3, linestyle="--", alpha=0.9, label="Low TL",  zorder=4)

    # detection marker
    entry = sig["entry"]
    ret20 = sig.get("ret_20")
    ret_label = f"+{ret20:.1f}%" if ret20 and ret20 >= 0 else (f"{ret20:.1f}%" if ret20 else "n/a")
    mc = "#ef4444" if (ret20 or 0) >= 0 else "#6b7280"
    ax.scatter([det_x], [entry], s=130, color=mc, marker="^", zorder=6,
               label=f"Signal  20d: {ret_label}")

    # vertical line at detection
    ax.axvline(det_x, color="#ffffff22", lw=1, linestyle=":")

    # title
    ticker = sym.replace(".VN", "")
    sector = app.VN30_STOCKS.get(sym, "")
    ax.set_title(
        f"{ticker}  [{sector}]  ·  {pattern_name}  "
        f"·  Pole/Prior gain: {sig['pole_gain']:.1f}%  "
        f"·  Entry: {entry:.1f}  ·  20d return: {ret_label}",
        color=TEXT, fontsize=8, pad=6, loc="left")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.3,
              labelcolor="white", facecolor=CARD, ncol=4)
    ax.set_xlim(-1, n + 1)
    plt.setp(ax.get_xticklabels(), visible=False)

    # volume
    vol_colors = ["#ef444455" if float(view["Close"].iloc[i]) < float(view["Open"].iloc[i])
                  else "#34d39955" for i in xs]
    axv.bar(xs, view["Volume"], color=vol_colors, width=0.8)
    axv.set_yticks([])

    step = max(1, n // 6)
    axv.set_xticks(range(0, n, step))
    axv.set_xticklabels(
        [str(view.index[i])[:10] for i in range(0, n, step)],
        rotation=25, ha="right", fontsize=6, color=MUTED)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    total    = len(rows)
    with_20  = [r for r in rows if r["ret_20"] is not None]
    wins     = [r for r in with_20 if r["win_20"]]
    win_rate = len(wins) / len(with_20) * 100 if with_20 else 0
    avg_r20  = np.mean([r["ret_20"]  for r in with_20])  if with_20 else 0
    avg_r10  = np.mean([r["ret_10"]  for r in rows if r["ret_10"]  is not None])
    avg_r5   = np.mean([r["ret_5"]   for r in rows if r["ret_5"]   is not None])
    avg_max  = np.mean([r["max_ret_20"] for r in rows if r["max_ret_20"] is not None])
    return {
        "signals":  total,
        "win_rate": round(win_rate, 1),
        "avg_ret_5":  round(avg_r5,  2),
        "avg_ret_10": round(avg_r10, 2),
        "avg_ret_20": round(avg_r20, 2),
        "avg_max_ret": round(avg_max, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────
DETECTORS = [
    ("Flag Pullback",     find_pullback_flag,     "pullback_flag_point",
     "pullback_flag_pole_gain",     FLAG_PARAMS),
    ("Pennant Pullback",  find_pullback_pennant,  "pullback_pennant_point",
     "pullback_pennant_pole_gain",  PENNANT_PARAMS),
    ("Triangle Pullback", find_pullback_triangle, "pullback_triangle_point",
     "pullback_triangle_prior_gain", TRIANGLE_PARAMS),
]

all_details: list[dict] = []

print(f"\n{'='*70}")
print(f"  VN30 PULLBACK PATTERN BACKTEST  |  {HISTORY} history  |  lookahead {LOOKAHEAD}d")
print(f"{'='*70}\n")

for sym, sector in app.VN30_STOCKS.items():
    print(f"  {sym.replace('.VN',''):<6} ({sector})…", end=" ", flush=True)
    df = fetch(sym)
    if df is None:
        print("no data")
        continue
    print(f"{len(df)} bars", end="")

    sym_count = 0
    for pname, fn, pcol, gcol, params in DETECTORS:
        sigs = run_detector(df, fn, pcol, gcol, params)
        for s in sigs:
            s["symbol"]  = sym.replace(".VN", "")
            s["sector"]  = sector
            s["pattern"] = pname
            all_details.append(s)
        sym_count += len(sigs)

    print(f"  → {sym_count} signals")

print(f"\n  Total signals: {len(all_details)}\n")

# ── save details CSV ──────────────────────────────────────────────────────────
cols = ["symbol", "sector", "pattern", "date", "entry", "pole_gain",
        "ret_5", "ret_10", "ret_20", "max_ret_20", "win_20"]
df_det = pd.DataFrame(all_details)[cols]
det_path = os.path.join(OUT_DIR, "vn30_pullback_details.csv")
df_det.to_csv(det_path, index=False)
print(f"  Details → {det_path}")

# ── summary per pattern ───────────────────────────────────────────────────────
summary_rows = []
for pname, _, _, _, _ in DETECTORS:
    rows = [r for r in all_details if r["pattern"] == pname]
    s    = summarise(rows)
    if not s:
        continue
    s["pattern"] = pname
    summary_rows.append(s)

    print(f"\n  {'─'*60}")
    print(f"  {pname}")
    print(f"  Signals: {s['signals']}  |  Win rate (20d): {s['win_rate']}%")
    print(f"  Avg ret  5d: {s['avg_ret_5']:+.2f}%  "
          f"10d: {s['avg_ret_10']:+.2f}%  "
          f"20d: {s['avg_ret_20']:+.2f}%  "
          f"Max 20d: {s['avg_max_ret']:+.2f}%")

    # per-symbol breakdown
    by_sym = {}
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)
    print(f"  {'Symbol':<8} {'Signals':>7} {'Win%':>6} {'Ret20':>8} {'MaxRet':>8}")
    for sym_k, sym_rows in sorted(by_sym.items()):
        ss = summarise(sym_rows)
        flag = "✓" if ss["win_rate"] >= 50 else ("?" if ss["win_rate"] >= 35 else "✗")
        print(f"  {sym_k:<8} {ss['signals']:>7}  {ss['win_rate']:>5.1f}%"
              f"  {ss['avg_ret_20']:>+6.2f}%  {ss['avg_max_ret']:>+6.2f}%  {flag}")

df_sum = pd.DataFrame(summary_rows)[["pattern","signals","win_rate",
                                      "avg_ret_5","avg_ret_10","avg_ret_20","avg_max_ret"]]
sum_path = os.path.join(OUT_DIR, "vn30_pullback_summary.csv")
df_sum.to_csv(sum_path, index=False)
print(f"\n  Summary  → {sum_path}")

# ── generate evidence charts (top 3 per pattern by 20d return) ────────────────
print(f"\n  Generating evidence charts…")

for pname, fn, pcol, gcol, params in DETECTORS:
    rows = [r for r in all_details
            if r["pattern"] == pname and r["ret_20"] is not None]
    if not rows:
        continue
    top3 = sorted(rows, key=lambda r: r["ret_20"], reverse=True)[:3]

    for rank, sig in enumerate(top3, 1):
        sym_ticker = sig["symbol"] + ".VN"
        df = fetch(sym_ticker)
        if df is None:
            continue
        safe = pname.replace(" ", "_").lower()
        fname = f"evidence_{safe}_{sig['symbol']}_{rank}.png"
        out   = os.path.join(OUT_DIR, fname)
        draw_evidence(sym_ticker, df, sig, pname, out)
        print(f"    saved → {out}  "
              f"({sig['symbol']} {sig['date']}  20d={sig['ret_20']:+.1f}%)")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}\n")
