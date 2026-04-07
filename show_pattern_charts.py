"""
show_pattern_charts.py — Scan VN30 stocks, detect patterns, generate candlestick charts.
Run: python show_pattern_charts.py
Saves: ./data/charts/vn30_pattern_<ticker>.png
"""

import sys, types, os, warnings
warnings.filterwarnings("ignore")

# ── stub streamlit ────────────────────────────────────────────────────
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
_st.columns       = lambda n, **kw: [_CM() for _ in range(n if isinstance(n,int) else len(n))]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import app

OUT_DIR = "./data/charts"
os.makedirs(OUT_DIR, exist_ok=True)

BG     = "#0a0e17"
CARD   = "#131829"
BORDER = "#2a3050"
TEXT   = "#e2e8f0"
MUTED  = "#8892a8"

MA_COLORS = {
    "MA20":  "#60a5fa",
    "MA50":  "#34d399",
    "MA150": "#f59e0b",
    "MA200": "#a78bfa",
}

STATUS_COLOR = {
    "🔥 BUY":        "#ef4444",
    "👀 Monitoring": "#f59e0b",
    "⏳ Setup":       "#6c9cff",
}


# ── fetch ─────────────────────────────────────────────────────────────
def fetch(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if df is None or df.empty or len(df) < 60:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception:
        return None


# ── candlestick drawing ───────────────────────────────────────────────
def draw_candle(ax, x, o, h, l, c, width=0.6):
    col = "#34d399" if c >= o else "#ef4444"
    ax.plot([x, x], [l, h], color=col, lw=0.8, zorder=2)
    body_y = min(o, c)
    body_h = max(abs(c - o), 0.0001)
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, body_y), width, body_h,
        boxstyle="square,pad=0", facecolor=col, edgecolor=col, lw=0, zorder=3,
    )
    ax.add_patch(rect)


def make_chart(sym: str, df: pd.DataFrame, pattern: dict) -> str:
    """Draw candlestick chart for one detected pattern. Returns saved path."""
    n_back = 70
    view   = df.iloc[-(n_back + 5):].copy()
    n      = len(view)

    close  = view["Close"]
    ma20   = close.rolling(20, min_periods=5).mean()
    ma50   = close.rolling(50, min_periods=20).mean()
    ma150  = close.rolling(150, min_periods=50).mean()
    ma200  = close.rolling(200, min_periods=80).mean()

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6.5), facecolor=BG)
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1], hspace=0.04)
    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    for a in [ax, axv]:
        a.set_facecolor(CARD)
        a.tick_params(colors=MUTED, labelsize=7)
        for s in a.spines.values():
            s.set_edgecolor(BORDER)

    xs = list(range(n))

    # ── candlesticks ──────────────────────────────────────────────────
    for i in xs:
        draw_candle(ax, i,
                    float(view["Open"].iloc[i]),
                    float(view["High"].iloc[i]),
                    float(view["Low"].iloc[i]),
                    float(view["Close"].iloc[i]))

    # ── MA lines ──────────────────────────────────────────────────────
    for ma_s, label in [(ma20,"MA20"),(ma50,"MA50"),(ma150,"MA150"),(ma200,"MA200")]:
        vals = [(i, float(v)) for i, v in enumerate(ma_s) if not pd.isna(v)]
        if vals:
            xi, yi = zip(*vals)
            ax.plot(xi, yi, color=MA_COLORS[label], lw=0.9, alpha=0.85,
                    linestyle="--" if label in ("MA150","MA200") else "-",
                    label=label, zorder=2)

    # ── pivot & stoploss lines ────────────────────────────────────────
    pivot    = pattern["pivot"]
    stoploss = pattern["stoploss"]
    ax.axhline(pivot,    color="#ef4444", lw=1.1, linestyle="--", alpha=0.9,
               label=f"Pivot {pivot:.1f}", zorder=4)
    ax.axhline(stoploss, color="#f59e0b", lw=0.9, linestyle=":",  alpha=0.8,
               label=f"SL {stoploss:.1f}")

    # ── pattern window shading ────────────────────────────────────────
    win_bars = 60 if pattern["pattern"] == "VCP" else \
               (40 if pattern["pattern"] == "Flat Base" else 20)
    ax.axvspan(max(0, n - win_bars - 1), n - 1, color="#6c9cff18",
               label="Pattern window")

    # ── current price marker ──────────────────────────────────────────
    cur_price = float(close.iloc[-1])
    cur_x     = n - 1
    status    = pattern.get("status", "⏳ Setup")
    mc        = STATUS_COLOR.get(status, "#6c9cff")
    ax.scatter([cur_x], [cur_price], s=110, color=mc, marker="^",
               zorder=6, label=f"Now {cur_price:.1f}")

    # ── entry candle annotation ───────────────────────────────────────
    entry_c = pattern.get("entry_candle", "None")
    if entry_c and entry_c not in ("None", ""):
        ax.annotate(
            f" {entry_c}",
            xy=(cur_x, cur_price),
            xytext=(max(0, cur_x - 6), cur_price * 1.018),
            fontsize=7.5, color=mc,
            arrowprops=dict(arrowstyle="->", color=mc, lw=0.8),
        )

    # ── title ─────────────────────────────────────────────────────────
    ticker  = sym.replace(".VN", "")
    sector  = app.VN30_STOCKS.get(sym, "")
    quality = pattern.get("quality", "★")
    # MA alignment check
    p_val   = float(close.iloc[-1])
    ma50_v  = float(close.rolling(50).mean().iloc[-1])
    ma150_v = float(close.rolling(150).mean().iloc[-1])
    ma200_v = float(close.rolling(200).mean().iloc[-1])
    trend_ok = p_val > ma50_v > ma150_v > ma200_v
    trend_lbl = "✅ Trend OK" if trend_ok else "⚠ Trend weak"
    ax.set_title(
        f"{ticker}  [{sector}]  ·  {pattern['pattern']}  "
        f"·  {quality}  ·  {status}  ·  {trend_lbl}  "
        f"·  Entry candle: {entry_c}  ·  Pivot: {pivot:.1f}  ·  SL: {stoploss:.1f}",
        color=TEXT, fontsize=8, pad=6, loc="left",
    )

    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.3,
              labelcolor="white", facecolor=CARD, ncol=4)
    ax.set_xlim(-1, n + 2)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ── volume ────────────────────────────────────────────────────────
    vol_colors = [
        "#ef444466" if float(view["Close"].iloc[i]) < float(view["Open"].iloc[i])
        else "#34d39966"
        for i in xs
    ]
    axv.bar(xs, view["Volume"], color=vol_colors, width=0.8)
    vol_ma20 = view["Volume"].rolling(20, min_periods=5).mean()
    axv.plot(xs, vol_ma20, color="#60a5fa", lw=0.8, alpha=0.8)
    axv.set_yticks([])

    step = max(1, n // 6)
    axv.set_xticks(range(0, n, step))
    axv.set_xticklabels(
        [str(view.index[i])[:10] for i in range(0, n, step)],
        rotation=25, ha="right", fontsize=6, color=MUTED,
    )

    plt.tight_layout(pad=0.5)
    safe = sym.replace(".", "_")
    out  = os.path.join(OUT_DIR, f"vn30_{safe}_{pattern['pattern'].replace(' ','_')}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    return out


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("\nScanning VN30 for patterns...\n")
    results = []

    for sym in app.VN30_STOCKS:
        print(f"  {sym}...", end=" ", flush=True)
        df = fetch(sym)
        if df is None:
            print("no data")
            continue
        # Run all pattern checks (including new flag/pennant/triangle pullback)
        found = app.detect_patterns(df, require_trend=False)
        if found:
            for p in found:
                results.append((sym, df, p))
            print(f"  → {[p['pattern'] for p in found]}")
        else:
            print("no pattern")

    if not results:
        print("\nNo patterns found in VN30. Market may be in downtrend.\n")
        return

    # Sort: BUY first, then Monitoring, then Setup; then by quality
    _s_ord = {"🔥 BUY": 0, "👀 Monitoring": 1, "⏳ Setup": 2}
    _q_ord = {"★★★": 0, "★★": 1, "★": 2}
    results.sort(key=lambda x: (
        _s_ord.get(x[2].get("status", "⏳ Setup"), 3),
        _q_ord.get(x[2].get("quality", "★"), 3),
    ))

    print(f"\nFound {len(results)} pattern(s) across {len({r[0] for r in results})} stocks.\n")
    print(f"  {'Sym':<10} {'Pattern':<16} {'Status':<16} {'Candle':<20} {'Quality'}")
    print(f"  {'-'*75}")
    for sym, _, p in results:
        print(f"  {sym.replace('.VN',''):<10} {p['pattern']:<16} "
              f"{p.get('status',''):<16} {p.get('entry_candle',''):<20} {p.get('quality','')}")

    # Generate charts (top 6 or all if fewer)
    print(f"\nGenerating charts for top {min(6, len(results))} patterns...")
    for sym, df, p in results[:6]:
        out = make_chart(sym, df, p)
        print(f"  saved → {out}")

    print()


if __name__ == "__main__":
    main()
