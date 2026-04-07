"""
Generate 3 sample Plotly PNG charts for every active pattern type
across VN30 + VNMID (best score first).

Patterns: VCP, Flat Base, Pullback MA20, Pennant Pullback, Triangle Pullback
Output: data/charts/sample_{pattern}_{rank}_{sym}.png
"""
import sys, types, os, warnings
warnings.filterwarnings("ignore")

# ── stub streamlit ────────────────────────────────────────────────────────────
class _CM:
    def __init__(self,*a,**kw): pass
    def __enter__(self): return self
    def __exit__(self,*a): return False
    def __getattr__(self,_): return lambda *a,**kw:_CM()

_st = types.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)
_st.session_state = {}
_st.button        = lambda *a,**kw: False
_st.radio         = lambda *a,**kw: "VN30"
_st.checkbox      = lambda *a,**kw: True
_st.columns       = lambda n,**kw: [_CM() for _ in range(n if isinstance(n,int) else len(n))]
_st.sidebar       = _CM()
_st.spinner       = _CM
_st.expander      = _CM
_st.progress      = lambda *a,**kw: _CM()
for _a in ["set_page_config","title","caption","info","warning","error",
           "subheader","markdown","metric","dataframe","line_chart","plotly_chart",
           "selectbox","slider","download_button","stop","rerun","header","write"]:
    setattr(_st, _a, lambda *a,**kw: None)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import app

OUT_DIR = "./data/charts"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES_PER_PATTERN = 3
ALL_SYMS = list(app.VN30_STOCKS) + list(app.VNMID_STOCKS)

# call each pattern checker directly (bypasses market/RS filters for sample generation)
CHECKERS = {
    "VCP":             app._check_vcp,
    "Flat Base":       app._check_flat_base,
    "Pullback MA20":   app._check_pullback_ma20,
    "Pennant Pullback":app._check_cp_pennant,
    "Triangle Pullback":app._check_cp_triangle,
}

# ── scan all symbols ──────────────────────────────────────────────────────────
print(f"\nScanning {len(ALL_SYMS)} symbols (VN30 + VNMID)...\n")

by_pattern: dict[str, list] = {}   # pattern_name → [(score, sym, df, pattern_dict)]

for sym in ALL_SYMS:
    ticker = sym.replace(".VN","")
    try:
        df_full = yf.Ticker(sym).history(period="2y")
        if df_full is None or df_full.empty or len(df_full) < 100:
            continue
        if df_full.index.tz:
            df_full.index = df_full.index.tz_localize(None)
    except Exception:
        continue

    # scan multiple historical cutpoints: current + 30/60/90/120 days ago
    cutpoints = [len(df_full)] + [max(100, len(df_full)-n)
                                   for n in [30, 60, 90, 120, 180]
                                   if len(df_full) - n >= 100]

    for cut in cutpoints:
        df_c = df_full.iloc[:cut].copy()
        df_c["MA20"]  = df_c["Close"].rolling(20).mean()
        df_c["MA50"]  = df_c["Close"].rolling(50).mean()
        df_c["MA150"] = df_c["Close"].rolling(150).mean()

        grade = app._trend_grade(df_c)
        for pname, checker in CHECKERS.items():
            if pname in by_pattern and len(by_pattern[pname]) >= 6:
                continue   # already have enough candidates
            p = checker(df_c)
            if p is None:
                continue
            p["trend_grade"] = grade
            score = p.get("score", 0)
            # store the df sliced to the cutpoint for charting
            by_pattern.setdefault(pname, []).append((score, sym, df_c, p))
            print(f"  {ticker:<6}  {pname:<22}  score={score}"
                  f"  {p.get('quality','')}  {grade}  (cut={cut})")

print(f"\nTotal patterns: {sum(len(v) for v in by_pattern.values())}\n")

# ── Plotly chart generator ────────────────────────────────────────────────────
def make_plotly_png(sym: str, df_c: pd.DataFrame, p: dict, out_path: str) -> None:
    df_chart = df_c.iloc[-252:].copy()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"],  close=df_chart["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))
    for col, color in [("MA20","#60a5fa"),("MA50","#34d399"),("MA150","#f59e0b")]:
        if col in df_chart.columns and not df_chart[col].isna().all():
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[col],
                mode="lines", name=col, line=dict(color=color, width=1.2),
            ))

    pivot    = p.get("pivot")
    stoploss = p.get("stoploss")
    tl       = p.get("_tl")
    wb       = p.get("_window_bars", 30)

    x0_zone = tl["x0"]      if tl else df_chart.index[-min(wb, len(df_chart))]
    x1_zone = tl["zone_x1"] if tl else df_chart.index[-1]
    fig.add_vrect(x0=x0_zone, x1=x1_zone,
                  fillcolor="rgba(108,156,255,0.09)", layer="below", line_width=0)

    if pivot:
        fig.add_hline(y=pivot, line_dash="dash", line_color="#ff4d94", line_width=1.2,
                      opacity=0.85, annotation_text=f"Pivot {pivot:.1f}",
                      annotation_font_color="#ff4d94", annotation_font_size=10,
                      annotation_position="right")
    if stoploss:
        fig.add_hline(y=stoploss, line_dash="dot", line_color="#ff9900", line_width=1.1,
                      opacity=0.75, annotation_text=f"SL {stoploss:.1f}",
                      annotation_font_color="#ff9900", annotation_font_size=10,
                      annotation_position="right")
    if tl:
        hv = [tl["high_y0"], tl["high_y1"]]
        lv = [tl["low_y0"],  tl["low_y1"]]
        if not any(v != v for v in hv):
            fig.add_shape(type="line",
                          x0=tl["x0"], y0=tl["high_y0"],
                          x1=tl["x1"], y1=tl["high_y1"],
                          line=dict(color="#00e5ff", width=1.4), layer="above")
        if not any(v != v for v in lv):
            fig.add_shape(type="line",
                          x0=tl["x0"], y0=tl["low_y0"],
                          x1=tl["x1"], y1=tl["low_y1"],
                          line=dict(color="#ff6b35", width=1.4), layer="above")

    ticker  = sym.replace(".VN","")
    sector  = app.VN30_STOCKS.get(sym, app.VNMID_STOCKS.get(sym, ""))
    score   = p.get("score", 0)
    entry_ok = "✓ entry zone" if p.get("entry_ok") else ""

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=480, width=1200,
        margin=dict(l=0, r=80, t=35, b=0),
        paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e2535"),
        yaxis=dict(gridcolor="#1e2535"),
        title=dict(
            text=(f"{ticker}  [{sector}]  ·  {p['pattern']}  "
                  f"·  {p.get('quality','')}  ·  score {score}  "
                  f"·  {p.get('status','')}  {entry_ok}  "
                  f"·  {p.get('notes','')}"),
            font=dict(size=11, color="#e2e8f0"), x=0.01),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.write_image(out_path, scale=2)


# ── generate top-3 per pattern ────────────────────────────────────────────────
for pname, entries in sorted(by_pattern.items()):
    entries.sort(key=lambda x: x[0], reverse=True)   # best score first
    top3 = entries[:SAMPLES_PER_PATTERN]
    print(f"\n{pname} — {len(entries)} signals found, generating top {len(top3)}:")
    for rank, (score, sym, df_c, p) in enumerate(top3, 1):
        safe_p = pname.replace(" ","_")
        safe_s = sym.replace(".VN","")
        fname  = f"sample_{safe_p}_{rank}_{safe_s}.png"
        out    = os.path.join(OUT_DIR, fname)
        make_plotly_png(sym, df_c, p, out)
        print(f"  [{rank}] {safe_s:<6}  score={score}  → {out}")

print(f"\nDone. Charts saved to {OUT_DIR}/\n")
