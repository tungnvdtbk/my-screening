"""Generate sample Plotly PNG charts showing the new trendline overlay."""
import sys, types, os

class _CM:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _): return lambda *a, **kw: _CM()

_st = types.ModuleType("streamlit")
_st.cache_data = lambda **kw: (lambda f: f)
_st.session_state = {}
_st.button   = lambda *a, **kw: False
_st.radio    = lambda *a, **kw: "VN30"
_st.checkbox = lambda *a, **kw: True
_st.columns  = lambda n, **kw: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
_st.sidebar  = _CM()
_st.spinner  = _CM
_st.expander = _CM
_st.progress = lambda *a, **kw: _CM()
for _a in ["set_page_config","title","caption","info","warning","error",
           "subheader","markdown","metric","dataframe","line_chart","plotly_chart",
           "selectbox","slider","download_button","stop","rerun","header","write"]:
    setattr(_st, _a, lambda *a, **kw: None)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", types.ModuleType("vnstock3"))

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import app

os.makedirs("./data/charts", exist_ok=True)

TARGETS = [
    ("STB", "Flag Pullback"),
    ("VNM", "Triangle Pullback"),
    ("SSI", "Pennant Pullback"),
]

for sym, pattern_name in TARGETS:
    ticker = sym + ".VN"
    df = yf.Ticker(ticker).history(period="1y")
    if df is None or df.empty:
        print(f"  {sym}: no data")
        continue
    if df.index.tz:
        df.index = df.index.tz_localize(None)

    df_c = df.copy()
    df_c["MA20"]  = df_c["Close"].rolling(20).mean()
    df_c["MA50"]  = df_c["Close"].rolling(50).mean()
    df_c["MA150"] = df_c["Close"].rolling(150).mean()
    df_chart = df_c.iloc[-252:].copy()

    live_p = app.detect_patterns(df_c, require_trend=False)
    patterns = [p for p in live_p if p["pattern"] == pattern_name]
    if not patterns:
        print(f"  {sym}: no {pattern_name} today")
        continue

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"],  close=df_chart["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))
    for col, color in [("MA20", "#60a5fa"), ("MA50", "#34d399"), ("MA150", "#f59e0b")]:
        if col in df_chart.columns and not df_chart[col].isna().all():
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[col],
                mode="lines", name=col, line=dict(color=color, width=1.2),
            ))

    for p in patterns:
        pivot    = p.get("pivot")
        stoploss = p.get("stoploss")
        tl       = p.get("_tl")
        wb       = p.get("_window_bars", 30)

        x0_zone = tl["x0"]      if tl else df_chart.index[-min(wb, len(df_chart))]
        x1_zone = tl["zone_x1"] if tl else df_chart.index[-1]
        fig.add_vrect(x0=x0_zone, x1=x1_zone,
                      fillcolor="rgba(108,156,255,0.09)", layer="below", line_width=0)

        if pivot:
            fig.add_hline(y=pivot, line_dash="dash", line_color="#ff4d94", line_width=1.8,
                          annotation_text=f"Pivot {pivot:.1f}",
                          annotation_font_color="#ff4d94", annotation_font_size=10,
                          annotation_position="right")
        if stoploss:
            fig.add_hline(y=stoploss, line_dash="dot", line_color="#ff9900", line_width=1.5,
                          annotation_text=f"SL {stoploss:.1f}",
                          annotation_font_color="#ff9900", annotation_font_size=10,
                          annotation_position="right")
        if tl:
            hv = [tl["high_y0"], tl["high_y1"]]
            lv = [tl["low_y0"],  tl["low_y1"]]
            if not any(v != v for v in hv):
                fig.add_shape(type="line",
                              x0=tl["x0"], y0=tl["high_y0"],
                              x1=tl["x1"], y1=tl["high_y1"],
                              line=dict(color="#00e5ff", width=2.5), layer="above")
            if not any(v != v for v in lv):
                fig.add_shape(type="line",
                              x0=tl["x0"], y0=tl["low_y0"],
                              x1=tl["x1"], y1=tl["low_y1"],
                              line=dict(color="#ff6b35", width=2.5), layer="above")

    safe = pattern_name.replace(" ", "_")
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=480, width=1200,
        margin=dict(l=0, r=80, t=35, b=0),
        paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e2535"),
        yaxis=dict(gridcolor="#1e2535"),
        title=dict(text=f"{sym}  —  {pattern_name}  |  "
                        f"Q:{patterns[0].get('quality','')}  "
                        f"{patterns[0].get('status','')}",
                   font=dict(size=13, color="#e2e8f0"), x=0.01),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out = f"./data/charts/plotly_{sym}_{safe}.png"
    fig.write_image(out, scale=2)
    print(f"  saved → {out}")
