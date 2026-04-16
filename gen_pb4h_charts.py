"""Generate 5 PNG evidence charts for 4H Pin Bar scanner."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app import load_price_data_4h, compute_indicators, scan_pinbar

TARGETS = [
    ("ACB.VN",  7,  "ACB - PINBAR_MA50 (MA20+MA50)"),
    ("CTG.VN", 13,  "CTG - PINBAR_MA50"),
    ("NKG.VN", 13,  "NKG - PINBAR_MA50"),
    ("GMD.VN", 20,  "GMD - PINBAR_SWING"),
    ("IDI.VN", 42,  "IDI - PINBAR_MA50 (MA20+MA50)"),
]

for idx, (sym, offset, title) in enumerate(TARGETS, 1):
    print(f"[{idx}/5] {sym} offset={offset}...")
    df_4h = load_price_data_4h(sym)
    if df_4h is None:
        print(f"  SKIP: no data")
        continue
    df_4h = compute_indicators(df_4h)
    sub = df_4h.iloc[:len(df_4h) - offset] if offset > 0 else df_4h
    sig = scan_pinbar(sub, None)
    if not sig:
        print(f"  SKIP: no pin bar")
        continue

    sig_pos = len(sub) - 1
    start = max(0, sig_pos - 30)
    end = min(len(df_4h), sig_pos + 6)
    view = df_4h.iloc[start:end]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.patch.set_facecolor("#0a0e17")
    for ax in (ax1, ax2):
        ax.set_facecolor("#131829")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    x_vals = np.arange(len(view))
    sig_x = sig_pos - start  # signal candle x-position in view

    # Candlesticks
    for i, (dt, row) in enumerate(view.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
        ax1.bar(i, abs(c - o), bottom=min(o, c), width=0.6, color=color, edgecolor=color)

    # MA lines
    for ma_col, ma_color, ma_label in [("ma20", "#f0b90b", "MA20"), ("ma50", "#2196f3", "MA50")]:
        if ma_col in view.columns:
            vals = view[ma_col].values
            mask = ~pd.isna(vals)
            if mask.any():
                ax1.plot(x_vals[mask], vals[mask], color=ma_color, linewidth=1.2,
                         label=ma_label, alpha=0.8)

    # Swing low
    if "swing_low20" in view.columns:
        sw = view["swing_low20"].values
        mask = ~pd.isna(sw)
        if mask.any():
            ax1.plot(x_vals[mask], sw[mask], color="#ff9800", linewidth=1,
                     linestyle="--", label="SwingLow20", alpha=0.6)

    # SL / TP lines
    sl, tp = sig["sl"], sig["tp"]
    ax1.axhline(y=sl, color="#ef5350", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.text(len(view) - 0.5, sl, f" SL {sl:.0f}", color="#ef5350", fontsize=9,
             va="center", fontweight="bold")
    ax1.axhline(y=tp, color="#26a69a", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1.text(len(view) - 0.5, tp, f" TP {tp:.0f}", color="#26a69a", fontsize=9,
             va="center", fontweight="bold")

    # Arrow on signal candle
    if 0 <= sig_x < len(view):
        sig_row = view.iloc[sig_x]
        arrow_y = sig_row["Low"] - (sig_row["High"] - sig_row["Low"]) * 1.5
        ax1.annotate(
            "PIN BAR", xy=(sig_x, sig_row["Low"]),
            xytext=(sig_x, arrow_y),
            color="#f0b90b", fontsize=10, fontweight="bold", ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="#f0b90b", lw=2),
        )

    ax1.legend(loc="upper left", fontsize=8, facecolor="#131829",
               edgecolor="#333", labelcolor="white")

    # Volume
    for i, (dt, row) in enumerate(view.iterrows()):
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax2.bar(i, row["Volume"], width=0.6, color=color, alpha=0.6)
    ax2.set_ylabel("Volume", color="white", fontsize=9)

    # X labels
    labels = [str(dt)[:10] if i % 5 == 0 else "" for i, dt in enumerate(view.index)]
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(labels, rotation=45, fontsize=7, color="white")

    fig.suptitle(
        f"4H Pin Bar #{idx}: {title}\n"
        f"Tier {sig['pin_tier']} | R:R {sig['rr']} | Ctx {sig['context']} | "
        f"Wick {sig['wick_ratio']:.0%} | {sig['vol_tier']} | {str(sig['date'])[:16]}",
        color="white", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    date_str = str(sig["date"])[:10].replace("-", "")
    fname = (f"data/backtest/pb4h_{idx:02d}_{sym.replace('.VN','')}_"
             f"{sig['signal']}_{date_str}.png")
    fig.savefig(fname, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {fname}")

print("Done.")
