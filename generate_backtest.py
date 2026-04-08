"""
Backtest script — Trend Pullback + Breakout (PB_LONG) on VN30 (3 years).
Generates 10 PNG sample charts into data/backtest/.

Run:
    python generate_backtest.py
"""

from __future__ import annotations

import os
import sys
import types as _t

# ── Stub Streamlit + vnstock3 so app.py can be imported without a server ──
_st = _t.ModuleType("streamlit")
_st.cache_data    = lambda **kw: (lambda f: f)
_st.session_state = {}
for _a in [
    "set_page_config", "title", "caption", "info", "warning", "error",
    "subheader", "markdown", "metric", "dataframe", "selectbox", "slider",
    "button", "checkbox", "header", "write", "divider", "tabs", "spinner",
    "columns", "progress", "number_input", "success", "plotly_chart",
    "image", "text_input",
]:
    setattr(_st, _a, lambda *a, **kw: None)
sys.modules["streamlit"] = _st
sys.modules.setdefault("vnstock3", _t.ModuleType("vnstock3"))

import app

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

DATA_PATH    = "./data"
CACHE_DIR    = f"{DATA_PATH}/cache"
BACKTEST_DIR = f"{DATA_PATH}/backtest"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BACKTEST_DIR, exist_ok=True)

BG_COLOR     = "#0a0e17"
UP_COLOR     = "#26a69a"
DOWN_COLOR   = "#ef5350"
MA50_COLOR   = "#f59e0b"
MA200_COLOR  = "#818cf8"
CONSOL_COLOR = "#a78bfa"

VN30_SYMBOLS = [
    "ACB.VN", "BID.VN", "CTG.VN", "HDB.VN", "LPB.VN", "MBB.VN",
    "SHB.VN", "SSB.VN", "STB.VN", "TCB.VN", "TPB.VN", "VCB.VN",
    "VIB.VN", "VPB.VN", "BCM.VN", "KDH.VN", "VHM.VN", "MSN.VN",
    "MWG.VN", "SAB.VN", "VNM.VN", "GAS.VN", "GVR.VN", "HPG.VN",
    "PLX.VN", "POW.VN", "FPT.VN", "BVH.VN", "SSI.VN", "VJC.VN",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, symbol.replace(".", "_") + ".parquet")


def load_data(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            if len(df) >= 100:
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker(symbol).history(period="3y")
        if df is None or df.empty:
            return None
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.to_parquet(path)
        return df
    except Exception:
        return None


# ── Find PB+BO signals across history ────────────────────────────────────────

def find_pb_signals(df: pd.DataFrame) -> list[tuple[int, dict]]:
    """
    Slide scan_pullback_breakout across full history.
    Returns list of (row_index, signal_dict).
    """
    results = []
    min_bars = 55 + 5 + 15   # default consol+pullback windows
    for i in range(min_bars, len(df)):
        sig = app.scan_pullback_breakout(df.iloc[:i + 1])
        if sig:
            results.append((i, sig))
    return results


# ── Trade simulation using signal's own SL / TP ───────────────────────────────

def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    sig: dict,
    max_hold: int = 20,
) -> dict | None:
    if signal_idx + 1 >= len(df):
        return None

    entry_row = df.iloc[signal_idx + 1]
    entry     = entry_row["Open"] * 1.001   # 0.1% slippage
    sl        = sig["sl"]                   # consol_low from scan
    tp        = sig["tp"]

    if entry <= sl or pd.isna(sl) or pd.isna(tp):
        return None

    rr = (tp - entry) / (entry - sl) if entry > sl else 0.0

    result     = "TIMEOUT"
    exit_price = df.iloc[min(signal_idx + max_hold, len(df) - 1)]["Close"]
    exit_idx   = min(signal_idx + max_hold, len(df) - 1)

    for j in range(signal_idx + 1, min(signal_idx + 1 + max_hold, len(df))):
        candle = df.iloc[j]
        if candle["Low"] <= sl:
            result     = "LOSS"
            exit_price = sl
            exit_idx   = j
            break
        if candle["High"] >= tp:
            result     = "WIN"
            exit_price = tp
            exit_idx   = j
            break

    return {
        "signal_type":      sig["signal"],
        "signal_idx":       signal_idx,
        "entry":            entry,
        "sl":               sl,
        "tp":               tp,
        "rr":               round(rr, 2),
        "result":           result,
        "exit_idx":         exit_idx,
        "exit_price":       exit_price,
        "pnl_pct":          (exit_price - entry) / entry * 100,
        "days_held":        exit_idx - signal_idx,
        "consol_high":      sig.get("consol_high"),
        "consol_low":       sig.get("consol_low"),
        "pullback_pct":     sig.get("pullback_pct"),
        "consol_range_pct": sig.get("consol_range_pct"),
    }


# ── Chart drawing ─────────────────────────────────────────────────────────────

def draw_chart(df: pd.DataFrame, trade: dict, symbol: str, out_path: str) -> None:
    si    = trade["signal_idx"]
    start = max(0, si - 40)
    end   = min(len(df), trade["exit_idx"] + 6)
    view  = df.iloc[start:end].copy()
    n     = len(view)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        ax.grid(True, color="#1e293b", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
    fig.patch.set_facecolor(BG_COLOR)

    # Candlesticks
    for i, (_, row) in enumerate(view.iterrows()):
        color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
        ax1.bar(i, abs(row["Close"] - row["Open"]),
                bottom=min(row["Open"], row["Close"]),
                color=color, width=0.6, linewidth=0)
        ax1.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8)

    # MAs
    x_range = range(n)
    if "ma50" in view.columns and not view["ma50"].isna().all():
        ax1.plot(x_range, view["ma50"],  color=MA50_COLOR,  linewidth=1.0, label="MA50")
    if "ma200" in view.columns and not view["ma200"].isna().all():
        ax1.plot(x_range, view["ma200"], color=MA200_COLOR, linewidth=1.0, label="MA200")

    # Consolidation zone
    if trade.get("consol_high"):
        ax1.axhline(trade["consol_high"], color=CONSOL_COLOR, linewidth=1.2,
                    linestyle="--", alpha=0.85, label=f"Consol H {trade['consol_high']:.1f}")
    if trade.get("consol_low"):
        ax1.axhline(trade["consol_low"], color=CONSOL_COLOR, linewidth=1.0,
                    linestyle=":", alpha=0.75, label=f"Consol L {trade['consol_low']:.1f}")

    # Locate signal / exit bars in view slice
    view_index = list(view.index)
    df_index   = list(df.index)

    def to_view_x(df_i: int) -> int | None:
        if 0 <= df_i < len(df_index):
            try:
                return view_index.index(df_index[df_i])
            except ValueError:
                return None
        return None

    sig_x  = to_view_x(si)
    exit_x = to_view_x(trade["exit_idx"])

    # Signal marker
    if sig_x is not None and sig_x < n:
        ax1.axvline(sig_x, color="#fbbf24", linewidth=1.5, linestyle="--", alpha=0.7)
        ax1.scatter(sig_x, view.iloc[sig_x]["High"] * 1.01,
                    marker="^", color="#fbbf24", s=120, zorder=5)

    # SL / TP
    ax1.axhline(trade["sl"], color=DOWN_COLOR, linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"SL {trade['sl']:.1f}")
    ax1.axhline(trade["tp"], color=UP_COLOR,   linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"TP {trade['tp']:.1f}")

    # Exit marker
    if exit_x is not None and exit_x < n:
        r_color = UP_COLOR if trade["result"] == "WIN" else (
            DOWN_COLOR if trade["result"] == "LOSS" else "#fbbf24")
        ax1.axvline(exit_x, color=r_color, linewidth=1.5, linestyle=":", alpha=0.8)
        ax1.scatter(exit_x, trade["exit_price"], marker="o", color=r_color, s=100, zorder=5)

    ax1.set_ylabel("Price", color="#94a3b8", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7,
               facecolor=BG_COLOR, edgecolor="#334155", labelcolor="#e2e8f0")

    # Volume
    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                  for c, o in zip(view["Close"], view["Open"])]
    ax2.bar(x_range, view["Volume"], color=vol_colors, alpha=0.7)
    if "avg_vol20" in view.columns and not view["avg_vol20"].isna().all():
        ax2.plot(x_range, view["avg_vol20"], color="#fbbf24", linewidth=0.8, label="AvgVol20")
    ax2.set_ylabel("Volume", color="#94a3b8", fontsize=8)

    # X-axis
    tick_step = max(1, n // 8)
    ticks     = list(range(0, n, tick_step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [view.index[i].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=7, color="#94a3b8",
    )

    # Title
    r_label  = {"WIN": "[WIN]", "LOSS": "[LOSS]", "TIMEOUT": "[TIMEOUT]"}.get(trade["result"], "")
    sig_date = df.index[si].strftime("%Y-%m-%d")
    pb_pct   = f"PB {trade['pullback_pct']:.1f}%" if trade.get("pullback_pct") else ""
    cr_pct   = f"Consol {trade['consol_range_pct']:.1f}%" if trade.get("consol_range_pct") else ""
    title = (
        f"{symbol} · PB_LONG · {sig_date}   "
        f"{r_label} {trade['result']}   "
        f"PnL {trade['pnl_pct']:+.1f}%   "
        f"R:R {trade['rr']:.1f}   "
        f"Hold {trade['days_held']}d   {pb_pct}  {cr_pct}"
    )
    fig.suptitle(title, color="#e2e8f0", fontsize=9, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BACKTEST — Trend Pullback + Breakout (PB_LONG) on VN30 (3yr)")
    print("=" * 60)

    all_trades: list[dict] = []

    for sym in VN30_SYMBOLS:
        print(f"  {sym:<12}", end=" ")
        df = load_data(sym)
        if df is None or len(df) < 100:
            print("skip (no data)")
            continue

        df = app.compute_indicators(df)
        pb_list = find_pb_signals(df)

        sym_trades: list[dict] = []
        prev_sig  = -999

        for idx, sig in sorted(pb_list, key=lambda x: x[0]):
            if idx - prev_sig < 5:   # 5-bar cooldown per symbol
                continue
            trade = simulate_trade(df, idx, sig)
            if trade and trade["rr"] >= 0.8:
                trade["symbol"] = sym.replace(".VN", "")
                trade["df"]     = df
                sym_trades.append(trade)
                all_trades.append(trade)
                prev_sig = idx

        wins = sum(1 for t in sym_trades if t["result"] == "WIN")
        print(f"{len(sym_trades):2d} PB trades  win {wins}/{len(sym_trades)}")

    if not all_trades:
        print("\nKhong co tin hieu. Kiem tra lai du lieu.")
        sys.exit(1)

    # ── Summary stats ─────────────────────────────────────────────
    total  = len(all_trades)
    n_win  = sum(1 for t in all_trades if t["result"] == "WIN")
    n_loss = sum(1 for t in all_trades if t["result"] == "LOSS")
    n_to   = total - n_win - n_loss
    pf_w   = sum(t["pnl_pct"] for t in all_trades if t["result"] == "WIN")
    pf_l   = abs(sum(t["pnl_pct"] for t in all_trades if t["result"] == "LOSS"))

    print("\n" + "=" * 60)
    print(f"TONG KET PB+BO  ({total} trades hop le, RR >= 0.8)")
    print(f"  Win rate      : {n_win/total*100:.1f}%  ({n_win}W / {n_loss}L / {n_to}TO)")
    print(f"  Profit Factor : {pf_w/max(pf_l, 0.01):.2f}")
    print(f"  Avg PnL WIN   : {pf_w/max(n_win,1):.1f}%")
    print(f"  Avg PnL LOSS  : {pf_l/max(n_loss,1):.1f}%")
    print(f"  Avg R:R       : {sum(t['rr'] for t in all_trades)/total:.2f}")
    print("=" * 60)

    # ── Select 10 balanced charts ──────────────────────────────────
    import random
    random.seed(42)

    wins_list    = [t for t in all_trades if t["result"] == "WIN"]
    losses_list  = [t for t in all_trades if t["result"] == "LOSS"]
    timeout_list = [t for t in all_trades if t["result"] == "TIMEOUT"]

    n_target = 10
    n_wins   = min(5, len(wins_list))
    n_losses = min(3, len(losses_list))
    n_tos    = min(n_target - n_wins - n_losses, len(timeout_list))

    selected  = random.sample(wins_list,    n_wins)
    selected += random.sample(losses_list,  n_losses)
    selected += random.sample(timeout_list, n_tos)

    remaining = n_target - len(selected)
    if remaining > 0:
        pool = [t for t in all_trades if t not in selected]
        selected += random.sample(pool, min(remaining, len(pool)))

    random.shuffle(selected)

    # ── Clear old PNGs ─────────────────────────────────────────────
    removed = 0
    for f in os.listdir(BACKTEST_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(BACKTEST_DIR, f))
            removed += 1
    if removed:
        print(f"\nXoa {removed} chart cu")

    # ── Draw charts ────────────────────────────────────────────────
    print(f"\nVe {len(selected)} charts -> {BACKTEST_DIR}/\n")
    for i, trade in enumerate(selected, 1):
        sym      = trade["symbol"]
        result   = trade["result"][:2]
        date_str = trade["df"].index[trade["signal_idx"]].strftime("%Y%m%d")
        pnl      = f"{trade['pnl_pct']:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        filename = f"{i:02d}_{sym}_PB_{result}_{date_str}_{pnl}pct.png"
        out_path = os.path.join(BACKTEST_DIR, filename)

        draw_chart(trade["df"], trade, sym, out_path)
        print(f"  [{i:02d}] {filename}")

    print(f"\nHoan tat. {len(selected)} charts trong {BACKTEST_DIR}/")


if __name__ == "__main__":
    main()
