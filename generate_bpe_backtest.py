"""
Backtest + 10 PNG samples for the BPE scanner
(spec: watchlist_breakout_pullback_scanner.md).

Run:
    python generate_bpe_backtest.py
"""

from __future__ import annotations

import os
import sys
import types as _t

# ── Stub Streamlit so app.py imports without a server ──────────────────
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

import app

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH    = "./data"
CACHE_DIR    = f"{DATA_PATH}/cache"
BACKTEST_DIR = f"{DATA_PATH}/backtest"
os.makedirs(BACKTEST_DIR, exist_ok=True)

BG_COLOR    = "#0a0e17"
UP_COLOR    = "#26a69a"
DOWN_COLOR  = "#ef5350"
MA20_COLOR  = "#60a5fa"
MA50_COLOR  = "#f59e0b"
MA200_COLOR = "#a78bfa"
BO_COLOR    = "#fbbf24"

MAX_HOLD   = 20
MIN_BARS   = 250
COOLDOWN   = 10


def load_cached(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(CACHE_DIR, symbol.replace(".", "_") + ".parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df if len(df) >= MIN_BARS else None
    except Exception:
        return None


def find_bpe_signals(df: pd.DataFrame) -> list[tuple[int, dict]]:
    """Walk-forward: at each bar i call scan_bpe on df up to bar i."""
    out: list[tuple[int, dict]] = []
    for i in range(MIN_BARS, len(df)):
        sig = app.scan_bpe(df.iloc[:i + 1])
        if sig:
            out.append((i, sig))
    return out


def simulate(df: pd.DataFrame, idx: int, sig: dict,
             max_hold: int = MAX_HOLD) -> dict | None:
    if idx + 1 >= len(df):
        return None
    entry_row = df.iloc[idx + 1]
    entry = float(entry_row["Open"]) * 1.001
    sl    = float(sig["sl"])
    tp    = float(sig["tp"])
    if entry <= sl:
        return None
    rr_planned = (tp - entry) / (entry - sl) if entry > sl else 0.0

    result     = "TIMEOUT"
    exit_price = float(df.iloc[min(idx + max_hold, len(df) - 1)]["Close"])
    exit_idx   = min(idx + max_hold, len(df) - 1)
    for j in range(idx + 1, min(idx + 1 + max_hold, len(df))):
        c = df.iloc[j]
        if c["Low"] <= sl:
            result, exit_price, exit_idx = "LOSS", sl, j
            break
        if c["High"] >= tp:
            result, exit_price, exit_idx = "WIN", tp, j
            break

    # Map d1/d2 timestamps back to integer indices in df
    d1_ts = sig["d1_date"]
    d2_ts = sig["d2_date"]
    try:
        d1_idx = df.index.get_loc(d1_ts)
        d2_idx = df.index.get_loc(d2_ts)
    except KeyError:
        d1_idx = d2_idx = None

    return {
        "signal_idx":    idx,
        "entry":         entry,
        "sl":            sl,
        "tp":            tp,
        "rr":            round(rr_planned, 2),
        "result":        result,
        "exit_idx":      exit_idx,
        "exit_price":    exit_price,
        "pnl_pct":       (exit_price - entry) / entry * 100.0,
        "days_held":     exit_idx - idx,
        "tier":          sig.get("bpe_tier"),
        "bo2":           sig.get("breakout_level_d2"),
        "bo1":           sig.get("breakout_level_d1"),
        "high_d2":       sig.get("high_d2"),
        "d1_idx":        d1_idx,
        "d2_idx":        d2_idx,
        "gap_d2_d1":     sig.get("gap_d2_d1"),
        "gap_t_d2":      sig.get("gap_t_d2"),
        "ext_vs_bo2":    sig.get("ext_vs_bo2"),
        "pullback_vs_high_d2": sig.get("pullback_vs_high_d2"),
        "ma200_slope20": sig.get("ma200_slope20"),
    }


def draw(df: pd.DataFrame, trade: dict, symbol: str, out_path: str) -> None:
    si     = trade["signal_idx"]
    start  = max(0, si - 60)
    end    = min(len(df), trade["exit_idx"] + 8)
    view   = df.iloc[start:end].copy()
    n      = len(view)

    view["_ma20"]  = df["Close"].rolling(20).mean().iloc[start:end].values
    view["_ma50"]  = df["Close"].rolling(50).mean().iloc[start:end].values
    view["_ma200"] = df["Close"].rolling(200).mean().iloc[start:end].values

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

    for i, (_, row) in enumerate(view.iterrows()):
        color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
        ax1.bar(i, abs(row["Close"] - row["Open"]),
                bottom=min(row["Open"], row["Close"]),
                color=color, width=0.6, linewidth=0)
        ax1.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8)

    x = range(n)
    if not view["_ma20"].isna().all():
        ax1.plot(x, view["_ma20"],  color=MA20_COLOR,  linewidth=1.0, label="MA20")
    if not view["_ma50"].isna().all():
        ax1.plot(x, view["_ma50"],  color=MA50_COLOR,  linewidth=1.0, label="MA50")
    if not view["_ma200"].isna().all():
        ax1.plot(x, view["_ma200"], color=MA200_COLOR, linewidth=1.2, label="MA200")

    view_dates = list(view.index)
    df_dates   = list(df.index)

    def to_x(i_full):
        if i_full is None or not (0 <= i_full < len(df_dates)):
            return None
        try:
            return view_dates.index(df_dates[i_full])
        except ValueError:
            return None

    d1_x   = to_x(trade.get("d1_idx"))
    d2_x   = to_x(trade.get("d2_idx"))
    sig_x  = to_x(si)
    exit_x = to_x(trade["exit_idx"])

    if d1_x is not None:
        ax1.axvline(d1_x, color=BO_COLOR, linewidth=1.0, linestyle=":",
                    alpha=0.6, label="d1")
    if d2_x is not None:
        ax1.axvline(d2_x, color=BO_COLOR, linewidth=1.5, linestyle="--",
                    alpha=0.9, label="d2")
    if sig_x is not None:
        ax1.scatter(sig_x, view.iloc[sig_x]["High"] * 1.01,
                    marker="^", color="#34d399", s=130, zorder=5, label="signal")

    if trade.get("bo2") is not None:
        ax1.axhline(float(trade["bo2"]), color=BO_COLOR, linewidth=1.0,
                    linestyle="-", alpha=0.6, label=f"bo2 {trade['bo2']:.1f}")
    ax1.axhline(trade["sl"], color=DOWN_COLOR, linewidth=1.2, linestyle="--",
                alpha=0.9, label=f"SL {trade['sl']:.1f}")
    ax1.axhline(trade["tp"], color=UP_COLOR,   linewidth=1.2, linestyle="--",
                alpha=0.9, label=f"TP {trade['tp']:.1f}")

    if exit_x is not None:
        c = UP_COLOR if trade["result"] == "WIN" else (
            DOWN_COLOR if trade["result"] == "LOSS" else "#fbbf24")
        ax1.axvline(exit_x, color=c, linewidth=1.5, linestyle=":", alpha=0.8)
        ax1.scatter(exit_x, trade["exit_price"], marker="o", color=c, s=100, zorder=5)

    ax1.set_ylabel("Price", color="#94a3b8", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7, facecolor=BG_COLOR,
               edgecolor="#334155", labelcolor="#e2e8f0", ncol=2)

    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                  for c, o in zip(view["Close"], view["Open"])]
    ax2.bar(x, view["Volume"], color=vol_colors, alpha=0.7)
    ax2.set_ylabel("Volume", color="#94a3b8", fontsize=8)

    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [view.index[i].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=7, color="#94a3b8",
    )

    sig_date = df.index[si].strftime("%Y-%m-%d")
    ext      = trade.get("ext_vs_bo2") or 0
    pullb    = trade.get("pullback_vs_high_d2") or 0
    slope    = trade.get("ma200_slope20") or 0
    title = (
        f"{symbol} - BPE (Tier {trade['tier']}) - {sig_date}   "
        f"[{trade['result']}]   PnL {trade['pnl_pct']:+.1f}%   "
        f"R:R {trade['rr']:.1f}   Hold {trade['days_held']}d   "
        f"gap(d1,d2) {trade.get('gap_d2_d1','?')}  age(d2) {trade.get('gap_t_d2','?')}   "
        f"Ext {ext:+.2f}%  Pullb {pullb:+.2f}%  MA200 {slope:+.2f}%"
    )
    fig.suptitle(title, color="#e2e8f0", fontsize=9, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("BACKTEST - BPE (watchlist_breakout_pullback_scanner.md)")
    print("=" * 60)

    universe = {**app.VN30_STOCKS, **app.VNMID_STOCKS}

    all_trades: list[dict] = []
    for sym in universe.keys():
        df = load_cached(sym)
        if df is None:
            continue
        hits = find_bpe_signals(df)
        if not hits:
            continue

        sym_trades: list[dict] = []
        prev_i = -999
        for idx, sig in hits:
            if idx - prev_i < COOLDOWN:
                continue
            tr = simulate(df, idx, sig)
            if tr is None:
                continue
            tr["symbol"] = sym.replace(".VN", "")
            tr["df"]     = df
            sym_trades.append(tr)
            all_trades.append(tr)
            prev_i = idx

        if sym_trades:
            w = sum(1 for t in sym_trades if t["result"] == "WIN")
            print(f"  {sym:<10} {len(sym_trades):2d} BPE  win {w}/{len(sym_trades)}")

    if not all_trades:
        print("\nNo BPE signals found across history.")
        sys.exit(1)

    total  = len(all_trades)
    n_win  = sum(1 for t in all_trades if t["result"] == "WIN")
    n_loss = sum(1 for t in all_trades if t["result"] == "LOSS")
    n_to   = total - n_win - n_loss
    pf_w   = sum(t["pnl_pct"] for t in all_trades if t["result"] == "WIN")
    pf_l   = abs(sum(t["pnl_pct"] for t in all_trades if t["result"] == "LOSS"))

    print("\n" + "=" * 60)
    print(f"TONG KET BPE  ({total} trades)")
    print(f"  Win rate     : {n_win/total*100:.1f}%  ({n_win}W / {n_loss}L / {n_to}TO)")
    print(f"  Profit Factor: {pf_w/max(pf_l, 0.01):.2f}")
    print(f"  Avg PnL WIN  : {pf_w/max(n_win,1):.1f}%")
    print(f"  Avg PnL LOSS : {pf_l/max(n_loss,1):.1f}%")
    print("=" * 60)

    # Pick 10 samples with a win/loss/timeout mix
    import random
    random.seed(42)
    wins = [t for t in all_trades if t["result"] == "WIN"]
    loss = [t for t in all_trades if t["result"] == "LOSS"]
    tos  = [t for t in all_trades if t["result"] == "TIMEOUT"]

    n_w = min(5, len(wins))
    n_l = min(3, len(loss))
    n_t = min(10 - n_w - n_l, len(tos))
    picked = (
        random.sample(wins, n_w)
        + random.sample(loss, n_l)
        + random.sample(tos,  n_t)
    )
    if len(picked) < 10:
        pool = [t for t in all_trades if t not in picked]
        picked += random.sample(pool, min(10 - len(picked), len(pool)))
    random.shuffle(picked)

    removed = 0
    for f in os.listdir(BACKTEST_DIR):
        if f.startswith("bpe_") and f.endswith(".png"):
            os.remove(os.path.join(BACKTEST_DIR, f))
            removed += 1
    if removed:
        print(f"\nXoa {removed} chart BPE cu")

    print(f"\nVe {len(picked)} charts -> {BACKTEST_DIR}/")
    for i, tr in enumerate(picked, 1):
        sym      = tr["symbol"]
        res      = tr["result"][:2]
        date_str = tr["df"].index[tr["signal_idx"]].strftime("%Y%m%d")
        pnl      = (f"{tr['pnl_pct']:+.1f}"
                    .replace("+", "p").replace("-", "m").replace(".", "d"))
        fn = f"bpe_{i:02d}_{sym}_{res}_{date_str}_{pnl}pct.png"
        draw(tr["df"], tr, sym, os.path.join(BACKTEST_DIR, fn))
        print(f"  [{i:02d}] {fn}")

    print(f"\nHoan tat. {len(picked)} charts trong {BACKTEST_DIR}/")


if __name__ == "__main__":
    main()
