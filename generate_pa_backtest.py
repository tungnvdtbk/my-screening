"""
Backtest script — Price Action Breakout & Pullback on VN100 (3yr).
Generates 10 PNG sample charts into data/backtest/.

Run:
    python generate_pa_backtest.py
"""

from __future__ import annotations

import os
import sys
import types as _t

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

BG_COLOR   = "#0a0e17"
UP_COLOR   = "#26a69a"
DOWN_COLOR = "#ef5350"
MA20_COLOR = "#34d399"
MA50_COLOR = "#f59e0b"

VN100_SYMBOLS = list(app.VN100_STOCKS.keys())


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


def _scan_pa_backtest(df: pd.DataFrame) -> dict | None:
    """
    Relaxed PA scanner for backtesting.
    - No value gate, no RS gate
    - buildup_days >= 2, buildup_score >= 4
    - rr >= 1.0
    - Breakout: vol_expand 1.3x, pullback: vol_ok 0.8x
    """
    if df is None or len(df) < 60:
        return None

    d = app.compute_pa_indicators(df)
    row = d.iloc[-1]

    required = [
        "pa_ma20", "pa_ma50", "pa_vol_ma20", "pa_rsi", "pa_tightness",
        "pa_buildup_score", "pa_buildup_days", "pa_pivot_20",
        "pa_mom_accel", "pa_is_buildup", "pa_trend_filter",
        "pa_range_expansion", "pa_bar_range", "pa_upper_tail",
    ]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    volume = float(row["Volume"])
    ma20 = float(row["pa_ma20"])
    ma50 = float(row["pa_ma50"])
    rsi = float(row["pa_rsi"])
    vol_ma20 = float(row["pa_vol_ma20"])
    pivot_20 = float(row["pa_pivot_20"])
    bar_range = float(row["pa_bar_range"])
    range_expansion = float(row["pa_range_expansion"])
    upper_tail = float(row["pa_upper_tail"])
    buildup_score = int(row["pa_buildup_score"])
    buildup_days = float(row["pa_buildup_days"])
    trend_ok = bool(row["pa_trend_filter"])
    strong_barrier = bool(row.get("pa_strong_barrier", False))
    is_squeeze = bool(row.get("pa_is_squeeze", False))
    mom_accel = float(row["pa_mom_accel"])

    if not trend_ok:
        return None
    if close > ma20 * 1.05:
        return None
    if close > ma50 * 1.15:
        return None
    if rsi >= 72:
        return None

    # Relaxed buildup
    near_ma20 = bool(row.get("pa_near_ma20", False))
    no_dist = bool(row.get("pa_no_distribution", False))
    if not (near_ma20 and no_dist and buildup_score >= 4):
        return None
    if buildup_days < 2:
        return None

    # Try breakout
    price_break = close > pivot_20
    break_dist = close / max(pivot_20, 1e-9)
    clean_break = break_dist <= 1.03
    vol_expand = volume > vol_ma20 * 1.30
    strong_close = ((close - low) / max(bar_range, 1e-9)) > 0.55
    small_upper = upper_tail < 0.40
    acceptable = range_expansion <= 2.50

    setup_breakout = (price_break and clean_break and vol_expand
                      and strong_close and small_upper and acceptable)

    # Try pullback
    prior_push = bool(row.get("pa_prior_push", False))
    pullback_depth = float(row.get("pa_pullback_depth", 0))
    pullback_to_ma20 = low <= ma20 * 1.02
    close_holds = close >= ma20 * 0.99
    shallow = close >= ma50

    bull_reclaim = len(d) >= 2 and close > float(d.iloc[-2]["High"])
    reversal_bar = (close > open_) and bull_reclaim
    vol_ok = volume >= vol_ma20 * 0.80

    setup_pullback = (prior_push and pullback_depth > 0.02
                      and pullback_to_ma20 and close_holds and shallow
                      and reversal_bar and vol_ok
                      and buildup_score >= 5)   # match production: higher quality pullbacks

    if setup_breakout:
        setup_type = "PA_BREAKOUT"
        stop_loss = round(max(low, ma20 * 0.97), 2)
    elif setup_pullback:
        setup_type = "PA_PULLBACK"
        stop_loss = round(max(low, ma50 * 0.98), 2)
    else:
        return None

    target_1 = round(close * 1.07, 2)
    target_2 = round(close * 1.12, 2)
    rr = round((target_1 - close) / max(close - stop_loss, 1e-9), 2)
    if rr < 1.0:
        return None

    trigger_score = (int(price_break) + int(vol_expand) + int(strong_close)
                     + int(mom_accel > 0) + int(strong_barrier) + int(is_squeeze))

    return {
        "signal": setup_type,
        "close": round(close, 2),
        "sl": stop_loss,
        "tp": target_1,
        "tp2": target_2,
        "rr": rr,
        "buildup_score": buildup_score,
        "trigger_score": trigger_score,
        "rsi": round(rsi, 1),
        "tightness": round(float(row["pa_tightness"]), 4),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
    }


def find_pa_signals(df: pd.DataFrame) -> list[tuple[int, dict]]:
    results = []
    for i in range(60, len(df)):
        sig = _scan_pa_backtest(df.iloc[:i + 1])
        if sig:
            results.append((i, sig))
    return results


def simulate_trade(df, signal_idx, sig, max_hold=15):
    if signal_idx + 1 >= len(df):
        return None

    entry = df.iloc[signal_idx + 1]["Open"] * 1.001
    sl = sig["sl"]
    tp = sig["tp"]

    if entry <= sl or pd.isna(sl) or pd.isna(tp):
        return None

    rr = (tp - entry) / (entry - sl) if entry > sl else 0.0
    result = "TIMEOUT"
    exit_price = df.iloc[min(signal_idx + max_hold, len(df) - 1)]["Close"]
    exit_idx = min(signal_idx + max_hold, len(df) - 1)
    current_sl = sl
    tp1_touched = False

    for j in range(signal_idx + 1, min(signal_idx + 1 + max_hold, len(df))):
        candle = df.iloc[j]
        days_in = j - signal_idx

        if candle["Low"] <= current_sl:
            if tp1_touched:
                result = "WIN"
                exit_price = max(current_sl, entry)
            else:
                result = "LOSS"
                exit_price = current_sl
            exit_idx = j
            break

        if candle["High"] >= tp and not tp1_touched:
            tp1_touched = True
            current_sl = entry
            result = "WIN"
            exit_price = tp
            exit_idx = j
            break

        if days_in >= 8 and j >= 3:
            trail = min(df.iloc[j-1]["Low"], df.iloc[j-2]["Low"], df.iloc[j-3]["Low"]) * 0.99
            if trail > current_sl:
                current_sl = trail

    return {
        "signal_type": sig["signal"],
        "signal_idx": signal_idx,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": round(rr, 2),
        "result": result,
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "pnl_pct": (exit_price - entry) / entry * 100,
        "days_held": exit_idx - signal_idx,
        "buildup_score": sig.get("buildup_score", 0),
        "trigger_score": sig.get("trigger_score", 0),
    }


def draw_chart(df, trade, symbol, out_path):
    si = trade["signal_idx"]
    start = max(0, si - 40)
    end = min(len(df), trade["exit_idx"] + 6)
    view = df.iloc[start:end].copy()
    n = len(view)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
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

    x_range = range(n)
    if "pa_ma20" in view.columns and not view["pa_ma20"].isna().all():
        ax1.plot(x_range, view["pa_ma20"], color=MA20_COLOR, linewidth=1.0, label="MA20")
    if "pa_ma50" in view.columns and not view["pa_ma50"].isna().all():
        ax1.plot(x_range, view["pa_ma50"], color=MA50_COLOR, linewidth=1.0, label="MA50")

    view_index = list(view.index)
    df_index = list(df.index)

    def to_view_x(df_i):
        if 0 <= df_i < len(df_index):
            try:
                return view_index.index(df_index[df_i])
            except ValueError:
                return None
        return None

    sig_x = to_view_x(si)
    exit_x = to_view_x(trade["exit_idx"])

    if sig_x is not None and sig_x < n:
        ax1.axvline(sig_x, color="#fbbf24", linewidth=1.5, linestyle="--", alpha=0.7)
        ax1.scatter(sig_x, view.iloc[sig_x]["High"] * 1.01,
                    marker="^", color="#fbbf24", s=120, zorder=5)

    ax1.axhline(trade["sl"], color=DOWN_COLOR, linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"SL {trade['sl']:.1f}")
    ax1.axhline(trade["tp"], color=UP_COLOR, linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"TP {trade['tp']:.1f}")

    if exit_x is not None and exit_x < n:
        r_color = UP_COLOR if trade["result"] == "WIN" else (
            DOWN_COLOR if trade["result"] == "LOSS" else "#fbbf24")
        ax1.axvline(exit_x, color=r_color, linewidth=1.5, linestyle=":", alpha=0.8)
        ax1.scatter(exit_x, trade["exit_price"], marker="o", color=r_color, s=100, zorder=5)

    ax1.set_ylabel("Price", color="#94a3b8", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7,
               facecolor=BG_COLOR, edgecolor="#334155", labelcolor="#e2e8f0")

    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                  for c, o in zip(view["Close"], view["Open"])]
    ax2.bar(x_range, view["Volume"], color=vol_colors, alpha=0.7)
    ax2.set_ylabel("Volume", color="#94a3b8", fontsize=8)

    tick_step = max(1, n // 8)
    ticks = list(range(0, n, tick_step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [view.index[i].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=7, color="#94a3b8",
    )

    r_label = {"WIN": "[WIN]", "LOSS": "[LOSS]", "TIMEOUT": "[TIMEOUT]"}.get(trade["result"], "")
    sig_date = df.index[si].strftime("%Y-%m-%d")
    sig_type = trade["signal_type"].replace("PA_", "")
    title = (
        f"{symbol} · {sig_type} · {sig_date}   "
        f"{r_label}   PnL {trade['pnl_pct']:+.1f}%   "
        f"R:R {trade['rr']:.1f}   Hold {trade['days_held']}d   "
        f"Buildup {trade['buildup_score']}/11   Trigger {trade['trigger_score']}/6"
    )
    fig.suptitle(title, color="#e2e8f0", fontsize=9, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def main():
    print("=" * 60)
    print("BACKTEST — Price Action Breakout & Pullback on VN100 (3yr)")
    print("=" * 60)

    all_trades = []

    for sym in VN100_SYMBOLS:
        print(f"  {sym:<12}", end=" ")
        df = load_data(sym)
        if df is None or len(df) < 100:
            print("skip")
            continue

        df = app.compute_pa_indicators(df)
        signals = find_pa_signals(df)

        sym_trades = []
        prev_sig = -999

        for idx, sig in sorted(signals, key=lambda x: x[0]):
            if idx - prev_sig < 5:
                continue
            trade = simulate_trade(df, idx, sig)
            if trade and trade["rr"] >= 0.8:
                trade["symbol"] = sym.replace(".VN", "")
                trade["df"] = df
                sym_trades.append(trade)
                all_trades.append(trade)
                prev_sig = idx

        n_bo = sum(1 for t in sym_trades if "BREAKOUT" in t["signal_type"])
        n_pb = sum(1 for t in sym_trades if "PULLBACK" in t["signal_type"])
        wins = sum(1 for t in sym_trades if t["result"] == "WIN")
        print(f"{len(sym_trades):2d} PA ({n_bo}BO/{n_pb}PB)  win {wins}/{len(sym_trades)}")

    if not all_trades:
        print("\nNo PA signals found.")
        sys.exit(1)

    total = len(all_trades)
    n_win = sum(1 for t in all_trades if t["result"] == "WIN")
    n_loss = sum(1 for t in all_trades if t["result"] == "LOSS")
    n_to = total - n_win - n_loss
    pf_w = sum(t["pnl_pct"] for t in all_trades if t["result"] == "WIN")
    pf_l = abs(sum(t["pnl_pct"] for t in all_trades if t["result"] == "LOSS"))

    print(f"\n{'='*60}")
    print(f"TOTAL PA  ({total} trades, RR >= 0.8)")
    print(f"  Win rate      : {n_win/total*100:.1f}%  ({n_win}W / {n_loss}L / {n_to}TO)")
    print(f"  Profit Factor : {pf_w/max(pf_l, 0.01):.2f}")
    print(f"  Avg PnL WIN   : {pf_w/max(n_win,1):.1f}%")
    print(f"  Avg PnL LOSS  : {pf_l/max(n_loss,1):.1f}%")
    print(f"  Avg R:R       : {sum(t['rr'] for t in all_trades)/total:.2f}")
    print(f"  Breakouts     : {sum(1 for t in all_trades if 'BREAKOUT' in t['signal_type'])}")
    print(f"  Pullbacks     : {sum(1 for t in all_trades if 'PULLBACK' in t['signal_type'])}")
    print("=" * 60)

    import random
    random.seed(42)

    wins_list = [t for t in all_trades if t["result"] == "WIN"]
    losses_list = [t for t in all_trades if t["result"] == "LOSS"]
    timeout_list = [t for t in all_trades if t["result"] == "TIMEOUT"]

    n_target = 10
    n_wins = min(5, len(wins_list))
    n_losses = min(3, len(losses_list))
    n_tos = min(n_target - n_wins - n_losses, len(timeout_list))

    selected = random.sample(wins_list, n_wins) if n_wins > 0 else []
    selected += random.sample(losses_list, n_losses) if n_losses > 0 else []
    selected += random.sample(timeout_list, n_tos) if n_tos > 0 else []

    remaining = n_target - len(selected)
    if remaining > 0:
        pool = [t for t in all_trades if t not in selected]
        selected += random.sample(pool, min(remaining, len(pool)))

    random.shuffle(selected)

    removed = 0
    for f in os.listdir(BACKTEST_DIR):
        if f.endswith(".png") and "_PA_" in f:
            os.remove(os.path.join(BACKTEST_DIR, f))
            removed += 1
    if removed:
        print(f"\nRemoved {removed} old PA charts")

    print(f"\nDrawing {len(selected)} charts -> {BACKTEST_DIR}/\n")
    for i, trade in enumerate(selected, 1):
        sym = trade["symbol"]
        sig_type = "BO" if "BREAKOUT" in trade["signal_type"] else "PB"
        result = trade["result"][:2]
        date_str = trade["df"].index[trade["signal_idx"]].strftime("%Y%m%d")
        pnl = f"{trade['pnl_pct']:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        filename = f"{i:02d}_{sym}_PA_{sig_type}_{result}_{date_str}_{pnl}pct.png"
        out_path = os.path.join(BACKTEST_DIR, filename)

        draw_chart(trade["df"], trade, sym, out_path)
        print(f"  [{i:02d}] {filename}")

    print(f"\nDone. {len(selected)} charts in {BACKTEST_DIR}/")


if __name__ == "__main__":
    main()
