"""
Backtest script — Swing Filter on VN30 (3 years).
Generates 10 PNG sample charts into data/backtest/.

Run:
    python generate_swing_backtest.py
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
MA20_COLOR   = "#34d399"
MA50_COLOR   = "#f59e0b"

VN30_SYMBOLS = list(app.VN30_STOCKS.keys())
VN100_SYMBOLS = list(app.VN100_STOCKS.keys())


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


# ── Find Swing Filter signals across history ────────────────────────────────

def find_swing_signals(df: pd.DataFrame) -> list[tuple[int, dict]]:
    """
    Slide a relaxed swing scan across full history for backtesting.
    Uses slightly looser thresholds than production to generate enough signals
    for a meaningful backtest (buildup_days >= 2, entry needs 2/3 confirms).
    """
    results = []
    min_bars = 60
    for i in range(min_bars, len(df)):
        window = df.iloc[:i + 1]
        sig = _scan_swing_backtest(window)
        if sig:
            results.append((i, sig))
    return results


def _scan_swing_backtest(df: pd.DataFrame) -> dict | None:
    """
    Relaxed swing filter for historical backtesting.
    Matches production improvements (tighter SL, ATR TP, mom mandatory)
    but relaxed on: no value gate, no RS gate, buildup_days >= 2, rr >= 1.0.
    """
    if df is None or len(df) < 60:
        return None

    d = app.compute_swing_indicators(df)
    row = d.iloc[-1]

    required = [
        "sw_ma20", "sw_ma50", "sw_vol_ma20", "sw_vol_ma5",
        "sw_rsi", "sw_tightness", "sw_buildup_score",
        "sw_buildup_days", "sw_pivot_high", "sw_mom_accel",
        "sw_is_buildup", "sw_atr10", "sw_low_5d",
    ]
    if any(pd.isna(row.get(c, float("nan"))) for c in required):
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    volume = float(row["Volume"])
    ma20 = float(row["sw_ma20"])
    ma50 = float(row["sw_ma50"])
    rsi = float(row["sw_rsi"])
    vol_ma20 = float(row["sw_vol_ma20"])
    pivot_high = float(row["sw_pivot_high"])
    mom_accel = float(row["sw_mom_accel"])
    tightness = float(row["sw_tightness"])
    buildup_score = int(row["sw_buildup_score"])
    buildup_days = float(row["sw_buildup_days"])
    is_buildup = bool(row["sw_is_buildup"])
    atr10 = float(row["sw_atr10"])
    low_5d = float(row["sw_low_5d"])

    # Hard filters
    if not (close > ma50):
        return None
    if not (ma20 > ma50):
        return None
    if close > ma20 * 1.05:
        return None
    if close > ma50 * 1.15:
        return None
    if rsi >= 72:
        return None
    if not is_buildup:
        return None
    if buildup_days < 2:
        return None

    # Entry trigger (relaxed: price_break + at least 1 confirm)
    price_break = close > pivot_high
    vol_expand = volume > vol_ma20 * 1.3   # relaxed from 1.5x
    candle_range = high - low
    strong_bar = ((close - low) / max(candle_range, 1e-9)) > 0.5
    mom_ok = mom_accel > 0

    if not price_break:
        return None
    if not (vol_expand or strong_bar):
        return None

    trigger_score = int(price_break) + int(vol_expand) + int(strong_bar) + int(mom_ok)

    # SL/TP from spec — improvements are in exit management (trailing + time stop)
    entry = close
    stop_loss = round(ma50 * 0.97, 2)
    target_1 = round(entry * 1.07, 2)
    target_2 = round(entry * 1.12, 2)
    rr_ratio = round((target_1 - entry) / max(entry - stop_loss, 1e-9), 2)

    if rr_ratio < 1.0:
        return None

    return {
        "signal": "SWING_FILTER",
        "date": df.index[-1],
        "close": round(close, 2),
        "rsi": round(rsi, 1),
        "tightness": round(tightness, 4),
        "buildup_score": buildup_score,
        "buildup_days": int(buildup_days),
        "trigger_score": trigger_score,
        "sl": stop_loss,
        "tp": target_1,
        "tp2": target_2,
        "rr": rr_ratio,
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
    }


# ── Trade simulation ─────────────────────────────────────────────────────────

def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    sig: dict,
    max_hold: int = 15,
) -> dict | None:
    """
    Simulate using improved Swing Filter exit logic:
    - Entry = open of next bar * 1.001 (0.1% slippage)
    - Day 1-3: hard SL only
    - Day 4+: if high >= TP1, move SL to breakeven
    - Day 8+: trailing stop = min(low of last 3 bars) * 0.99
    - Day 15: hard time-stop, exit at close
    """
    if signal_idx + 1 >= len(df):
        return None

    entry_row = df.iloc[signal_idx + 1]
    entry     = entry_row["Open"] * 1.001   # 0.1% slippage
    sl        = sig["sl"]
    tp        = sig["tp"]

    if entry <= sl or pd.isna(sl) or pd.isna(tp):
        return None

    rr = (tp - entry) / (entry - sl) if entry > sl else 0.0

    result     = "TIMEOUT"
    exit_price = df.iloc[min(signal_idx + max_hold, len(df) - 1)]["Close"]
    exit_idx   = min(signal_idx + max_hold, len(df) - 1)

    tp1_touched = False
    current_sl  = sl

    for j in range(signal_idx + 1, min(signal_idx + 1 + max_hold, len(df))):
        candle    = df.iloc[j]
        days_in   = j - signal_idx

        # Check SL first (conservative)
        if candle["Low"] <= current_sl:
            if tp1_touched:
                result     = "WIN"
                exit_price = max(current_sl, entry)
            else:
                result     = "LOSS"
                exit_price = current_sl
            exit_idx = j
            break

        # TP1 hit: move SL to breakeven (lock in gains)
        if candle["High"] >= tp and not tp1_touched:
            tp1_touched = True
            current_sl  = entry
            result      = "WIN"
            exit_price  = tp
            exit_idx    = j
            break

        # Trailing stop after day 8 using 3-bar trailing low (wide trail)
        if days_in >= 8 and j >= 3:
            trail_low = min(
                df.iloc[j-1]["Low"], df.iloc[j-2]["Low"], df.iloc[j-3]["Low"]
            )
            trail_sl = trail_low * 0.99   # 1% below 3-bar low
            if trail_sl > current_sl:
                current_sl = trail_sl

    return {
        "signal_type":    sig["signal"],
        "signal_idx":     signal_idx,
        "entry":          entry,
        "sl":             sl,
        "tp":             tp,
        "tp2":            sig.get("tp2", tp * 1.05),
        "rr":             round(rr, 2),
        "result":         result,
        "exit_idx":       exit_idx,
        "exit_price":     exit_price,
        "pnl_pct":        (exit_price - entry) / entry * 100,
        "days_held":      exit_idx - signal_idx,
        "buildup_score":  sig.get("buildup_score", 0),
        "trigger_score":  sig.get("trigger_score", 0),
        "rsi":            sig.get("rsi", 0),
        "tightness":      sig.get("tightness", 0),
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
    if "sw_ma20" in view.columns and not view["sw_ma20"].isna().all():
        ax1.plot(x_range, view["sw_ma20"], color=MA20_COLOR, linewidth=1.0, label="MA20")
    if "sw_ma50" in view.columns and not view["sw_ma50"].isna().all():
        ax1.plot(x_range, view["sw_ma50"], color=MA50_COLOR, linewidth=1.0, label="MA50")

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
                label=f"TP1 {trade['tp']:.1f}")

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
    if "sw_vol_ma20" in view.columns and not view["sw_vol_ma20"].isna().all():
        ax2.plot(x_range, view["sw_vol_ma20"], color="#fbbf24", linewidth=0.8, label="VolMA20")
    ax2.set_ylabel("Volume", color="#94a3b8", fontsize=8)

    # X-axis labels
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
    title = (
        f"{symbol} · SWING_FILTER · {sig_date}   "
        f"{r_label} {trade['result']}   "
        f"PnL {trade['pnl_pct']:+.1f}%   "
        f"R:R {trade['rr']:.1f}   "
        f"Hold {trade['days_held']}d   "
        f"Buildup {trade['buildup_score']}/7   "
        f"Trigger {trade['trigger_score']}/4"
    )
    fig.suptitle(title, color="#e2e8f0", fontsize=9, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BACKTEST — Swing Filter on VN100 (3yr)")
    print("=" * 60)

    all_trades: list[dict] = []

    for sym in VN100_SYMBOLS:
        print(f"  {sym:<12}", end=" ")
        df = load_data(sym)
        if df is None or len(df) < 100:
            print("skip (no data)")
            continue

        df = app.compute_swing_indicators(df)
        signals = find_swing_signals(df)

        sym_trades: list[dict] = []
        prev_sig = -999

        for idx, sig in sorted(signals, key=lambda x: x[0]):
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
        print(f"{len(sym_trades):2d} SW trades  win {wins}/{len(sym_trades)}")

    if not all_trades:
        print("\nKhong co tin hieu Swing Filter. Kiem tra lai du lieu.")
        sys.exit(1)

    # ── Summary stats ─────────────────────────────────────────────
    total  = len(all_trades)
    n_win  = sum(1 for t in all_trades if t["result"] == "WIN")
    n_loss = sum(1 for t in all_trades if t["result"] == "LOSS")
    n_to   = total - n_win - n_loss
    pf_w   = sum(t["pnl_pct"] for t in all_trades if t["result"] == "WIN")
    pf_l   = abs(sum(t["pnl_pct"] for t in all_trades if t["result"] == "LOSS"))

    print("\n" + "=" * 60)
    print(f"TONG KET SWING FILTER  ({total} trades, RR >= 0.8)")
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

    selected  = random.sample(wins_list,    n_wins)   if n_wins   > 0 else []
    selected += random.sample(losses_list,  n_losses) if n_losses > 0 else []
    selected += random.sample(timeout_list, n_tos)    if n_tos    > 0 else []

    remaining = n_target - len(selected)
    if remaining > 0:
        pool = [t for t in all_trades if t not in selected]
        selected += random.sample(pool, min(remaining, len(pool)))

    random.shuffle(selected)

    # ── Clear old Swing backtest PNGs only ──────────────────────────
    removed = 0
    for f in os.listdir(BACKTEST_DIR):
        if f.endswith(".png") and "_SW_" in f:
            os.remove(os.path.join(BACKTEST_DIR, f))
            removed += 1
    if removed:
        print(f"\nXoa {removed} swing chart cu")

    # ── Draw charts ────────────────────────────────────────────────
    print(f"\nVe {len(selected)} charts -> {BACKTEST_DIR}/\n")
    for i, trade in enumerate(selected, 1):
        sym      = trade["symbol"]
        result   = trade["result"][:2]
        date_str = trade["df"].index[trade["signal_idx"]].strftime("%Y%m%d")
        pnl      = f"{trade['pnl_pct']:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        filename = f"{i:02d}_{sym}_SW_{result}_{date_str}_{pnl}pct.png"
        out_path = os.path.join(BACKTEST_DIR, filename)

        draw_chart(trade["df"], trade, sym, out_path)
        print(f"  [{i:02d}] {filename}")

    print(f"\nHoan tat. {len(selected)} charts trong {BACKTEST_DIR}/")


if __name__ == "__main__":
    main()
