"""
Backtest script — Breakout Momentum & Reversal Hunter on VN30 (2 years).
Generates ~30 PNG charts into data/backtest/.

Run:
    python generate_backtest.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
MA50_COLOR = "#f59e0b"
MA200_COLOR = "#818cf8"

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
            if len(df) >= 60:
                return df
        except Exception:
            pass
    try:
        df = yf.Ticker(symbol).history(period="2y")
        if df is None or df.empty:
            return None
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.to_parquet(path)
        return df
    except Exception:
        return None


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift(1)).abs(),
        (d["Low"]  - d["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["atr10"]        = tr.shift(2).rolling(10).mean()
    d["avg_vol20"]    = d["Volume"].shift(2).rolling(20).mean()
    d["avg_vol_pre5"] = d["Volume"].shift(2).rolling(5).mean()
    d["high10"]       = d["High"].shift(2).rolling(10).max()
    d["high20"]       = d["High"].shift(2).rolling(20).max()
    d["ma50"]         = d["Close"].rolling(50).mean()
    d["ma200"]        = d["Close"].rolling(200).mean()
    d["ma50_prev5"]   = d["ma50"].shift(5)
    return d


# ── Signal detection ──────────────────────────────────────────────────────────

def find_breakout_signals(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Returns list of (index, signal_type) where signal_type is BREAKOUT_STRONG or BREAKOUT_EARLY."""
    signals = []
    for i in range(60, len(df)):
        row = df.iloc[i]
        if any(pd.isna(row[c]) for c in ["atr10", "avg_vol20", "high10", "high20", "ma50", "ma50_prev5"]):
            continue
        atr10 = row["atr10"]; ma50 = row["ma50"]
        close = row["Close"]; open_ = row["Open"]
        high  = row["High"];  low   = row["Low"]
        if not (close > ma50 and ma50 > row["ma50_prev5"]):
            continue
        if not (close > open_):
            continue
        if not (close >= high * 0.998):
            continue
        if not ((high - low) > 1.5 * atr10):
            continue
        # [4b] Body ≥ 60% of full range
        candle_range = high - low
        if candle_range > 0 and (close - open_) / candle_range < 0.6:
            continue
        if not (high > row["high10"]):
            continue
        if close > ma50 * 1.08:
            continue
        sig_type = "BREAKOUT_STRONG" if high > row["high20"] else "BREAKOUT_EARLY"
        signals.append((i, sig_type))
    return signals


def find_reversal_signals(df: pd.DataFrame) -> list[int]:
    signals = []
    for i in range(210, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        if any(pd.isna(row[c]) for c in ["atr10", "avg_vol20", "ma50", "ma200"]):
            continue
        atr10 = row["atr10"]; ma50 = row["ma50"]; ma200 = row["ma200"]
        close = row["Close"]; open_ = row["Open"]
        high  = row["High"];  low   = row["Low"]
        if not (close < ma50):
            continue
        if not (abs(close - ma200) <= 1.0 * atr10):
            continue
        if not (low < prev["Low"]):
            continue
        if not (close > open_):
            continue
        if not (close >= high * 0.998):
            continue
        if not ((high - low) > 1.5 * atr10):
            continue
        # [6b] Body ≥ 60% of full range
        candle_range = high - low
        if candle_range > 0 and (close - open_) / candle_range < 0.6:
            continue
        if close < ma200 * 0.92:
            continue
        signals.append(i)
    return signals


# ── Trade simulation ──────────────────────────────────────────────────────────

def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    signal_type: str,
    tp_mult: float = 2.0,
    max_hold: int  = 15,
) -> dict | None:
    if signal_idx + 1 >= len(df):
        return None

    sig_row   = df.iloc[signal_idx]
    entry_row = df.iloc[signal_idx + 1]

    entry = entry_row["Open"] * 1.001   # 0.1% slippage
    sl    = sig_row["Low"]
    atr10 = sig_row["atr10"]
    tp    = entry + tp_mult * atr10

    if entry <= sl or pd.isna(atr10):
        return None

    rr = (tp - entry) / (entry - sl)

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
        "signal_type": signal_type,
        "signal_idx":  signal_idx,
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "rr":          round(rr, 2),
        "result":      result,
        "exit_idx":    exit_idx,
        "exit_price":  exit_price,
        "pnl_pct":     (exit_price - entry) / entry * 100,
        "days_held":   exit_idx - signal_idx,
    }


# ── Chart drawing ─────────────────────────────────────────────────────────────

def draw_chart(df: pd.DataFrame, trade: dict, symbol: str, out_path: str) -> None:
    si    = trade["signal_idx"]
    start = max(0, si - 30)
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

    # MA lines
    x_range = range(n)
    if not view["ma50"].isna().all():
        ax1.plot(x_range, view["ma50"],  color=MA50_COLOR,  linewidth=1.0, label="MA50")
    if not view["ma200"].isna().all():
        ax1.plot(x_range, view["ma200"], color=MA200_COLOR, linewidth=1.0, label="MA200")

    # Locate signal and exit bars within the view slice
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
        bar_high = view.iloc[sig_x]["High"]
        ax1.scatter(sig_x, bar_high * 1.01, marker="^", color="#fbbf24", s=120, zorder=5)

    # SL / TP lines
    ax1.axhline(trade["sl"], color=DOWN_COLOR, linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"SL {trade['sl']:.1f}")
    ax1.axhline(trade["tp"], color=UP_COLOR,   linewidth=1.2, linestyle="--", alpha=0.9,
                label=f"TP {trade['tp']:.1f}")

    # Exit marker
    if exit_x is not None and exit_x < n:
        r_color = UP_COLOR if trade["result"] == "WIN" else (DOWN_COLOR if trade["result"] == "LOSS" else "#fbbf24")
        ax1.axvline(exit_x, color=r_color, linewidth=1.5, linestyle=":", alpha=0.8)
        ax1.scatter(exit_x, trade["exit_price"], marker="o", color=r_color, s=100, zorder=5)

    ax1.set_ylabel("Giá", color="#94a3b8", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7,
               facecolor=BG_COLOR, edgecolor="#334155", labelcolor="#e2e8f0")

    # Volume
    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                  for c, o in zip(view["Close"], view["Open"])]
    ax2.bar(x_range, view["Volume"], color=vol_colors, alpha=0.7)
    if not view["avg_vol20"].isna().all():
        ax2.plot(x_range, view["avg_vol20"], color="#fbbf24", linewidth=0.8, label="AvgVol20")
    ax2.set_ylabel("Volume", color="#94a3b8", fontsize=8)

    # X-axis ticks
    tick_step = max(1, n // 8)
    ticks     = list(range(0, n, tick_step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [view.index[i].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=7, color="#94a3b8",
    )

    # Title
    r_emoji = {"WIN": "[WIN]", "LOSS": "[LOSS]", "TIMEOUT": "[TIMEOUT]"}.get(trade["result"], "")
    sig_date = df.index[si].strftime("%Y-%m-%d")
    title = (
        f"{symbol} · {trade['signal_type']} · {sig_date}   "
        f"{r_emoji} {trade['result']}   "
        f"PnL {trade['pnl_pct']:+.1f}%   "
        f"R:R {trade['rr']:.1f}   "
        f"Hold {trade['days_held']}d"
    )
    fig.suptitle(title, color="#e2e8f0", fontsize=9, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BACKTEST — Breakout & Reversal trên VN30 (2 năm)")
    print("=" * 60)

    all_trades: list[dict] = []

    for sym in VN30_SYMBOLS:
        print(f"  {sym:<12}", end=" ")
        df = load_data(sym)
        if df is None or len(df) < 60:
            print("skip (no data)")
            continue

        df = compute_indicators(df)

        bo_list  = find_breakout_signals(df)   # list of (idx, sig_type)
        rev_idx  = find_reversal_signals(df)   # list of int
        bo_type_map = {idx: t for idx, t in bo_list}
        bo_idx_set  = set(bo_type_map.keys())

        sym_trades: list[dict] = []
        cooldown  = 10
        prev_sig  = -999

        for idx in sorted(bo_idx_set | set(rev_idx)):
            if idx - prev_sig < cooldown:
                continue
            sig_type = bo_type_map.get(idx, "REVERSAL")
            trade    = simulate_trade(df, idx, sig_type)
            if trade and trade["rr"] >= 1.0:
                trade["symbol"] = sym.replace(".VN", "")
                trade["df"]     = df
                sym_trades.append(trade)
                all_trades.append(trade)
                prev_sig = idx

        bo_count  = sum(1 for t in sym_trades if t["signal_type"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY"))
        rev_count = sum(1 for t in sym_trades if t["signal_type"] == "REVERSAL")
        wins      = sum(1 for t in sym_trades if t["result"] == "WIN")
        print(f"{len(sym_trades):2d} trades "
              f"(BO:{bo_count} REV:{rev_count})  "
              f"win {wins}/{len(sym_trades)}")

    if not all_trades:
        print("\nKhông có tín hiệu. Kiểm tra lại dữ liệu.")
        sys.exit(1)

    # ── Summary stats
    total  = len(all_trades)
    n_win  = sum(1 for t in all_trades if t["result"] == "WIN")
    n_loss = sum(1 for t in all_trades if t["result"] == "LOSS")
    bo_strong = [t for t in all_trades if t["signal_type"] == "BREAKOUT_STRONG"]
    bo_early  = [t for t in all_trades if t["signal_type"] == "BREAKOUT_EARLY"]
    rev_all   = [t for t in all_trades if t["signal_type"] == "REVERSAL"]
    pf_w      = sum(t["pnl_pct"] for t in all_trades if t["result"] == "WIN")
    pf_l      = abs(sum(t["pnl_pct"] for t in all_trades if t["result"] == "LOSS"))

    def _wr(lst: list[dict]) -> str:
        if not lst: return "N/A"
        w = sum(1 for t in lst if t["result"] == "WIN")
        return f"{w}/{len(lst)} ({w/len(lst)*100:.0f}%)"

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT  ({total} trades hợp lệ)")
    print(f"  Win rate         : {n_win/total*100:.1f}%  ({n_win}W / {n_loss}L / {total-n_win-n_loss}TO)")
    print(f"  Profit Factor    : {pf_w/max(pf_l, 0.01):.2f}")
    print(f"  BREAKOUT_STRONG  : {len(bo_strong)} trades  win {_wr(bo_strong)}")
    print(f"  BREAKOUT_EARLY   : {len(bo_early)} trades  win {_wr(bo_early)}")
    print(f"  REVERSAL         : {len(rev_all)} trades  win {_wr(rev_all)}")
    print("=" * 60)

    # ── Select ~30 representative charts
    wins_sorted   = sorted(
        [t for t in all_trades if t["result"] == "WIN"],
        key=lambda t: t["pnl_pct"], reverse=True,
    )
    losses_sorted = sorted(
        [t for t in all_trades if t["result"] == "LOSS"],
        key=lambda t: t["pnl_pct"],
    )
    timeouts      = [t for t in all_trades if t["result"] == "TIMEOUT"]

    selected: list[dict] = []
    # Mix BREAKOUT and REVERSAL, WIN and LOSS
    selected += [t for t in wins_sorted   if t["signal_type"] == "BREAKOUT_STRONG"][:5]
    selected += [t for t in wins_sorted   if t["signal_type"] == "BREAKOUT_EARLY"][:3]
    selected += [t for t in wins_sorted   if t["signal_type"] == "REVERSAL"][:7]
    selected += [t for t in losses_sorted if t["signal_type"] in ("BREAKOUT_STRONG", "BREAKOUT_EARLY")][:5]
    selected += [t for t in losses_sorted if t["signal_type"] == "REVERSAL"][:5]
    selected += timeouts[:5]
    selected = selected[:30]

    # ── Clear old PNGs
    removed = 0
    for f in os.listdir(BACKTEST_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(BACKTEST_DIR, f))
            removed += 1
    if removed:
        print(f"\nXóa {removed} chart cũ")

    # ── Draw charts
    print(f"\nVẽ {len(selected)} charts → {BACKTEST_DIR}/\n")
    for i, trade in enumerate(selected, 1):
        sym      = trade["symbol"]
        sig      = trade["signal_type"][:2]   # BO / RE
        result   = trade["result"][:2]         # WI / LO / TI
        date_str = trade["df"].index[trade["signal_idx"]].strftime("%Y%m%d")
        pnl      = f"{trade['pnl_pct']:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        filename = f"{i:02d}_{sym}_{sig}_{result}_{date_str}_{pnl}pct.png"
        out_path = os.path.join(BACKTEST_DIR, filename)

        draw_chart(trade["df"], trade, sym, out_path)
        print(f"  [{i:02d}] {filename}")

    print(f"\nHoàn tất. {len(selected)} charts trong {BACKTEST_DIR}/")


if __name__ == "__main__":
    main()
