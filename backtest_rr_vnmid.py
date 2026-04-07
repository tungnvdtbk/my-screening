"""
backtest_rr_vnmid.py — Same 5%/15% R:R backtest as backtest_rr_vn30.py
but run on the 70 VNMID stocks (outside VN30).

Output:
  data/backtest/rr_vnmid_details.csv
  data/backtest/rr_vnmid_summary.csv
  data/backtest/rr_vnmid_report.txt
"""

# ── reuse all stubs and helpers from the VN30 backtest ───────────────────────
import backtest_rr_vn30 as bt
import app
import pandas as pd
import numpy as np
import os

OUT_DIR = "./data/backtest"
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = app.VNMID_STOCKS

all_rows = []

print(f"\n{'='*72}")
print(f"  VNMID R:R BACKTEST  |  Risk {bt.RISK_PCT*100:.0f}%  Reward {bt.REWARD_PCT*100:.0f}%"
      f"  Max hold {bt.MAX_HOLD}d  |  {bt.HISTORY} history")
print(f"  Symbols: {len(SYMBOLS)} VNMID stocks (outside VN30)")
print(f"{'='*72}\n")

for sym, sector in SYMBOLS.items():
    ticker = sym.replace(".VN", "")
    print(f"  {ticker:<6}…", end=" ", flush=True)
    df = bt.fetch(sym)
    if df is None:
        print("skip")
        continue
    count = 0
    for pname, fn, pcol, gcol in bt.DETECTORS:
        for cname, cfg_params in bt.CONFIGS.items():
            params = cfg_params[pname]
            rows = bt.run_config(df, sym, pname, fn, pcol, gcol, params, cname)
            all_rows.extend(rows)
            count += len(rows)
    print(f"{len(df)}b  {count} signals")

print(f"\n  Total signals: {len(all_rows)}\n")

# ── save details ──────────────────────────────────────────────────────────────
df_det = pd.DataFrame(all_rows)
df_det.to_csv(os.path.join(OUT_DIR, "rr_vnmid_details.csv"), index=False)

# ── report ────────────────────────────────────────────────────────────────────
lines = []
lines.append(f"\n{'='*72}")
lines.append(f"  WIN RATE BY PATTERN × CONFIG  "
             f"(Win = +{bt.REWARD_PCT*100:.0f}% before −{bt.RISK_PCT*100:.0f}%)")
lines.append(f"  Universe: {len(SYMBOLS)} VNMID stocks")
lines.append(f"{'='*72}")

summary_rows = []
for pname in ["Flag Pullback", "Pennant Pullback", "Triangle Pullback"]:
    lines.append(f"\n  {pname}")
    lines.append(f"  {'Config':<16} {'Signals':>8} {'Decided':>8} "
                 f"{'Win%':>7} {'AvgRet':>8} {'AvgMAE':>8} {'AvgDays':>8}")
    lines.append(f"  {'-'*65}")
    for cname in bt.CONFIGS:
        rows = [r for r in all_rows if r["pattern"] == pname and r["config"] == cname]
        s    = bt.summarise(rows)
        wr   = f"{s['win_rate']:.1f}%" if s["win_rate"] is not None else "n/a"
        ar   = f"{s['avg_ret']:+.2f}%" if s["avg_ret"]  is not None else "n/a"
        mae  = f"{s['avg_mae']:.2f}%"  if s["avg_mae"]  is not None else "n/a"
        days = f"{s['avg_days']:.1f}"  if s["avg_days"] is not None else "n/a"
        flag = ("✓" if (s["win_rate"] or 0) >= 50
                else "?" if (s["win_rate"] or 0) >= 35 else "✗")
        lines.append(f"  {cname:<16} {s['signals']:>8} {s['decided']:>8} "
                     f"{wr:>7} {ar:>8} {mae:>8} {days:>8}  {flag}")
        summary_rows.append({"pattern": pname, "config": cname, **s})

lines.append(f"\n{'='*72}")
lines.append("  IMPROVEMENT: combined vs baseline")
lines.append(f"{'='*72}")
for pname in ["Flag Pullback", "Pennant Pullback", "Triangle Pullback"]:
    base = bt.summarise([r for r in all_rows if r["pattern"] == pname and r["config"] == "baseline"])
    comb = bt.summarise([r for r in all_rows if r["pattern"] == pname and r["config"] == "combined"])
    if base["win_rate"] and comb["win_rate"]:
        delta = comb["win_rate"] - base["win_rate"]
        lines.append(f"  {pname:<22} {base['win_rate']:.1f}% → {comb['win_rate']:.1f}%"
                     f"  ({delta:+.1f}pp)   signals {base['signals']} → {comb['signals']}")

be = 1 / (1 + bt.REWARD_PCT / bt.RISK_PCT) * 100
lines.append(f"\n  Breakeven win rate at {bt.RISK_PCT*100:.0f}%/{bt.REWARD_PCT*100:.0f}% R:R = {be:.1f}%\n")

report = "\n".join(lines)
print(report)

rpt_path = os.path.join(OUT_DIR, "rr_vnmid_report.txt")
with open(rpt_path, "w") as f:
    f.write(report)

pd.DataFrame(summary_rows).to_csv(
    os.path.join(OUT_DIR, "rr_vnmid_summary.csv"), index=False)

print(f"  Details → {os.path.join(OUT_DIR, 'rr_vnmid_details.csv')}")
print(f"  Summary → {os.path.join(OUT_DIR, 'rr_vnmid_summary.csv')}")
print(f"  Report  → {rpt_path}\n")
