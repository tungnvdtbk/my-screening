# Swing Scanner — Professional Specification (Production-Ready, VN Stocks)

> This document is written as an **engineering-grade specification** for direct implementation into a Python-based scanning system. All rules are deterministic, vectorizable, and free from look-ahead bias when implemented correctly.

---

## 1) Objective
This scanner is designed for **short-to-medium term swing trading in the Vietnam stock market**, focused on:
- stocks already in an **established uptrend**,
- forming a **constructive buildup near support**,
- then showing a **high-quality breakout with volume expansion**,
- while preserving **acceptable risk/reward** and avoiding **overextended entries**.

The system is intended for **daily bars**, one DataFrame per ticker, sorted in **ascending date order**.

---

## 2) Technical Stack
- Python
- pandas >= 2.0
- numpy
- scipy
- All calculations must be **vectorized**.
- No Python loops over rows.
- Group-level ticker iteration is allowed only at orchestration level, not row logic.

---

## 3) Data Requirements
Per ticker DataFrame must include at minimum:
- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `ticker`

All prices must be adjusted consistently if adjusted data is used.

---

## 4) Data Quality Gate
Run before indicator calculation.

```python
Rules:
- Drop tickers with > 5% missing rows in required OHLCV fields.
- Forward-fill close for at most 2 consecutive NaN values.
- Do not forward-fill high, low, or volume blindly.
- Drop rows where volume <= 0.
- Drop rows where high < low.
- Drop rows where close <= 0.
- Minimum 60 valid rows per ticker after cleaning.
```

If ticker fails data quality gate, skip ticker entirely.

---

## 5) Indicators

```python
ma20         = close.rolling(20).mean()
ma50         = close.rolling(50).mean()
vol_ma20     = volume.rolling(20).mean()
vol_ma5      = volume.rolling(5).mean()
ret_5d       = close.pct_change(5)
ret_2d       = close.pct_change(2)
volatility10 = close.pct_change().rolling(10).std()
distance_ma20 = close / ma20
volume_spike  = volume / vol_ma20
value          = close * volume
high_20d      = high.rolling(20).max()
body_ratio    = abs(close - open) / (high - low).replace(0, 1e-9)
range_pct     = (high - low) / close.replace(0, 1e-9)
```

### RSI(14) — Wilder Smoothing
```python
delta = close.diff()
gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
rs    = gain / loss.replace(0, 1e-9)
rsi   = 100 - (100 / (1 + rs))
```

### Momentum Acceleration
Avoid unstable ratio explosions from very small denominators.

```python
mom_accel = (ret_2d - (ret_5d / 2)).clip(-0.30, 0.30)
```

---

## 6) Market Regime Gate
Run before all ticker filters. If market regime is invalid, return empty result set.

### Input
VNINDEX daily DataFrame with ascending date order.

### Indicators
```python
vn_ma20 = vn_close.rolling(20).mean()
vn_ma50 = vn_close.rolling(50).mean()
vn_ret_3d = vn_close.pct_change(3)
```

### Rules — all must be True on latest bar
```python
last.vn_ma20 > last.vn_ma50
last.vn_close > last.vn_ma50 * 0.97
last.vn_ret_3d > -0.03
```

If any condition is False, do not scan longs.

---

## 7) Relative Strength Filter
A stock must outperform the market over the intermediate window.

```python
stock_perf_20 = close / close.shift(20)
index_perf_20 = vn_close / vn_close.shift(20)
rs_vs_vni_20  = stock_perf_20 / index_perf_20
```

### Minimum requirement
```python
last.rs_vs_vni_20 > 1.0
```

This is a hard filter, not just a scoring input.

---

## 8) Per-Stock Hard Filters
Apply in fail-fast order. Skip ticker on first failure.

```python
1. value > 1e10
2. close > ma50
3. ma20 > ma50
4. close > ma50 * 0.97
5. close <= ma20 * 1.05
6. close <= ma50 * 1.15
7. rsi < 72
8. rs_vs_vni_20 > 1.0
9. is_buildup == True
10. buildup_days >= 3
11. entry_confirmed == True
12. rr_ratio >= 1.5
```

### Notes
- `value > 1e10` is preferred over weaker liquidity thresholds for more professional execution quality.
- Filters 5 and 6 prevent chasing extended names.
- `rr_ratio` must be computed using the actual entry bar and stop reference.

---

## 9) Buildup Rules
The buildup must be **tight, constructive, low-distribution, and near dynamic support**.

### Components
```python
tightness        = close.rolling(8).std() / close.replace(0, 1e-9)
vol_dryup        = vol_ma5 < vol_ma20 * 0.80
near_ma20        = close.between(ma20 * 0.99, ma20 * 1.02)
higher_low       = low > low.shift(3)
small_body       = body_ratio.rolling(5).mean() < 0.50
range_contract   = range_pct.rolling(5).mean() < range_pct.rolling(15).mean()

down_day         = close < open
distribution_day = down_day & (volume > vol_ma20 * 1.20)
no_distribution  = distribution_day.rolling(5).sum() <= 1
```

### Buildup Score
```python
buildup_score = (
    (tightness < 0.025).astype(int) +
    vol_dryup.astype(int) +
    near_ma20.astype(int) +
    higher_low.astype(int) +
    small_body.astype(int) +
    range_contract.astype(int) +
    no_distribution.astype(int)
)
```

### Buildup Definition
```python
is_buildup = (
    (tightness < 0.025) &
    near_ma20 &
    no_distribution &
    (buildup_score >= 4)
)
```

### Persistence Requirement
```python
buildup_days = is_buildup.rolling(5).sum()
```

Latest bar must satisfy:
```python
last.is_buildup == True
last.buildup_days >= 3
```

This ensures the pattern is established over time, not just a one-bar coincidence.

---

## 10) Entry Trigger
Evaluate on the latest bar only.

### Pivot Reference
Use a more meaningful pivot than 3 bars.

```python
pivot_high = high.shift(1).rolling(10).max()
price_break = last.close > last.pivot_high
```

Optional stricter breakout filter:
```python
price_break = last.close > last.pivot_high * 1.005
```

### Confirmation Conditions
```python
vol_expand = last.volume > last.vol_ma20 * 1.5
strong_bar = ((last.close - last.low) / max(last.high - last.low, 1e-9)) > 0.5
mom_ok     = last.mom_accel > 0
```

### Trigger Score
```python
trigger_score = int(price_break) + int(vol_expand) + int(strong_bar) + int(mom_ok)
```

### Entry Confirmed
```python
entry_confirmed = price_break and vol_expand and strong_bar
```

### Minimum Viable Entry
```python
buildup_score >= 4
buildup_days >= 3
entry_confirmed == True
rr_ratio >= 1.5
```

---

## 11) Entry Price and Exit Levels
Compute only after ticker passes all hard filters.

### Entry
```python
entry = last.close
```

### Base Risk Levels
```python
stop_loss     = last.ma50 * 0.97
trailing_stop = entry * 0.95
target_1      = entry * 1.07
target_2      = entry * 1.12
rr_ratio      = (target_1 - entry) / max(entry - stop_loss, 1e-9)
```

### Exit Logic
```python
Day 1-3:
  if close < stop_loss -> exit all

Day 4+:
  if high >= target_1 -> sell 50%, move stop to breakeven on remaining position

Day 8+:
  if close < target_1 and target_1 has never been touched -> exit all

Day 15:
  hard time-stop -> exit remaining position at close
```

### Optional advanced trailing rule for remaining half
For stronger trend capture, the remaining 50% can use:

```python
trail_ref = low.rolling(3).min()
exit_rest = close < trail_ref.shift(1)
```

If advanced trailing is enabled, document it clearly and do not mix with the fixed `target_2` logic silently.

---

## 12) Cross-Sectional Scoring
Only candidates that pass all hard filters enter the scoring stage.

### Percentile Rank Definition
```python
pct_rank(x) = rankdata(x.fillna(0)) / len(x)
```

### Ranked Features
```python
r_momentum   = pct_rank(ret_5d)
r_vol_spike  = pct_rank(volume_spike)
r_ma20_dist  = pct_rank(-(distance_ma20 - 1.0).clip(lower=0))
r_tightness  = pct_rank(-tightness)
r_buildup    = pct_rank(buildup_score)
r_trigger    = pct_rank(trigger_score)
r_rs         = pct_rank(rs_vs_vni_20)
```

### Score Formula
```python
score = (
    0.20 * r_momentum  +
    0.10 * r_vol_spike +
    0.10 * r_ma20_dist +
    0.15 * r_tightness +
    0.20 * r_buildup   +
    0.15 * r_trigger   +
    0.10 * r_rs
)
```

### Scoring Notes
- Momentum matters, but should not overpower setup quality.
- Tightness and buildup quality deserve high weight.
- Relative strength must influence ranking, not just gating.
- Overextension above MA20 should be penalized.

---

## 13) Final Output
Sort by `score` descending and return top 10.

### Required Columns
```python
ticker
date
close
score
rsi
tightness
near_ma20
buildup_score
buildup_days
vol_dryup
range_contract
no_distribution
rs_vs_vni_20
price_break
vol_expand
strong_bar
mom_accel
trigger_score
entry_confirmed
stop_loss
target_1
target_2
rr_ratio
```

Recommended additional columns:
```python
ma20
ma50
value
volume_spike
distance_ma20
pivot_high
```

---

## 14) Implementation Safety Rules
To avoid hidden bugs and false backtests:

```python
- Never use future bars in any rolling calculation.
- All entry decisions must be based on latest fully closed bar only.
- pivot_high must use high.shift(1), not current high.
- Do not compute score using rows that failed hard filters.
- Do not rank each ticker internally; rank candidates cross-sectionally across the same scan date.
- Protect all divisions with epsilon where denominator can be zero.
- Ensure VNINDEX series is aligned by date before rs_vs_vni_20 calculation.
```

---

## 15) Philosophy of This Scanner
This scanner is not trying to buy the fastest stock.
It is trying to buy the **cleanest constructive continuation setup** with:
- trend alignment,
- support proximity,
- controlled volatility,
- low distribution,
- real relative strength,
- and breakout confirmation.

The goal is not maximum signal count.
The goal is **higher-quality candidates with cleaner execution and more stable expectancy**.

---

## 16) Summary of Minimum Acceptable Trade Setup
A ticker is eligible only if all are true:

```python
market_regime_ok == True
value > 1e10
close > ma50
ma20 > ma50
close <= ma20 * 1.05
close <= ma50 * 1.15
rsi < 72
rs_vs_vni_20 > 1.0
is_buildup == True
buildup_days >= 3
entry_confirmed == True
rr_ratio >= 1.5
```

If multiple tickers qualify, prioritize by `score` descending.

---

## 17) Quality Tier (A / B) — Hard Gate

Only Tier A and Tier B signals pass. All others are rejected.

### Tier A (target ~100% WR)
```python
is_buildup == True
AND risk_pct < 2.5%
AND range_contract == True
AND vol_dryup == True
AND higher_low == True
AND rr_ratio >= 2.0
```
Tightest buildup: low risk + contraction + volume dryup + higher lows.

### Tier B (target >50% WR, R:R >= 2:1)
```python
is_buildup == True
AND risk_pct < 4.0%
AND range_contract == True
AND rr_ratio >= 2.0
```

### Rejection
```python
if not (Tier A or Tier B):
    return None
```

---

## 18) Recommended Next Step
After implementation, validate this rule in three separate market environments:
1. strong uptrend market,
2. sideways market,
3. corrective market.

Do not trust aggregate backtests alone.
This rule is regime-sensitive by design.


