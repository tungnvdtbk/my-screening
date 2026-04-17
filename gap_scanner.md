# Gap-Up Breakout Scanner — Institutional Conviction

> Extracted from `app.py:scan_gap()`. This spec documents the Gap-Up Breakout scanner for the Vietnam stock market.

---

## 1. Concept

A gap-up (open significantly above prior close) that holds and breaks the 10-day high signals **institutional demand**. The gap itself is a statement: buyers are willing to pay a premium at the open. If the gap holds throughout the day (no fade), it confirms conviction.

---

## 2. Input Data

Same OHLCV DataFrame as other scanners. Minimum 60 bars history.

---

## 3. Indicators

```python
gap_pct      = (Open - Close.shift(1)) / Close.shift(1)

# Shared indicators from compute_indicators():
atr10        = ATR(10) on bars [-11] to [-2]
avg_vol20    = volume avg on bars [-21] to [-2]
avg_vol_pre5 = volume avg on bars [-6] to [-2]
high10       = max(High) on bars [-11] to [-2]
high20       = max(High) on bars [-21] to [-2]
ma50         = SMA(Close, 50)
ma50_prev5   = ma50.shift(5)
```

---

## 4. Signal Conditions

All must be True on the signal candle (df.iloc[-1]):

```python
[1] UPTREND      close > ma50 AND ma50 > ma50_prev5
[2] GAP_UP       gap_pct >= 0.005   (open >= 0.5% above prior close)
[3] GAP_HOLDS    close > prev_close  (gap not fully faded)
[4] BULL         close > open
[5] BREAKOUT     high > high10      (breaks above 10-day high)
[X1] EXTENSION   close <= ma50 * 1.08
```

---

## 5. Signal Types

```python
GAP_STRONG = high > high20   (gap breaks 20-day high)
GAP_EARLY  = high > high10 but NOT > high20
```

---

## 6. Entry / SL / TP

```python
# Scan-time proxy (for filtering/ranking on the latest closed bar)
entry_plan    = close * 1.001
sl            = low
tp_plan       = entry_plan + 2.0 * atr10
rr_plan       = (tp_plan - entry_plan) / max(entry_plan - sl, 1e-9)
risk_pct_plan = (entry_plan - sl) / max(entry_plan, 1e-9)

# Live execution
entry_exec    = open[next_bar] * 1.001
tp_exec       = entry_exec + 2.0 * atr10
rr_exec       = (tp_exec - entry_exec) / max(entry_exec - sl, 1e-9)
risk_pct_exec = (entry_exec - sl) / max(entry_exec, 1e-9)
```

---

## 7. Volume Tier

```python
TIER1 = vol > 2.0x avg_vol20 AND avg_vol_pre5 < 0.75x avg_vol20
TIER2 = vol > 2.0x avg_vol20
TIER3 = neither
```

---

## 8. Output Columns

```python
signal          # GAP_STRONG or GAP_EARLY
date, close, gap_pct, high10, high20, atr10, ma50
entry_plan, sl, tp_plan, rr_plan, risk_pct_plan
vol_tier, weekly_ok, supply_overhead
volume, avg_vol20
```

---

## 9. Quality Tier (A / B) — Hard Gate

Only Tier A and Tier B signals pass. All others are rejected.

### Shared quality fields (define explicitly)
```python
weekly_ok = (weekly_close > weekly_ma20) and (weekly_ma20 > weekly_ma50)
supply_overhead = nearest_resistance_atr <= 1.0
```

### Tier A (highest quality)
```python
signal == "GAP_STRONG"
AND vol_tier == "TIER1"
AND weekly_ok == True
AND supply_overhead == False
AND rr_plan >= 2.0
```

### Tier B (tradeable quality, R:R >= 2:1)
```python
signal == "GAP_STRONG"
AND vol_tier in ("TIER1", "TIER2")
AND rr_plan >= 2.0
AND risk_pct_plan < 5.0%
```

### Rejection
```python
if not (Tier A or Tier B):
    return None   # signal rejected
```

At execution time, recheck with actual open:

```python
if rr_exec < 2.0 or risk_pct_exec >= 0.05:
    skip_entry
```
