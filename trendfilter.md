# Trend Filter — Scan Rules (Implementation)

Two signal types: **TF_MA20** (shallow pullback) and **TF_MA50** (deep pullback).

---

## Shared Rules (both signals)

```
Low > MA200
Close > MA200
MA20 > MA50 > MA200
MA20 > MA20[5]
MA50 > MA50[5]
MA200 > MA200[20]
Close > Open
(MA20 - MA200) / MA200 <= 0.12
(Close - MA20) / MA20 <= 0.05
(Close - MA50) / MA50 <= 0.08
AvgVolume(20) >= 100,000
(High - Low) > 0.8 × ATR10
RS4W >= 0.95
```

---

## TF_MA20 — Shallow Pullback to MA20

Priority: checked first (higher quality signal).

```
Low <= MA20 × 1.03              (low tested MA20 within 3%)
Close > Highest(High, 10)[1]   (breaks above 10-day high)
Volume > 1.2 × AvgVolume(20)   (buying spike)
(MA50 - MA200) / MA200 <= 0.08 (MAs not over-extended)
```

- SL = min(Low, MA20) × 0.995
- TP = Close + 2 × ATR10

---

## TF_MA50 — Deep Pullback to MA50 (early uptrend only)

Checked only if TF_MA20 does not qualify.

```
Low <= MA50 × 1.03              (low tested MA50 within 3%)
Close > MA50                    (recovered above MA50)
(MA50 - MA200) / MA200 <= 0.05 (early uptrend — MAs close together)
Volume > 1.5 × AvgVolume(20)   (stronger buying required)
Close > Close[5]               (5-day upward momentum)
```

- SL = min(Low, MA50) × 0.995
- TP = Close + 2 × ATR10

---

## Notes

- Signal candle = last closed bar (`df.iloc[-1]`)
- All MAs computed on close prices (simple moving average)
- `HIGH10` = max(High[-2] … High[-11]) — excludes signal candle (no look-ahead)
- `MA20[5]` = MA20 from 5 bars ago; `MA200[20]` = MA200 from 20 bars ago
- `RS4W` = 21-day stock return / 21-day VNINDEX return (> 1.0 = outperforming)
- `ATR10` = average of True Range over bars [-11] to [-2]
- `AvgVolume(20)` = average volume over bars [-21] to [-2]
- Minimum 210 bars of history required

---

## Quality Tier (A / B) — Hard Gate

Only Tier A and Tier B signals pass. All others are rejected.

### Tier A (highest quality — target ~100% WR)
```
TF_MA20
AND vol_tier in (TIER1, TIER2)
AND weekly_ok == True
AND supply_overhead == False
AND R:R >= 2.0
AND risk_pct < 3.0%
```

### Tier B (good quality — target >50% WR with R:R >= 2:1)
```
TF_MA20 + vol_tier in (TIER1, TIER2) + R:R >= 2.0 + risk < 5%
OR
TF_MA50 + vol_tier in (TIER1, TIER2) + weekly_ok + R:R >= 2.0 + risk < 3%
```

### Rejection
```
if not (Tier A or Tier B):
    reject signal
```
