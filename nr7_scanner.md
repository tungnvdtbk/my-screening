# NR7 Breakout Scanner — Narrow Range Coil Setup

> Extracted from `app.py:scan_nr7()`. This spec documents the NR7 (Narrow Range 7) breakout scanner for the Vietnam stock market.

---

## 1. Concept

NR7 = the signal candle has the **smallest range of the last 7 days**. This indicates price compression ("coiling"). When price then closes above the 10-day high, it signals a potential breakout from equilibrium.

Key edge: tight SL (narrow candle = small risk), good R:R.

---

## 2. Input Data

Same OHLCV DataFrame as other scanners. Minimum 60 bars history.

---

## 3. Indicators

```python
candle_range = High - Low
nr7          = candle_range <= candle_range.rolling(7).min()

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

## 4. NR7-Specific Quality Factors

### Inside Bar Detection
```python
is_inside_bar = High[-1] < High[-2] AND Low[-1] > Low[-2]
ib_chain      = count consecutive inside bars ending at signal candle
```
Inside bar = even tighter compression. NR7 + Inside Bar = strongest coil.

### Nearest Resistance (ATR units)
```python
resist_atr = distance to nearest swing-high resistance above signal high
             within 4 x ATR lookback of 60 bars
             Swing high: high[i] > high[i-1] AND high[i] > high[i+1]
```
Closer resistance = breakout has nearby target = higher conviction.

### Volume Quality on NR7 Day
```python
NR7 wants LOW volume (market in equilibrium, not contested):
  ratio = Volume / avg_vol20
  QUIET++  : ratio < 0.50   (perfect coil)
  QUIET    : ratio < 0.75   (good)
  NORMAL   : ratio < 1.00
  HIGH     : ratio >= 1.00  (contested — less ideal)
```

### Volume Declining
```python
vol_declining = avg_vol_pre5 < avg_vol20 * 0.80
```

---

## 5. Signal Conditions

All must be True on the signal candle (df.iloc[-1]):

```python
[1] UPTREND     close > ma50 AND ma50 > ma50_prev5
[2] NR7         candle is narrowest of last 7 days
[3] BULL        close > open
[4] BREAKOUT    high > high10 (breaks above 10-day high)
[5] UPPER_HALF  close >= (high + low) / 2
[X1] EXTENSION  close <= ma50 * 1.08
```

---

## 6. Rejection Filters

```python
# Reject HIGH volume on NR7 day
# High volume = contested candle, not quiet compression
# Historical WR when vol_quiet_lbl == "HIGH": 12%
if vol_quiet_lbl == "HIGH":
    reject

# Drop NR7_EARLY unless score >= 50
# NR7_EARLY (only breaks HIGH10, not HIGH20) historical WR = 17%
if signal_type == "NR7_EARLY" and nr7_score < 50:
    reject

# Minimum score gate
# score < 30 → 33% WR, not worth it
if nr7_score < 30:
    reject
```

---

## 7. Signal Types

```python
NR7_STRONG = high > high20   (breaks 20-day high — higher quality)
NR7_EARLY  = high > high10 but NOT > high20
```

---

## 8. NR7 Composite Score (0-100)

```python
nr7_score = 0
if is_inside_bar:                nr7_score += 25
if ib_chain >= 2:                nr7_score += 15
if resist_atr is not None:
    if resist_atr <= 1.0:        nr7_score += 25   # resistance very close
    elif resist_atr <= 2.0:      nr7_score += 15
    elif resist_atr <= 3.0:      nr7_score += 8
if vol_quiet_lbl == "QUIET++":   nr7_score += 25
elif vol_quiet_lbl == "QUIET":   nr7_score += 15
if vol_declining:                nr7_score += 10
```

Maximum = 100 (inside bar + 2+ chain + near resistance + very quiet vol + declining vol).

---

## 9. Entry / SL / TP

```python
Entry = close (signal candle)
SL    = low (signal candle)
TP    = entry + 2.0 * ATR10
R:R   = (TP - entry) / (entry - SL)
```

---

## 10. Volume Tier (same as Breakout)

```python
TIER1 = vol > 2.0x avg_vol20 AND avg_vol_pre5 < 0.75x avg_vol20
TIER2 = vol > 2.0x avg_vol20
TIER3 = neither
```

---

## 11. Output Columns

```python
signal          # NR7_STRONG or NR7_EARLY
date, close, high10, high20, atr10, ma50
is_inside_bar, ib_chain, resist_atr, vol_quiet
nr7_score       # composite quality score (0-100)
sl, tp, rr
vol_tier, vol_char, tight_days, weekly_ok, supply_overhead
rs4w, volume, avg_vol20
```

---

## 12. Quality Tier (A / B) — Hard Gate

Only Tier A and Tier B signals pass. All others are rejected.

### Tier A (highest quality — target ~100% WR)
```python
signal_type == "NR7_STRONG"
AND is_inside_bar == True
AND nr7_score >= 60
AND supply_overhead == False
AND rr >= 2.0
```

### Tier B (good quality — target >50% WR with R:R >= 2:1)
```python
signal_type == "NR7_STRONG"
AND nr7_score >= 40
AND rr >= 2.0
```

### Rejection
```python
if not (Tier A or Tier B):
    return None   # signal rejected
```
