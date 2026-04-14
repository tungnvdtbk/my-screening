# Mean Reversion Range Scanner — Buy Near Support in Sideways Range

> Extracted from `app.py:scan_mean_reversion()`. This spec documents the Mean Reversion scanner for the Vietnam stock market.

---

## 1. Concept

Find stocks trading in a **stable horizontal range** (not trending up or down), currently near the **bottom support zone**, showing a **reversal signal**. The thesis: price oscillates within the range, so buying near support and selling near resistance offers a repeatable edge.

This is a **long-only range swing** setup — buy near support, target resistance.

---

## 2. Input Data

Same OHLCV DataFrame. Minimum 80 bars history.

---

## 3. Configuration

```python
MR_CONFIG = {
    "range_window":          50,     # bars to compute range
    "ema_fast":              20,
    "ema_slow":              50,
    "volume_window":         20,
    "atr_window":            14,
    "rsi_window":             3,     # short-term RSI for oversold detection
    "slope_lookback":         5,
    "support_tolerance":   0.03,     # 3% tolerance for range boundary touches
    "bottom_zone_threshold": 0.25,   # bottom 25% of range = near support
    "min_avg_volume":    100_000,
    "min_avg_traded_value": 1_000_000_000,   # 1B VND/day
    "min_range_pct":        0.08,    # range size at least 8%
    "max_range_pct":        0.30,    # range size at most 30%
    "max_abs_ema20_slope":  0.02,    # EMA20 must be flat (not trending)
    "max_abs_ema50_slope": 0.015,    # EMA50 must be flat
    "max_10bar_drop":      -0.10,    # reject if 10-bar drop > 10%
    "min_history":           80,
}
```

---

## 4. Indicators

```python
ema20             = EMA(Close, 20)
ema50             = EMA(Close, 50)
atr14             = ATR(14, Wilder smoothing)
rsi3              = RSI(3) — short-term oscillator for oversold
avg_vol20         = volume.rolling(20).mean()
avg_traded_val20  = (volume * close).rolling(20).mean()

# 50-bar rolling range
range_high        = High.rolling(50).max()
range_low         = Low.rolling(50).min()
range_size_pct    = (range_high - range_low) / range_low
position_in_range = (Close - range_low) / (range_high - range_low)
dist_to_support   = (Close - range_low) / range_low

# EMA slopes (5-bar lookback)
ema20_slope       = (ema20 - ema20.shift(5)) / ema20.shift(5)
ema50_slope       = (ema50 - ema50.shift(5)) / ema50.shift(5)
```

---

## 5. Range Validation

All must be True:

```python
[1] RANGE_SIZE      8% <= range_size_pct <= 30%
[2] FLAT_EMA20      abs(ema20_slope) <= 2%
[3] FLAT_EMA50      abs(ema50_slope) <= 1.5%
[4] NOT_BREAKOUT    close < range_high
[5] BOUNDARY_TESTS  at least 2 touches of range_high AND 2 touches of range_low
                    in last range_window bars (within support_tolerance)
```

---

## 6. Near Support

```python
position_in_range <= 0.25   (bottom 25% of range)
close > range_low * 0.97    (not a breakdown below range)
```

---

## 7. Reversal Detection

Check for the first matching pattern on the last bar:

```python
1. CLOSE_ABOVE_PREV_HIGH   close > prev_high
2. HIGHER_CLOSE_LOWER_LOW  close > prev_close AND low <= prev_low
3. HAMMER                  lower_wick > body AND close >= midpoint
4. TWO_HIGHER_CLOSES       close > prev_close > prev2_close
5. RSI3_BOUNCE             rsi3 < 20 AND close > prev_close
```

Priority order: first match wins. At least one must be present.

---

## 8. Rejection Filters

```python
# Strong downtrend
if close < ema50 AND ema20 < ema50 AND ema50_slope < -2%:
    reject

# 10-bar drop too deep (> 10%)
if 10-bar return < -10%:
    reject

# 3 consecutive large bearish candles
if 3 bars in a row: close < open AND body > 0.5 * atr14:
    reject
```

---

## 9. Scoring

```python
support_score  = 1.0 - (position_in_range / 0.25)   # closer to support = higher
reversal_score = {CLOSE_ABOVE_PREV_HIGH: 1.0, HAMMER: 0.9, RSI3_BOUNCE: 0.85,
                  HIGHER_CLOSE_LOWER_LOW: 0.75, TWO_HIGHER_CLOSES: 0.70}
range_score    = flatness of EMAs (flatter = better)
rsi_score      = more oversold RSI3 = higher
liq_score      = log10(avg_traded_value) / 13

final_score = 0.30 * support + 0.30 * reversal + 0.20 * range + 0.10 * rsi + 0.10 * liq
```

---

## 10. Output Columns

```python
signal             # MR_LONG
date, close
range_high, range_low, range_size_pct, position_in_range, dist_support_pct
ema20, ema50, ema20_slope, ema50_slope
rsi3, atr14, avg_vol20
reversal_signal    # which pattern triggered
final_score
score_support, score_reversal, score_range
rs4w, vol_tier
```

---

## 11. Quality Tier (A / B) — Hard Gate

Only Tier A and Tier B signals pass. All others are rejected.

### SL / TP for tier evaluation
```python
SL    = range_low * 0.98
TP    = range_high
R:R   = (TP - entry) / (entry - SL)
```

### Tier A (highest quality — target ~100% WR)
```python
reversal in ("CLOSE_ABOVE_PREV_HIGH", "HAMMER")
AND position_in_range < 0.15
AND rsi3 < 20
AND rr >= 2.0
```

### Tier B (good quality — target >50% WR with R:R >= 2:1)
```python
reversal in ("CLOSE_ABOVE_PREV_HIGH", "HAMMER", "HIGHER_CLOSE_LOWER_LOW")
AND position_in_range < 0.25
AND rr >= 2.0
```

### Rejection
```python
if not (Tier A or Tier B):
    return None   # signal rejected
```
