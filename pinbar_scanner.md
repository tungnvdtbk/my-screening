# Pin Bar at Context Scanner — Price Action Rejection Setup

> Replaces the Watchlist (Developing Setup Scorer). This scanner detects **bullish pin bars at meaningful support levels** — the highest-probability price action reversal pattern.

---

## 1. Concept

A **pin bar** (pinocchio bar) is a single-candle reversal pattern with a long wick that "rejects" a price level. The long lower wick shows that sellers pushed price down but buyers absorbed all supply and closed near the high.

**Critical rule: A pin bar alone is noise. A pin bar AT CONTEXT is a signal.**

Context = a confluent support level where other traders have orders waiting. Without context, pin bars have no edge. This scanner only fires when the pin bar occurs at one or more of these levels:

| Context Level | Why it matters |
|---|---|
| MA20 | Pullback to short-term trend support — institutional buy zone in strong uptrends |
| MA50 | Deeper pullback support — major trend line, swing traders' favorite |
| MA200 | Long-term support — "line in the sand" for the primary trend |
| Swing Low (20-bar) | Structure support — prior price memory, obvious horizontal level |

---

## 2. Input Data

Same OHLCV DataFrame as other scanners. Minimum 60 bars history.

---

## 3. Indicators

```python
# Shared indicators from compute_indicators():
atr10        = ATR(10) on bars [-11] to [-2]
avg_vol20    = volume avg on bars [-21] to [-2]
avg_vol_pre5 = volume avg on bars [-6] to [-2]
ma20         = SMA(Close, 20)
ma50         = SMA(Close, 50)
ma200        = SMA(Close, 200)
ma50_prev5   = ma50.shift(5)

# Pin Bar specific (computed on signal candle):
body         = abs(close - open)
candle_range = high - low
upper_wick   = high - max(open, close)
lower_wick   = min(open, close) - low

# Support levels:
swing_low20  = min(Low) on bars [-21] to [-2]   # reuse: Low.shift(2).rolling(20).min()
```

---

## 4. Pin Bar Detection (Signal Candle = df.iloc[-1])

### 4a. Bullish Pin Bar — Canonical Form

All conditions must be True:

```python
[1] LONG_LOWER_WICK   lower_wick >= 0.60 * candle_range
                       # The tail is at least 60% of total range
                       # This IS the pin bar — the wick that "tests and rejects"

[2] SMALL_BODY        body <= 0.33 * candle_range
                       # Body is at most 1/3 of range
                       # Small body = indecision resolved in buyers' favor

[3] SHORT_UPPER_WICK  upper_wick <= 0.25 * candle_range
                       # Little to no upper wick
                       # Close near high = buyers in control at close

[4] MIN_SIZE           candle_range >= 0.5 * atr10
                       # Not a doji — must have meaningful range
                       # Too small = no conviction, likely noise
                       # Lower threshold than breakout (0.5x vs 1.5x) because
                       # pin bars are rejection patterns, not momentum patterns
```

### 4b. Bullish Close Preference (bonus, not required)

```python
[5] BULL_CLOSE         close > open
                       # Bullish close preferred but not mandatory
                       # A bearish-close pin bar (close < open but still near high
                       # of body) is valid if wick rejection is strong
                       # Impact: bull close → higher tier scoring
```

---

## 5. Context Conditions (AT LEAST ONE must be True)

The pin bar's **low** must have tested (wicked into) a support zone. "Tested" means the wick reached within proximity of the level.

```python
# Proximity threshold: within 0.5 * ATR10 of the level
# This accounts for different price scales across stocks
proximity = 0.5 * atr10

[C1] AT_MA20     abs(low - ma20) <= proximity
                  AND close > ma20
                  AND ma50 > ma50_prev5          # MA50 rising = uptrend intact
                  # Pin bar tested MA20 and bounced → pullback buy in uptrend
                  # Require uptrend: only buy MA20 pullback when trend is up

[C2] AT_MA50     abs(low - ma50) <= proximity
                  AND close > ma50 * 0.98        # closed at or above MA50
                  # Pin bar tested MA50 → deeper pullback, strong support
                  # No trend requirement: MA50 itself IS the trend line

[C3] AT_MA200    abs(low - ma200) <= proximity
                  AND close >= ma200 * 0.98       # didn't collapse below
                  # Pin bar tested MA200 → major long-term support
                  # Last defense line — high reward if it holds

[C4] AT_SWING    abs(low - swing_low20) <= proximity
                  AND close > swing_low20          # bounced above support
                  # Pin bar tested the 20-bar swing low → horizontal support
                  # Price memory: market remembers this level
```

### Context Confluence Scoring

```python
context_count = sum of [C1, C2, C3, C4] conditions that are True

# More confluent contexts = stronger signal
# e.g., pin bar AT MA50 AND AT swing_low20 = double support = highest quality
```

---

## 6. Safety Filters

```python
[X1] NOT_FREEFALL   close >= ma200 * 0.88
                     # Skip if price crashed >12% below MA200
                     # Catching falling knives in freefall is negative EV

[X2] NOT_EXTENDED   For MA20 context: close <= ma50 * 1.10
                     # Don't buy pullback to MA20 if price is 10%+ above MA50
                     # Over-extended = mean reversion risk dominates

[X3] MIN_RANGE      candle_range > 0
                     # Avoid division by zero on flat days
```

---

## 7. Signal Types

```python
PINBAR_MA20    = pin bar at MA20 in uptrend      [C1 matched]
PINBAR_MA50    = pin bar at MA50                  [C2 matched]
PINBAR_MA200   = pin bar at MA200                 [C3 matched]
PINBAR_SWING   = pin bar at swing low support     [C4 matched]

# If multiple contexts match, use priority: MA200 > MA50 > MA20 > SWING
# (deeper support = rarer = more significant)
# But store context_count for confluence scoring
```

---

## 8. Volume Conditions

```python
[V1] VOL_SPIKE       volume > 1.5 * avg_vol20
                      # Volume on pin bar day confirms participation
                      # Lower threshold than breakout (1.5x vs 2.0x):
                      # pin bars don't need climax volume, just above-average

[V2] VOL_CONTRACT    avg_vol_pre5 < avg_vol20 * 0.80
                      # Volume was declining before the pin bar
                      # Sellers exhausting → pin bar = capitulation

Volume tier:
  TIER 1 = [V1] + [V2]     # spike from quiet = best reversal signature
  TIER 2 = [V1] only        # spike without prior contraction = good
  TIER 3 = neither           # low conviction, still valid if context is strong
```

---

## 9. Entry / SL / TP

```python
# Status = PENDING (always) — require confirmation before entry
# Confirm: next candle closes above pin bar's high

Entry   = open of confirmation candle (or close if intraday)
          Slippage: entry * 1.001

SL      = low of pin bar candle
          # Below the wick = the level was rejected
          # If price returns below this, the rejection failed

TP1     = ma50  (if context is MA200 or SWING and price is below MA50)
          # Mean reversion to the intermediate trend

TP2     = entry + 2.0 * atr10
          # Minimum R:R = 2.0

TP      = max(TP1, TP2)    # take the higher target

R:R     = (TP - entry) / (entry - SL)
          # Must be >= 2.0 to pass — reject if not

risk_pct = (entry - SL) / entry * 100
          # Max risk: 7% — pin bar wicks can be long
```

---

## 10. Quality Tier (A / B) — Hard Gate

Only keep Tier A and Tier B. Signals that don't qualify are rejected.

```
Tier A (highest quality):
  context_count >= 2                     # multiple support levels confluent
  AND vol_tier in (TIER1, TIER2)         # volume confirms
  AND close > open                        # bullish close
  AND R:R >= 2.5                          # excellent reward
  AND risk_pct < 5%

Tier B (tradeable):
  context_count >= 1                     # at least one support level
  AND R:R >= 2.0
  AND risk_pct < 7%
  AND NOT (vol_tier == TIER3 AND context_count == 1)
  # Single context + no volume = too weak

If not Tier A or Tier B → reject signal
```

---

## 11. Status: PENDING / CONFIRMED

```python
# Pin bar signals ALWAYS start as PENDING
# Unlike breakout (which can be entered next open), pin bars need confirmation

When signal candle detected → status = PENDING

Next candle:
  if close[next] > high[signal]  → CONFIRMED → entry allowed
  if close[next] < low[signal]   → INVALID → reject signal
  else                           → still PENDING → wait one more day
```

---

## 12. Output Format

```python
{
    "signal":         "PINBAR_MA50",       # signal type
    "date":           date[-1],             # signal candle date
    "close":          close,
    "status":         "PENDING",            # always PENDING at scan time
    "pin_tier":       "A",                  # quality tier
    "context":        "MA50+SWING",         # which support levels matched
    "context_count":  2,                    # number of confluent supports
    "wick_ratio":     0.68,                 # lower_wick / candle_range
    "body_ratio":     0.18,                 # body / candle_range
    "sl":             low[-1],
    "tp":             max(tp1, tp2),
    "rr":             rr_ratio,
    "atr10":          atr10,
    "ma20":           ma20,
    "ma50":           ma50,
    "ma200":          ma200,
    "vol_tier":       "TIER1",
    "rs4w":           rs4w,
    "volume":         volume,
    "avg_vol20":      avg_vol20,
}
```

---

## 13. Implementation Notes

### What changes from current codebase

1. **Remove**: `score_developing()` function — the "WATCH" developing setup scorer
2. **Remove**: `_render_watchlist()` function
3. **Remove**: `scan_watchlist` from session state and all references
4. **Remove**: Watchlist tab from the UI tabs
5. **Add**: `scan_pinbar()` function in the main scan chain
6. **Add**: `"PINBAR_MA20"`, `"PINBAR_MA50"`, `"PINBAR_MA200"`, `"PINBAR_SWING"` to `_SIGNAL_PRIORITY`
7. **Add**: Pin Bar tab in UI (replaces Watchlist tab)
8. **Add**: `swing_low20` to `compute_indicators()` — `d["swing_low20"] = d["Low"].shift(2).rolling(20).min()`

### Scan chain order

```python
sig = (
    scan_breakout(df, vnindex_df)     or
    scan_gap(df, vnindex_df)           or
    scan_nr7(df, vnindex_df)           or
    scan_pinbar(df, vnindex_df)        or    # NEW — before trend_filter
    scan_trend_filter(df, vnindex_df)
)
```

Pin bar is placed after NR7 but before trend_filter because:
- Breakout / Gap / NR7 = momentum signals (higher priority)
- Pin bar = reversal/pullback signal (medium priority)
- Trend filter = catch-all (lowest priority)

### Indicators to add

```python
# In compute_indicators():
d["swing_low20"] = d["Low"].shift(2).rolling(20).min()    # [-21] to [-2]
```

---

## 14. Comparison with Other Scanners

|  | Pin Bar | Breakout | Reversal (removed) | Climax |
|---|---|---|---|---|
| Pattern type | Single candle rejection | Momentum breakout | Momentum reversal | Sell climax + reversal |
| Context required | YES — must be at support | NO | YES — near MA200 | YES — deep decline |
| Trend requirement | Depends on context | Uptrend | Downtrend | Downtrend |
| Volume requirement | Preferred, not required | Required (2x) | Required (2x) | Required |
| Body filter | Small body (<=33%) | Large body (>=60%) | Large body (>=60%) | Varies |
| Wick filter | Long lower wick (>=60%) | Close near high | Close near high | Pin bar OR marubozu |
| Confirmation | Required (PENDING) | Not required | Required (PENDING) | Required (PENDING) |
| R:R target | >= 2.0 | >= 2.0 | >= 2.0 | >= 2.0 |
| Hold time | 3–10 days | 5–15 days | 3–10 days | 3–10 days |

---

## 15. Why Pin Bar Replaces Watchlist

The old watchlist scored how close a stock was to triggering a breakout/reversal. Problems:
1. **Not actionable** — "almost breaking out" doesn't help the trader make a decision
2. **Noisy** — many stocks are always "close" to something
3. **No price action** — scored distance to levels, not what price did at those levels

Pin bar at context is the opposite:
1. **Actionable** — a pin bar IS a trade setup, with clear SL/TP/entry
2. **Filtered** — strict candle structure + mandatory support context
3. **Price action first** — the candle tells you what the market did, not what it might do
