# Price Action Scanner — Breakout & Pullback Rules (v2.1 — Vietnam Market)

> An implementation-ready Markdown specification for scanning **two core long setups** in a **daily-bar swing trading** workflow on the **Vietnam stock market (HOSE / HNX)**: **Breakout after buildup** and **Pullback continuation to MA20**.
>
> This version is intentionally **simple, objective, and production-oriented**. It does **not** attempt to encode every discretionary nuance from price action reading. It focuses on the minimum structure that remains robust enough for systematic scanning.
>
> **v2.1 additions**: VNINDEX market regime gate, relative strength filter, limit-up/limit-down detection, RSI overbought guard, exit strategy, VN-specific parameter tuning, Volman cluster barrier detection, Volman squeeze detection.

---

## 1. Objective

This scanner is designed to find stocks that:

- are already in a **healthy short-term uptrend**,
- show **constructive compression / buildup**,
- hold near a rising **MA20**,
- and are setting up for one of two continuation structures:
  - **Breakout after buildup**, or
  - **Pullback to MA20 with bullish resumption**.

This is a **long-only daily-bar swing scanner**.

---

## 2. Design Principles

The scanner follows four non-negotiable ideas:

1. **Trend first** — no trend, no setup.
2. **Structure second** — buildup or pullback must be constructive.
3. **Trigger third** — entry signal must show initiative buying.
4. **Do not chase** — reject bars that are too extended or too climactic.

The goal is not to scan every bullish chart.
The goal is to scan only **repeatable continuation structures**.

---

## 3. Input Data

Required columns per ticker:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `ticker`

Requirements:

- data sorted in ascending date order,
- one row per trading day,
- one DataFrame per ticker or grouped by ticker,
- all calculations vectorized,
- use only fully closed daily bars,
- minimum history: **at least 60 bars** before evaluating any setup.

---

## 3.5. Data Quality Gate

Run before indicator calculation. If ticker fails this gate, skip entirely.

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

### Why

VN market data (via yfinance / vnstock) contains frequent gaps, missing volume on half-day sessions, and corporate action artifacts. Without this gate, indicator calculations produce NaN cascades or false signals.

---

## 4. Core Indicators

```python
ma20        = close.rolling(20).mean()
ma50        = close.rolling(50).mean()
vol_ma20    = volume.rolling(20).mean()
vol_ma5     = volume.rolling(5).mean()
value       = close * volume

ma20_slope  = ma20 - ma20.shift(5)
ma50_slope  = ma50 - ma50.shift(10)

bar_range   = (high - low).replace(0, 1e-9)
range_pct   = (high - low) / close.replace(0, 1e-9)
body_ratio  = abs(close - open) / bar_range
upper_tail  = (high - close) / bar_range

pivot_10    = high.shift(1).rolling(10).max()
pivot_20    = high.shift(1).rolling(20).max()
low_3       = low.shift(1).rolling(3).min()
low_5       = low.shift(1).rolling(5).min()

ma20_distance   = close / ma20.replace(0, 1e-9)
range_avg20     = range_pct.rolling(20).mean()
range_expansion = range_pct / range_avg20.replace(0, 1e-9)

ret_5d          = close.pct_change(5)
ret_2d          = close.pct_change(2)
value_avg5      = value.rolling(5).mean()
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

Detects short-term momentum picking up relative to the 5-day trend.
Clipped to avoid unstable ratio explosions from tiny denominators.

```python
mom_accel = (ret_2d - (ret_5d / 2)).clip(-0.30, 0.30)
```

### Relative Strength vs VNINDEX

```python
# Requires VNINDEX daily close aligned by date
stock_perf_20 = close / close.shift(20)
index_perf_20 = vn_close / vn_close.shift(20)
rs_vs_vni     = stock_perf_20 / index_perf_20.replace(0, 1e-9)
```

### Barrier Cluster Detection (Volman)

A meaningful barrier is not just the highest high — it is a **level that has been tested multiple times**. Multiple touches at the same zone show that sellers defend that price. When buildup compresses into a cluster barrier, the eventual break carries more conviction.

```python
# Resistance zone = within 1.5% of the 20-bar pivot high
resistance_zone_lo = pivot_20 * 0.985

# Count how many of the last 20 bars had highs reaching into this zone
barrier_touches = high.shift(1).rolling(20).apply(
    lambda w: ((w >= w.max() * 0.985) & (w <= w.max())).sum(),
    raw=False
)

strong_barrier = barrier_touches >= 3   # 3+ tests = real resistance cluster
```

### Squeeze Detection (Volman)

A squeeze is price compressed between **converging levels**: rising support (higher lows) and flat or declining resistance (the pivot). This is Volman's highest-quality buildup — pressure building from both sides with nowhere to go except break.

```python
# Rising support: 5-bar low trough is climbing
support_rising = low_5 > low_5.shift(5)

# Flat or declining resistance: pivot_20 is not expanding
resist_flat = pivot_20 <= pivot_20.shift(5) * 1.005   # within 0.5% = flat

# Squeeze = support converging into resistance
is_squeeze = support_rising & resist_flat
```

---

## 5. Trend Filter

Only scan stocks already in a valid uptrend.

### Required conditions

```python
trend_filter = (
    (close > ma20) &
    (ma20 > ma50) &
    (ma20_slope > 0) &
    (ma50_slope >= 0)
)
```

### Interpretation

- `close > ma20` → price is above short-term value.
- `ma20 > ma50` → short-term trend is above medium-term trend.
- `ma20_slope > 0` → MA20 is rising.
- `ma50_slope >= 0` → medium trend is not deteriorating.

If `trend_filter` is `False`, skip the ticker.

---

## 5.5. Market Regime Gate — VNINDEX

Run **before** any per-stock scan. If the market regime is invalid, return an empty result set — do not scan longs.

### Input

VNINDEX daily DataFrame with ascending date order, aligned by date with stock data.

### Indicators

```python
vn_ma20  = vn_close.rolling(20).mean()
vn_ma50  = vn_close.rolling(50).mean()
vn_ret3d = vn_close.pct_change(3)
```

### Rules — all must be True on the latest bar

```python
market_ok = (
    (vn_close.iloc[-1] > vn_ma50.iloc[-1] * 0.97) &
    (vn_ma20.iloc[-1] > vn_ma50.iloc[-1]) &
    (vn_ret3d.iloc[-1] > -0.03)
)
```

### Interpretation

- `vn_close > vn_ma50 * 0.97` → index not in freefall (3% tolerance below MA50).
- `vn_ma20 > vn_ma50` → short-term index trend is above medium-term.
- `vn_ret3d > -0.03` → no sharp 3-day selloff in progress.

### Why

The VN market is ~80-90% correlated. When VNINDEX breaks down, almost all stocks fall together. Breakout signals fired during an index correction have near-zero win rate. This gate prevents destructive counter-trend entries.

If `market_ok` is `False`, return empty — do not scan longs.

---

## 6. Buildup Filter

Both setups must come **after buildup**.

Buildup means:

- price tightens,
- closes become more compressed,
- price stays near MA20,
- price remains close to the breakout area,
- down days do not show heavy distribution,
- sellers fail to push price materially lower.

### Components

```python
tightness       = close.rolling(8).std() / close.replace(0, 1e-9)
tight_closes    = close.rolling(5).std() / close.replace(0, 1e-9)
vol_dryup       = vol_ma5 < vol_ma20 * 0.80          # ← tightened from 0.90 for VN
near_ma20       = close.between(ma20 * 0.99, ma20 * 1.02)  # ← tightened from 1.03
higher_low      = low > low.shift(3)
small_body      = body_ratio.rolling(5).mean() < 0.55 # ← tightened from 0.60
range_contract  = range_pct.rolling(5).mean() < range_pct.rolling(15).mean()
below_pivot     = close <= pivot_20 * 1.02

# Distribution day: red bar with 1.2x+ volume = institutional selling
distribution_day = (close < open) & (volume > vol_ma20 * 1.20)  # ← was 1.0x
no_distribution  = distribution_day.rolling(5).sum() <= 1
```

### Buildup score

```python
buildup_score = (
    (tightness < 0.025).astype(int) +     # tight base
    (tight_closes < 0.020).astype(int) +   # compressed closes
    vol_dryup.astype(int) +                # volume drying up
    near_ma20.astype(int) +                # near dynamic support
    higher_low.astype(int) +               # no panic selling
    small_body.astype(int) +               # controlled bars
    range_contract.astype(int) +           # narrowing ranges
    below_pivot.astype(int) +              # at or below resistance
    no_distribution.astype(int) +          # no institutional selling
    strong_barrier.astype(int) +           # Volman: cluster resistance (3+ touches)
    is_squeeze.astype(int)                 # Volman: converging support + flat resistance
)
```

Total possible: **11 points** (was 9). The two Volman additions reward higher-quality barrier structure and squeeze pressure.

### Buildup definition

```python
is_buildup = (
    near_ma20 &
    below_pivot &
    no_distribution &                                    # hard requirement
    (tightness < 0.025) &                                # tight base
    (buildup_score >= 5)
)
```

Note: `buildup_score >= 5` out of 11 means roughly half the components must be present. The Volman additions (`strong_barrier`, `is_squeeze`) are **bonus points** — they improve ranking but are not hard requirements. A buildup can still qualify without a cluster barrier or squeeze, but setups that have them will score higher and rank better.

### Persistence requirement

```python
buildup_days = is_buildup.rolling(5).sum()
```

### Minimum requirement

```python
is_buildup == True
buildup_days >= 3
```

This ensures the setup comes from a **real base / pressure zone**, not from a random isolated green bar.

---

## 7. Shared Hard Filters

These filters apply before setup classification. Applied in fail-fast order — skip ticker on first failure.

```python
hard_filter = (
    market_ok &                    # ← NEW: VNINDEX regime gate (Section 5.5)
    trend_filter &
    (value_avg5 > 1e10) &          # ← rolling 5-day average, not single bar
    is_buildup &
    (buildup_days >= 3) &
    (close <= ma20 * 1.05) &
    (close <= ma50 * 1.15) &       # ← NEW: not too far above MA50
    (rsi < 72) &                   # ← NEW: overbought guard
    (rs_vs_vni > 1.0)              # ← NEW: must outperform VNINDEX
)
```

### Notes

- `value_avg5 > 1e10` — rolling 5-day average traded value (VND). Prevents single-bar spikes (block trades, ATC anomalies) from passing the liquidity gate. 10 billion VND is the minimum for professional execution in VN100.
- `close <= ma20 * 1.05` — avoids chasing bars too far above short-term mean.
- `close <= ma50 * 1.15` — prevents buying into parabolic extension above the medium-term trend.
- `rsi < 72` — VN retail-dominated market reverses hard from overbought. Stocks with RSI > 70 on the breakout bar frequently give back gains within 2-3 days.
- `rs_vs_vni > 1.0` — stock must outperform VNINDEX over 20 bars. Prevents buying laggards disguised as breakouts.
- If any hard filter fails, skip the ticker.

---

## 8. Setup A — Breakout After Buildup

### Concept (Volman: Buildup Break)

A breakout setup is valid when:

- trend is already up,
- price has built a **tight base near MA20**,
- price remains close to a **meaningful barrier** (ideally tested multiple times),
- buildup compresses into the barrier (**squeeze** = rising support + flat resistance),
- then price closes through that barrier,
- with enough volume and close quality to show **initiative buying**,
- without becoming climactic.

The highest-quality breakouts come from a **cluster barrier** (3+ touches at the same zone) with a **squeeze** underneath. These are Volman's core concepts: the more pressure that builds, the more decisive the eventual break.

### Trigger conditions

```python
breakout_level = pivot_20                                  # 20-bar high as barrier reference

price_break      = close > breakout_level
break_distance   = close / breakout_level.replace(0, 1e-9)
clean_break      = break_distance <= 1.03                  # VN ±7% limits allow bigger bars
vol_expand       = volume > vol_ma20 * 1.50                # 1.3x is noise in VN (ATC/blocks)
strong_close     = ((close - low) / bar_range) > 0.60
small_upper_tail = upper_tail < 0.35
acceptable_bar   = range_expansion <= 2.00                 # not climactic
mom_ok           = mom_accel > 0                           # momentum accelerating
```

### Trigger score (for ranking)

```python
trigger_score = (
    int(price_break) +
    int(vol_expand) +
    int(strong_close) +
    int(mom_ok) +
    int(strong_barrier) +    # Volman: cluster barrier bonus
    int(is_squeeze)          # Volman: squeeze bonus
)
```

Total possible: **6** (was 4). Higher trigger score = higher conviction break.

### Breakout setup definition

```python
setup_breakout = (
    hard_filter &
    price_break &
    clean_break &
    vol_expand &
    strong_close &
    small_upper_tail &
    acceptable_bar
    # not_extended removed — already enforced in hard_filter (close <= ma20 * 1.05)
)
```

Note: `strong_barrier` and `is_squeeze` are **not hard requirements** for the breakout trigger. They are scoring bonuses. A breakout can fire without a cluster barrier, but setups that break through a well-tested resistance zone with squeeze pressure will rank materially higher.

### Breakout intent

This setup is designed to capture:

- **pressure building under resistance** (buildup compressing into barrier),
- **converging support into flat resistance** (squeeze, when present),
- then a **decisive but not overly emotional break** through the barrier,
- with **real participation** (volume expansion),
- and **short-term momentum accelerating**.

---

## 9. Setup B — Pullback Continuation to MA20

### Concept

A pullback setup is valid when:

- trend is already up,
- there was a prior upward push,
- price retraces in a controlled way toward MA20,
- the pullback does not damage the broader structure,
- then price reclaims upward with a constructive bullish bar.

This is a **continuation setup**, not a reversal setup.

### Pullback structure filters

```python
prior_push         = close.shift(3) > close.shift(8)
pullback_depth     = (close.rolling(5).max() - low) / close.replace(0, 1e-9)
pullback_exists    = pullback_depth > 0.02
down_days_pullback = (close < close.shift(1)).rolling(4).sum() >= 1

pullback_to_ma20   = low <= ma20 * 1.02
close_holds_ma20   = close >= ma20 * 0.99
shallow_pullback   = close >= ma50
no_deep_damage     = low >= ma50 * 0.98
```

### Pullback trigger conditions

```python
bull_reclaim   = close > high.shift(1)
strong_close   = ((close - low) / bar_range) > 0.55
reversal_bar   = (close > open) & bull_reclaim & strong_close
vol_ok         = volume >= vol_ma20 * 1.00   # ← was 0.90; recovery bar needs at least avg volume
```

### Pullback setup definition

```python
setup_pullback = (
    hard_filter &
    prior_push &
    pullback_exists &
    down_days_pullback &
    pullback_to_ma20 &
    close_holds_ma20 &
    shallow_pullback &
    no_deep_damage &
    reversal_bar &
    vol_ok
)
```

### Pullback intent

This setup is designed to capture:

- a controlled pause into support,
- weak selling pressure,
- then renewed demand at or near MA20.

---

## 10. Setup Label

Each ticker on the latest closed bar can be classified as:

```python
if setup_breakout:
    setup_type = "breakout"
elif setup_pullback:
    setup_type = "pullback"
else:
    setup_type = None
```

If both are `True` on the same bar, prioritize:

```python
setup_type = "breakout"
```

Reason: breakout is the more decisive signal.

---

## 11. Ranking Logic

After filtering, candidates can be ranked by a simple score.

### Ranking features

```python
volume_spike      = volume / vol_ma20.replace(0, 1e-9)
extension_penalty = (ma20_distance - 1.0).clip(lower=0)
close_quality     = ((close - low) / bar_range)
```

### Cross-sectional score

Rank candidates against each other using percentile ranks. Only candidates that pass all hard filters enter this stage.

```python
score = (
    0.20 * ret_5d.rank(pct=True) +               # momentum (reduced from 0.30)
    0.10 * volume_spike.rank(pct=True) +          # participation (reduced from 0.25)
    0.15 * buildup_score.rank(pct=True) +         # setup quality
    0.15 * (-tightness).rank(pct=True) +          # NEW: tighter base = higher rank
    0.10 * close_quality.rank(pct=True) +         # bar quality
    0.10 * (-extension_penalty).rank(pct=True) +  # not stretched
    0.10 * rs_vs_vni.rank(pct=True) +             # NEW: relative strength vs VNINDEX
    0.10 * trigger_score.rank(pct=True)            # NEW: entry trigger quality
)
```

Sort by `score` descending and return **top 10**.

### Interpretation

Prefer stocks that:

- have stronger recent momentum (but not over-weighted),
- show meaningful participation,
- have tighter and more constructive buildup,
- close well within the bar,
- are not too stretched above MA20,
- outperform the market,
- and have stronger entry trigger confirmation.

### Scoring notes

- Momentum matters, but should not overpower setup quality.
- Tightness and buildup quality deserve high weight — these are the core edge.
- Relative strength must influence ranking, not just gating.
- Do not rank tickers internally; rank candidates cross-sectionally across the same scan date.

---

## 12. Exit Strategy

Compute only after a ticker passes all hard filters and is classified as breakout or pullback.

### Entry

```python
entry = open[signal + 1] * 1.001   # next-day open + 0.1% slippage
```

### Breakout exit levels

```python
stop_loss  = max(low[-1], ma20[-1] * 0.97)   # signal bar low or 3% below MA20
target_1   = entry * 1.07                      # 7% — partial exit (50%)
target_2   = entry * 1.12                      # 12% — remainder
rr_ratio   = (target_1 - entry) / max(entry - stop_loss, 1e-9)
```

### Pullback exit levels

```python
stop_loss  = max(low[-1], ma50[-1] * 0.98)   # signal bar low or 2% below MA50
target_1   = entry * 1.07                      # 7% — partial exit (50%)
target_2   = entry * 1.10                      # 10% — remainder
rr_ratio   = (target_1 - entry) / max(entry - stop_loss, 1e-9)
```

### RR gate

```python
rr_ratio >= 1.5   # hard requirement — skip if risk/reward is not acceptable
```

### Exit logic

```python
Day 1-3:
  if close < stop_loss -> exit all

Day 4+:
  if high >= target_1 -> sell 50%, move stop to breakeven on remaining

Day 8+:
  if close < target_1 and target_1 never touched -> exit all (setup failed)

Day 15:
  hard time-stop -> exit remaining at close
```

### T+2.5 settlement note

After buying on the Vietnam market, you cannot sell for ~2.5 trading days. This means:
- **Minimum effective hold is 3 days** — stop-loss on Day 1-2 is a mental note, not executable.
- Position sizing should assume the SL may only be hit on Day 3, not Day 1.
- Wide stops on Day 1 are acceptable if the setup quality justifies the risk.

---

## 13. Output Columns

### Required output fields

```python
ticker
date
close
setup_type
score
ma20
ma50
ma20_slope
ma50_slope
rsi
rs_vs_vni
buildup_score
buildup_days
is_buildup
no_distribution
tightness
strong_barrier       # Volman: cluster resistance (3+ touches)
barrier_touches      # Volman: number of touches at resistance zone
is_squeeze           # Volman: converging support + flat resistance
trigger_score
price_break
vol_expand
pullback_to_ma20
bull_reclaim
hard_filter
trend_filter
market_ok
stop_loss
target_1
target_2
rr_ratio
```

### Recommended extra fields

```python
volume
vol_ma20
value_avg5
body_ratio
range_pct
pivot_20
upper_tail
range_expansion
ma20_distance
mom_accel
volume_spike
```

These fields help debugging and rule validation.

---

## 14. Implementation Rules

To keep the scanner valid and backtest-safe:

```python
- Use only fully closed daily bars.
- No look-ahead logic.
- pivot_20 must use high.shift(1), never current high.
- All rolling calculations must be based only on past and current bars.
- Run setup detection on the latest closed bar only for live scan.
- Do not use intraday assumptions in daily-bar logic.
- All ratio calculations must defend against divide-by-zero.
- If required rolling history is insufficient, filters should evaluate to False.
- Ensure VNINDEX series is aligned by date before rs_vs_vni calculation.
- Do not compute score using rows that failed hard filters.
- Do not rank each ticker internally; rank candidates cross-sectionally.
```

### Limit-Up / Limit-Down Detection (HOSE ±7%)

HOSE enforces ±7% daily price limits (HNX ±10%, UPCoM ±15%). A breakout on a limit-up day is un-tradeable — you cannot buy at market next morning.

```python
ref_price      = close.shift(1)                  # previous close as reference price
ceil_price     = ref_price * 1.07                 # HOSE ceiling
floor_price    = ref_price * 0.93                 # HOSE floor

hit_limit_up   = close >= ceil_price * 0.998      # within 0.2% of ceiling
hit_limit_down = close <= floor_price * 1.002     # within 0.2% of floor

# For breakout: reject if hit_limit_up (can't execute next day at reasonable price)
# For pullback: reject if hit_limit_down (panic selling, not constructive pullback)
```

### Half-Day Session Detection

Vietnam has multiple half-day sessions (Tet holidays, etc.). Volume on half-days is 40-60% of normal, making volume-based metrics unreliable.

```python
half_day_suspect = volume < vol_ma20 * 0.50
# If half_day_suspect on signal bar, skip signal — volume metrics unreliable
```

### Gap Risk at Execution

Entry is next-day open. VN stocks can gap 2-5% on overnight news or ATC results.

```python
# Evaluate at execution time, not scan time
gap_pct = (open_next - close_signal) / close_signal
if gap_pct > 0.04:    # >4% gap up — risk/reward destroyed
    skip_entry
if gap_pct < -0.03:   # >3% gap down — setup invalidated
    skip_entry
```

### Sector Concentration Cap

If 3+ signals come from the same sector, it is likely a sector rotation play, not independent setups. Cap output at **2 signals per sector** to maintain portfolio diversification.

---

## 15. Final Acceptance Rules

A stock is a valid long candidate only if **all** of the following are true:

```python
market_ok == True                                    # VNINDEX regime
trend_filter == True                                 # per-stock uptrend
value_avg5 > 1e10                                    # liquidity (rolling 5-day)
is_buildup == True                                   # constructive base
buildup_days >= 3                                    # persistent base
close <= ma20 * 1.05                                 # not extended above MA20
close <= ma50 * 1.15                                 # not extended above MA50
rsi < 72                                             # not overbought
rs_vs_vni > 1.0                                      # outperforming VNINDEX
(setup_breakout == True) or (setup_pullback == True) # valid entry trigger
rr_ratio >= 1.5                                      # acceptable risk/reward
not hit_limit_up (breakout) / not hit_limit_down (pullback)  # tradeable bar
```

If multiple tickers qualify, prioritize by `score` descending. Return **top 10**.

---

## 16. Practical Philosophy

This scanner is intentionally narrow.

It is not designed to find every bullish pattern.
It is designed to find only two continuation structures that are objective enough to scan repeatedly:

- **Breakout after buildup**
- **Pullback continuation to MA20**

That is enough.

A scanner that is too clever often becomes:

- noisy,
- overfit,
- hard to debug,
- and difficult to trust in live conditions.

Simplicity is a feature, but only if the structure is strict.

---

## 17. Vietnam Market Context

This scanner is tuned for the Vietnam stock market (HOSE / HNX). Key structural differences from US/developed markets:

| Factor | VN Reality | How This Spec Handles It |
|--------|-----------|--------------------------|
| **T+2.5 settlement** | Cannot sell for ~2.5 days after buying | SL is mental on Day 1-2; position sizing accounts for 3-day risk |
| **±7% daily limits (HOSE)** | Price can lock at ceiling/floor | Reject signals on limit-up/down bars |
| **~80% retail** | More emotional, momentum-driven | RSI guard, buildup persistence prevents FOMO entries |
| **High correlation** | Most stocks fall when VNINDEX falls | Market regime gate suppresses all longs in corrections |
| **ATC volume spikes** | Closing auction creates artificial volume | Rolling avg value filter, 1.5x volume expansion threshold |
| **Half-day sessions** | Tet, holidays — abnormally low volume | Half-day detection skips signals on suspect bars |
| **Low free float** | Some stocks have concentrated ownership | Liquidity gate (10B VND avg value) filters out illiquid names |
| **Foreign ownership limits** | FOL can restrict institutional buying | Not explicitly handled — monitor for stocks near FOL |

---

## 18. Summary

This rule set encodes six core ideas, with Volman's buildup-and-barrier philosophy at the center:

1. **market must be healthy** — VNINDEX regime gate,
2. **trend must already be up** — per-stock MA20/MA50 alignment,
3. **stock must outperform** — relative strength vs VNINDEX,
4. **setup must show real buildup** — tight, constructive, persistent, with cluster barrier and squeeze detection (Volman),
5. **entry must be either breakout or pullback continuation** — with volume confirmation and barrier quality scoring,
6. **risk/reward must be acceptable** — RR >= 1.5 with defined exits.

If those six are present, the stock is worth attention.
If not, skip it.

---

## 19. Validation Checklist

After implementation, validate against these criteria:

```python
- Minimum 50 signals across 3+ years of VN data
- Win rate >= 40%
- Profit factor >= 1.5
- Max consecutive losses <= 6
- Test in 3 market regimes: uptrend, sideways, correction
- Verify: no signal fires on a limit-up day
- Verify: no signal fires when VNINDEX < MA50
- Verify: out-of-sample performance not worse than 30% vs in-sample
```
