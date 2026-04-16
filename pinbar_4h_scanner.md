# Pin Bar 4H Scanner — Intraday Pin Bar at Context

> Scans **4-hour candles** for bullish pin bars at meaningful support levels. Reuses the same `scan_pinbar()` detection logic as the D1 scanner but adds **multi-timeframe (MTF) trend alignment** from D1 data and a **lookback window** over recent 4H bars.

---

## 1. Concept

Same pin bar at context logic as the D1 scanner (`pinbar_scanner.md`), applied to 4H candles. The key difference:

- **D1 scanner**: checks only the last daily candle
- **4H scanner**: checks the last N 4H candles (default: 4 bars = ~2 trading days) and uses D1 trend as an MTF filter

A pin bar on 4H gives **earlier entries** than waiting for D1 — you can spot intraday rejection at support before the daily candle closes.

---

## 2. Data Pipeline

### 2a. Fetch 1H data

```python
# Primary: yfinance (no strict rate limit, good for parallel scan)
raw = yf.Ticker(symbol).history(period="60d", interval="1h")

# Fallback: vnstock3 (VCI or TCBS source)
stock = Vnstock().stock(symbol=sym_clean, source="VCI")  # or "TCBS"
raw = stock.quote.history(symbol=sym_clean, start=start, end=end, interval="1H")
```

### 2b. Resample 1H to 4H

```python
df_4h = raw.resample("4h").agg({
    "Open": "first", "High": "max", "Low": "min",
    "Close": "last", "Volume": "sum",
}).dropna(subset=["Open", "Close"])
df_4h = df_4h[df_4h["Volume"] > 0]   # drop non-trading windows
# Minimum: 60 bars of 4H data required
```

### 2c. Timezone handling

```python
# yfinance: convert to Asia/Ho_Chi_Minh then strip to naive
if raw.index.tz is not None:
    raw.index = raw.index.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
```

**Note**: 4H data is NOT cached to parquet (unlike D1). It is fetched fresh each scan because intraday data changes frequently.

---

## 3. D1 Trend Check (MTF Alignment)

Before scanning 4H candles, load the D1 data for each symbol to determine trend direction:

```python
def _check_d1_trend(sym) -> bool | None:
    df = load_price_data(sym, use_cache=True)    # D1 parquet cache
    ma50      = df["Close"].rolling(50).mean()
    ma50_prev5 = ma50.shift(5)
    c = df["Close"].iloc[-1]
    m50 = ma50.iloc[-1]
    m50p = ma50_prev5.iloc[-1]
    return bool(c > m50 and m50 > m50p)
    # True  = D1 uptrend (close > MA50, MA50 rising)
    # False = D1 downtrend
    # None  = insufficient data
```

This `d1_trend_up` value is passed to `scan_pinbar()` as the `d1_trend_up` parameter, providing the **+3 MTF trend score** when aligned.

---

## 4. Scan Logic

### 4a. Lookback window

```python
lookback_bars = 4   # default: scan last 4 candles of 4H data (~2 trading days)

for offset in range(lookback_bars):
    sub = df_4h.iloc[:len(df_4h) - offset] if offset > 0 else df_4h
    sig = scan_pinbar(sub, vnindex_df, d1_trend_up=d1_trend)
    if sig:
        sig["timeframe"] = "4H"
        hits.append(sig)
```

Each offset slides the "signal candle" backwards by one 4H bar, so signals from the last ~2 days are detected. A single symbol can produce **multiple hits** (one per 4H bar that qualifies).

### 4b. Pin bar detection

Reuses `scan_pinbar()` from the D1 scanner — same shape rules, context conditions, volume tiers, SL/TP/R:R, and quality scoring. See `pinbar_scanner.md` for full spec.

Summary of detection rules:

**Shape (all required):**
```
[1] lower_wick >= 0.60 * candle_range    # long rejection wick
[2] body       <= 0.33 * candle_range    # small body
[3] upper_wick <= 0.25 * candle_range    # short upper wick
[4] candle_range >= 0.5 * ATR10          # minimum size
```

**Context (at least one required):**
```
[C1] AT_MA20:  abs(low - ma20)  <= 0.3*ATR  AND close > ma20  AND MA50 rising AND close <= MA50*1.10
[C2] AT_MA50:  abs(low - ma50)  <= 0.3*ATR  AND close > ma50*0.98
[C3] AT_MA200: abs(low - ma200) <= 0.3*ATR  AND close >= ma200*0.98
[C4] AT_SWING: abs(low - swing_low20) <= 0.3*ATR  AND close > swing_low20
```

**Safety filters:**
```
[X1] close >= ma200 * 0.88         # not in freefall
[X2] atr10 >= 0.3 * atr30          # not dead market
[X3] candle_range > 0              # avoid division by zero
```

---

## 5. Quality Scoring (0-13 points)

Same scoring as D1, but the **MTF trend** score uses D1 data instead of self-referencing:

| Points | Condition | 4H-specific note |
|--------|-----------|-------------------|
| +3 | MTF trend aligned | D1 `close > MA50` AND `MA50 rising` (passed via `d1_trend_up`) |
| +2 | Pullback structure | >= 2 bearish 4H candles before pin bar |
| +2 | Context confluence | >= 2 support levels matched |
| +2 | Volume spike | TIER1 or TIER2 (vol > 1.5x avg20) |
| +1 | Volume dry-up | 3 quiet 4H bars before pin bar |
| +1 | Bull close | close > open |
| +1 | Body position | Body in upper 25% of candle range |
| +1 | RSI oversold | RSI(14) < 40 on 4H |

**Tier gate:**
```
Tier A: score >= 7    (high quality)
Tier B: score >= 4    (tradeable)
Below 4: rejected
```

---

## 6. Entry / SL / TP

```python
Entry   = close * 1.001              # 0.1% slippage
SL      = low of pin bar candle
TP      = max(MA50, swing_high20, entry + 2 * ATR10)
R:R     = (TP - entry) / (entry - SL)    # must be >= 2.0
risk_pct = (entry - SL) / entry * 100    # must be <= 7%

Status  = PENDING (always — require next candle close > pin bar high)
```

---

## 7. Signal Types & Priority

Same signal naming as D1, priority order: `MA200 > MA50 > MA20 > SWING`

```python
PINBAR_MA200    # pin bar at MA200 — rarest, strongest
PINBAR_MA50     # pin bar at MA50 — major trend support
PINBAR_MA20     # pin bar at MA20 in uptrend
PINBAR_SWING    # pin bar at 20-bar swing low
```

---

## 8. Parallelization

```python
# 8 threads, one per symbol
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(_one, sym, sec): sym for sym, sec in symbols.items()}
```

Each thread:
1. Loads D1 data → `_check_d1_trend()` → `d1_trend_up`
2. Fetches 1H data → resamples to 4H → `load_price_data_4h()`
3. Computes indicators on 4H data → `compute_indicators()`
4. Loops through lookback window → `scan_pinbar()` per offset

---

## 9. Result Sorting

```python
candidates.sort(key=lambda r: (
    0 if r.get("pin_tier") == "A" else 1,   # Tier A first
    -r.get("pin_score", 0),                   # higher score first
    -r.get("rr", 0),                           # higher R:R first
))
```

---

## 10. UI Section

- Located below the D1 scan section, separated by a divider
- Two buttons: **Scan PB4H VN30** (30 stocks) and **Scan PB4H VN100** (100 stocks)
- Results table columns: Ma, Tier, Score, Signal, Date, Close, Context, Wick, RSI, SL, TP, R:R, Volume, Quality, Nganh
- Quality tags in table: `MTF` (D1 trend aligned), `PB{n}` (pullback count), `DryUp` (volume dry-up), `TopBody` (body in upper range)
- Click row → chart panel with metrics: Close, MA20, MA50, R:R, SL, TP, Wick%, RSI, Vol tier, Score detail
- Session state keys: `pb4h_results`, `pb4h_universe`, `pb4h_sel`

---

## 11. Differences from D1 Pin Bar Scanner

| Aspect | D1 Scanner | 4H Scanner |
|--------|-----------|------------|
| Timeframe | Daily candles | 4-hour candles (resampled from 1H) |
| Data source | Parquet cache (incremental) | Fresh fetch each scan (no cache) |
| Lookback | Last 1 candle only | Last 4 candles (~2 trading days) |
| MTF trend | Self-referencing (MA50 rising) | D1 close > MA50 AND MA50 rising |
| Multiple hits | 1 per symbol max | Up to `lookback_bars` per symbol |
| Data period | 500 bars D1 history | 60 days of 1H → ~60 bars of 4H |
| Scan buttons | Part of main Scan VN30/VN100 | Separate Scan PB4H VN30/VN100 |
| Min bars required | 60 D1 bars | 60 4H bars |

---

## 12. Implementation Reference

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `_resample_1h_to_4h()` | `app.py:3183` | Resample 1H OHLCV to 4H bars |
| `load_price_data_4h()` | `app.py:3193` | Fetch 1H data (yfinance/vnstock3), resample to 4H |
| `_check_d1_trend()` | `app.py:3232` | D1 trend check for MTF alignment |
| `run_pinbar_4h_scan()` | `app.py:3250` | Main scan orchestrator with lookback loop |
| `_render_pinbar4h_results()` | `app.py:3303` | Render results table in Streamlit |
| `scan_pinbar()` | `app.py:979` | Shared pin bar detection logic (used by both D1 and 4H) |
| `compute_indicators()` | `app.py:243` | Shared indicator computation |
| `_pinbar_vol_tier()` | `app.py:321` | Volume tier classification (1.5x threshold) |
