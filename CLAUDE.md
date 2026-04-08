# CLAUDE.md — VN Stock Screener (Swing D1)

## Project overview
Streamlit app that scans Vietnamese stocks (VN30 / VN100) for Swing Trading signals on the Daily timeframe. Two signal types: **Breakout Momentum** and **Reversal Hunter**, defined in `guide.md`.

## Key files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app (~750 lines) — all logic in one file |
| `guide.md` | Canonical rule definitions for Breakout & Reversal signals |
| `generate_backtest.py` | Generates ~30 PNG backtest charts into `data/backtest/` |
| `requirements.txt` | `streamlit yfinance pandas numpy matplotlib pyarrow plotly vnstock3 openpyxl tqdm` |
| `data/cache/` | Parquet files (one per symbol), incremental price cache |
| `data/backtest/` | Static PNG charts for display in the app |

## Architecture

### Symbol universe
- `VN30_STOCKS` — 30 `.VN` symbols (dict: symbol → sector)
- `VNMID_STOCKS` — 70 `.VN` symbols
- `VN100_STOCKS = {**VN30_STOCKS, **VNMID_STOCKS}` — 100 symbols total

### Price cache (parquet, incremental)
- `load_price_data(symbol)` — reads from `data/cache/<SYMBOL_VN>.parquet`, appends only new bars via yfinance, keeps last 500 bars
- Timezone stripped to UTC-naive before saving

### Indicators (`compute_indicators`)
All shift-based to **exclude the signal candle [-1]** and avoid look-ahead bias:
```python
d["atr10"]        = tr.shift(2).rolling(10).mean()   # bars [-11] to [-2]
d["avg_vol20"]    = d["Volume"].shift(2).rolling(20).mean()
d["avg_vol_pre5"] = d["Volume"].shift(2).rolling(5).mean()
d["high20"]       = d["High"].shift(2).rolling(20).max()
d["ma50"]         = d["Close"].rolling(50).mean()
d["ma200"]        = d["Close"].rolling(200).mean()
d["ma50_prev5"]   = d["ma50"].shift(5)
```

### Scan logic
- `scan_breakout(df)` — returns dict or None; signal candle = `df.iloc[-1]`
- `scan_reversal(df)` — returns dict or None; status always `PENDING`
- `run_scan(symbols)` — ThreadPoolExecutor (8 workers), tries breakout first then reversal per symbol

### UI
- Two scan buttons: **Scan VN30** and **Scan VN100** — no general "Scan Now"
- Results shown in 3 tabs: All / Breakout / Reversal
- Click a row → interactive Plotly chart with SL/TP lines
- Backtest PNGs grid at bottom (up to 30, 3 per row)
- Sidebar: VNINDEX market filter, cache controls, strategy description

## Signal rules (summary — see guide.md for full spec)

### Breakout Momentum
1. `close > MA50` AND `MA50 > MA50_PREV5` (uptrend)
2. `close > open` (bull candle)
3. `close >= high * 0.998` (close near high)
4. `(high - low) > 1.5 × ATR10` (large candle)
5. `high > HIGH20` (20-day breakout)
- Filter: `close <= MA50 * 1.08` (not over-extended)
- SL = low[-1], TP = entry + 2× ATR10

### Reversal Hunter
1. `close < MA50` (downtrend)
2. `abs(close - MA200) <= 1.0 × ATR10` (near MA200 support)
3. `low < prev_low` (rejection — tested lower)
4. `close > open` (bull candle)
5. `close >= high * 0.998` (close near high)
6. `(high - low) > 1.5 × ATR10` (large candle)
- Filter: `close >= MA200 * 0.92` (not in freefall)
- SL = low[-1], TP1 = MA50, TP2 = entry + 2× ATR10
- Status always `PENDING` (confirm when next candle closes > high[-1])

### Volume tags
- `SPIKE+CONTRACT` — vol > 1.5× avg20 AND pre5_avg < 0.8× avg20
- `SPIKE_ONLY` — vol > 1.5× avg20
- `NO_SPIKE` — otherwise

## Index candle convention
- `[-1]` = signal candle (last closed bar = `df.iloc[-1]`)
- `[-2]` = bar before signal = `df.iloc[-2]`
- `[0]` = today's forming candle — **never used in scan logic**
- All indicators use `shift(2).rolling(N)` to cover `[-N-1]` to `[-2]`, excluding `[-1]`

## Backtest charts
Run `python generate_backtest.py` (in Docker or locally) to regenerate.
- Entry = `open[signal+1] * 1.001` (0.1% slippage)
- SL = `low[-1]`, TP = `entry + 2× ATR10`, max hold = 15 bars
- Filename: `{nn}_{SYMBOL}_{BO/RE}_{WI/LO/TI}_{YYYYMMDD}_{pnl}pct.png`
- Output: ~26 charts, mix of WIN/LOSS/TIMEOUT for both signal types

## Docker / local dev

```bash
# Run app locally
streamlit run app.py

# Docker (container: ai_trading_project_v2-app-1)
docker exec ai_trading_project_v2-app-1 python -m py_compile app.py   # syntax check
docker exec ai_trading_project_v2-app-1 python generate_backtest.py   # regenerate charts
docker cp ai_trading_project_v2-app-1:/app/data/backtest/. c:/code_demo/my-screening/data/backtest/  # sync PNGs
```

## What NOT to do
- Do not restore old pattern detection code (VCP, Flat Base, Pullback MA20, chart_patterns module)
- Do not add a general "Scan Now" button — only VN30 and VN100 buttons
- Do not add `BEAR_COUNT` or `BULL_COUNT` to scan rules — removed intentionally
- Do not use `df.iloc[-1]` data when computing rolling indicators (use `shift(2)`)
- Do not use matplotlib emoji characters in chart titles (DejaVu Sans doesn't support them — use `[WIN]` `[LOSS]` `[TIMEOUT]` ASCII instead)

## Streamlit-specific notes
- `st.dataframe(..., on_select="rerun", selection_mode="single-row")` wrapped in try/except for older Streamlit versions
- `@st.cache_data(ttl=3600)` on `get_vnindex_data()`
- Scan results stored in `st.session_state["scan_results"]` and `st.session_state["scan_universe"]`
- Dark theme: `BG_COLOR = "#0a0e17"`, `CARD_COLOR = "#131829"`
