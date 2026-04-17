# CLAUDE.md — Contributor Notes for VN Stock Screener

This file is contributor context for the current repo. Keep it aligned with `app.py`, not with older summaries or one-off prompt files.

## Project Overview

Single-file Streamlit app for scanning Vietnamese stocks on multiple workflows:

- Combined daily scan: Breakout Momentum, Gap-Up, NR7, Pin Bar at Context, Trend Filter
- Separate sections: Mean Reversion, Swing Filter, Price Action, Climax Reversal, Pin Bar 4H

The app currently lives mostly in `app.py` and is roughly 3.4k lines long.

## Source of Truth

- Runtime behavior and UI: `app.py`
- Strategy rules: scanner-specific markdown files such as `gap_scanner.md`, `nr7_scanner.md`, `pinbar_scanner.md`, `trendfilter.md`, `mean_reversion_scanner.md`, `climax_scanner.md`, `swing_scanner_rules_pro_v_2.md`, and `price_action_scanner_breakout_pullback_v2.md`
- Reference-only material: `guide.md` and `instruction.md`

Do not treat `guide.md` as the canonical spec for the whole application. It is only a focused strategy reference.

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app, scanner logic, UI, data loading, charts |
| `test_app.py` | Unit tests for scanners, cache helpers, scoring, and gating logic |
| `generate_backtest.py` | Backtest image generation |
| `gen_pb4h_charts.py` | Pin Bar 4H chart utilities |
| `data/cache/` | Incremental daily price cache |
| `data/backtest/` | Generated backtest images |
| `README.md` | User-facing entry doc |
| `SUMMARY.md` | High-level system summary |

## Architecture

### Universes

- `VN30_STOCKS` — 30 `.VN` symbols
- `VNMID_STOCKS` — additional mid-cap symbols
- `VN100_STOCKS = {**VN30_STOCKS, **VNMID_STOCKS}`

### Daily Data

- `load_price_data(symbol, use_cache=True)` reads/writes incremental cache files under `data/cache/`
- Daily charts and daily scanners work on cached D1 data
- `get_vnindex_data()` is cached with `@st.cache_data(ttl=3600)`

### Intraday 4H Data

- `load_price_data_4h()` fetches 1H data and resamples to 4H
- 4H data is fetched fresh; it is not persisted like the D1 cache

### Indicator Convention

For the daily combined scan, many shared indicators intentionally exclude the signal candle using `shift(2)` to avoid look-ahead bias:

```python
d["atr10"]        = tr.shift(2).rolling(10).mean()
d["avg_vol20"]    = d["Volume"].shift(2).rolling(20).mean()
d["avg_vol_pre5"] = d["Volume"].shift(2).rolling(5).mean()
d["high10"]       = d["High"].shift(2).rolling(10).max()
d["high20"]       = d["High"].shift(2).rolling(20).max()
```

Pin Bar 4H has its own fetch/resample pipeline before reusing the shared pin-bar logic.

## Scan Runners

- `run_scan()` — combined daily scan, one prioritized signal per symbol
- `run_mr_scan()` — Mean Reversion results sorted by `final_score`
- `run_swing_scan()` — Swing Filter top candidates sorted by cross-sectional score
- `run_pa_scan()` — Price Action top candidates with sector cap
- `run_climax_scan()` — Climax reversal candidates
- `run_pinbar_4h_scan()` — recent 4H pin-bar hits

## Current UI Layout

### Sidebar

- Market status
- Cache controls
- Capital / risk inputs
- Scan options
- Mean Reversion config
- Strategy cheat-sheet

### Main page sections

1. Daily multi-signal scan
2. Mean Reversion Range
3. Swing Filter
4. Price Action — Breakout & Pullback
5. Climax Reversal
6. Pin Bar 4H

## Commands

```bash
# local
streamlit run app.py

# docker
docker compose up --build

# validation
python test_app.py
python -m py_compile app.py
```

Local Streamlit defaults to `http://localhost:8501`. Docker is configured for port `8000`.

## Keep Docs in Sync

- If scanner logic changes, update the matching scanner spec
- If the visible app sections change, update `README.md`, `SUMMARY.md`, and this file
- Avoid hardcoding fragile counts such as total tests, exact chart totals, or file lengths unless generated automatically
- Do not describe the app as a 5-scanner system; that is stale
- Do not resurrect the old BUY/WATCH/HOLD summary in top-level docs; it does not describe the current app
