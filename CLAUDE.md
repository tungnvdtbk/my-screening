# CLAUDE.md — Contributor Notes for VN Stock Screener

This file is contributor context for the current repo. Keep it aligned with `app.py`, not with older summaries or one-off prompt files.

## Project Overview

Single-file Streamlit app for scanning Vietnamese stocks. **Breakout / breakout-pullback only** — reversal scanners (Mean Reversion, Climax, Pin Bar D1, Pin Bar 4H, Pin Bar v2) were removed.

- Combined daily scan: Breakout Momentum, Gap-Up, NR7, Pullback V2, Trend Filter
- Separate sections: Swing Filter, Price Action, Pullback V2, BCP (Bull Cluster Pullback), BPE (Watchlist Breakout Pullback)

The app currently lives mostly in `app.py`.

## Source of Truth

- Runtime behavior and UI: `app.py`
- Strategy rules: scanner-specific markdown files such as `gap_scanner.md`, `nr7_scanner.md`, `trendfilter.md`, `swing_scanner_rules_pro_v_2.md`, `price_action_scanner_breakout_pullback_v2.md`, `vn_pullback_ma_rule_with_score.md`, `watchlist_breakout_pullback_scanner.md`, and `bull_cluster_pullback_scanner.md`
- Reference-only material: `guide.md` and `instruction.md`

Do not treat `guide.md` as the canonical spec for the whole application. It is only a focused strategy reference.

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app, scanner logic, UI, data loading, charts |
| `test_app.py` | Unit tests for scanners, cache helpers, scoring, and gating logic |
| `generate_backtest.py` | Backtest image generation |
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

### Indicator Convention

For the daily combined scan, many shared indicators intentionally exclude the signal candle using `shift(2)` to avoid look-ahead bias:

```python
d["atr10"]        = tr.shift(2).rolling(10).mean()
d["avg_vol20"]    = d["Volume"].shift(2).rolling(20).mean()
d["avg_vol_pre5"] = d["Volume"].shift(2).rolling(5).mean()
d["high10"]       = d["High"].shift(2).rolling(10).max()
d["high20"]       = d["High"].shift(2).rolling(20).max()
```

## Scan Runners

- `run_scan()` — combined daily scan, one prioritized signal per symbol
- `run_swing_scan()` — Swing Filter top candidates sorted by cross-sectional score
- `run_pa_scan()` — Price Action top candidates with sector cap
- `run_pullback_v2_scan()` — Pullback-to-MA continuation candidates with score
- `run_bcp_scan()` — Bull Cluster Pullback (top 15, gap_t DESC)
- `run_bpe_scan()` — Watchlist Breakout Pullback (top 20, Tier A→B→C)

## Current UI Layout

### Sidebar

- Market status
- Cache controls
- Capital / risk inputs
- Scan options
- Strategy cheat-sheet

### Main page sections

1. Daily multi-signal scan
2. Swing Filter
3. Price Action — Breakout & Pullback
4. Pullback V2
5. BCP — Bull Cluster Pullback (ranked above BPE)
6. BPE — Watchlist Breakout Pullback Test

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
