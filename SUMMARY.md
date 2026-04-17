# VN Stock Screener — Current System Summary

This summary reflects the current app structure in `app.py`. It is a high-level overview, not the per-scanner source of truth.

## Main Sections

### 1. Daily multi-signal scan

The top section scans `VN30` or `VN100` and returns one prioritized daily signal per symbol from this chain:

```text
Breakout -> Gap -> NR7 -> Pin Bar -> Trend Filter
```

Key behavior:

- Uses the shared daily cache from `data/cache/`
- Sorts results by signal priority and quality fields
- Shows results in tabs: All / Breakout / NR7 / Gap / Pin Bar / Trend Filter
- If `VNINDEX < MA50`, breakout-style signals are suppressed and only non-breakout signals remain

### 2. Mean Reversion Range

Separate long-only range scanner for sideways markets:

- Buys near support inside a validated range
- Sidebar exposes MR-specific parameters such as range size, support tolerance, bottom-zone threshold, and target R:R
- Results are sorted by `final_score`

### 3. Swing Filter

Cross-sectional scanner based on `swing_scanner_rules_pro_v_2.md`:

- Requires constructive buildup plus breakout confirmation
- Uses a VNINDEX market-regime gate by default
- Returns top candidates sorted by score

### 4. Price Action — Breakout & Pullback

Volman-style continuation scanner based on `price_action_scanner_breakout_pullback_v2.md`:

- Detects breakout-after-buildup and pullback-to-MA20 setups
- Uses barrier clustering, squeeze detection, RS vs VNINDEX, and sector-cap logic
- Returns top candidates sorted by cross-sectional score

### 5. Climax Reversal

Reversal scanner for sell-climax / false-break setups:

- Looks for sharp prior decline, support violation, and reversal candle
- Tracks `PENDING` / `CONFIRMED` style state fields and reversal type

### 6. Pin Bar 4H

Intraday pin-bar scanner over recent 4H candles:

- Fetches 1H data, resamples to 4H
- Uses D1 trend alignment as a multi-timeframe filter
- Scans a short recent lookback window instead of only the latest bar

## Sidebar Controls

Current sidebar groups:

- Market status (`VNINDEX`, MA20, MA50)
- Cache management
- Capital and risk-per-trade inputs
- Scan options, including optional VNINDEX bypass for Swing / Price Action
- Mean Reversion configuration
- Strategy cheat-sheet text

## Data and Caching

- Daily data is cached incrementally to `data/cache/<SYMBOL>.parquet` with CSV fallback
- `get_vnindex_data()` is cached in Streamlit for 1 hour
- 4H price data is fetched fresh and resampled from 1H data
- Chart panels load the selected symbol again and render Plotly candles plus signal overlays

## Universes

- `VN30_STOCKS` — 30 symbols
- `VNMID_STOCKS` — mid-cap basket
- `VN100_STOCKS` — merged universe used by most scan sections

## Run and Validate

```bash
# local
pip install -r requirements.txt
streamlit run app.py

# docker
docker compose up --build

# tests / validation
python test_app.py
python -m py_compile app.py
```

## Documentation Roles

- Use the scanner-specific markdown files for rule details
- Use `README.md` for onboarding
- Use `CLAUDE.md` for contributor context
- Treat `guide.md` and `instruction.md` as reference material, not the full current system spec
