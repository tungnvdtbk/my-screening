# VN Stock Screener — Current System Summary

This summary reflects the current app structure in `app.py`. It is a high-level overview, not the per-scanner source of truth.

The system is now breakout/breakout-pullback only. Reversal-style scanners (Mean Reversion, Climax, Pin Bar D1, Pin Bar 4H, Pin Bar v2) were removed to focus exclusively on trend-continuation setups.

## Main Sections

### 1. Daily multi-signal scan

The top section scans `VN30` or `VN100` and returns one prioritized daily signal per symbol from this chain:

```text
Breakout -> Gap -> NR7 -> Pullback V2 -> Trend Filter
```

Key behavior:

- Uses the shared daily cache from `data/cache/`
- Sorts results by signal priority and quality fields
- Shows results in tabs: All / Breakout / NR7 / Gap / Trend Filter
- If `VNINDEX < MA50`, breakout-style signals are suppressed and only Trend Filter signals remain

### 2. Swing Filter

Cross-sectional scanner based on `swing_scanner_rules_pro_v_2.md`:

- Requires constructive buildup plus breakout confirmation
- Uses a VNINDEX market-regime gate by default
- Returns top candidates sorted by score

### 3. Price Action — Breakout & Pullback

Volman-style continuation scanner based on `price_action_scanner_breakout_pullback_v2.md`:

- Detects breakout-after-buildup and pullback-to-MA20 setups
- Uses barrier clustering, squeeze detection, RS vs VNINDEX, and sector-cap logic
- Returns top candidates sorted by cross-sectional score

### 4. Pullback V2

Continuation-to-MA scanner based on `vn_pullback_ma_rule_with_score.md`:

- Requires uptrend (close>MA20>MA50, MA20 rising) plus positive RS20/RS55 vs VNINDEX
- Looks for a 5-bar coil at MA10 with a confirmation trigger candle
- Alerts when the composite score ≥ 70

### 5. BCP — Bull Cluster Pullback

Actionable-pullback scanner ranked above BPE in the UI:

- Trio gate inherited from BPE Filter C: 3 consecutive bull bars in last 25 bars
  with the last bar closing above its MA20 (uptrend gate Close>MA200, MA200 rising)
- Pullback gate: current close has dropped under cluster_high but is still above
  cluster_low — the support cluster is being tested, not broken
- Top 15 sorted by `gap_t` DESC (longer pullback wins), tie-break depth_in_zone DESC
- Spec: `bull_cluster_pullback_scanner.md`

### 6. BPE — Watchlist Breakout Pullback Test

Watchlist scanner producing top 20 candidates from two filters:

- Filter A/B: 2 consecutive bull bars in last 25 bars, ≥1 big body with matching volume
- Filter C: 3 consecutive bull bars in last 25 bars, last close above MA20
- Tier A (both big), B (one big), C (3-bull soft); sort A → B → C, then gap_t ASC
- Spec: `watchlist_breakout_pullback_scanner.md`

## Sidebar Controls

Current sidebar groups:

- Market status (`VNINDEX`, MA20, MA50)
- Cache management
- Capital and risk-per-trade inputs
- Scan options, including optional VNINDEX bypass for Swing / Price Action
- Strategy cheat-sheet text

## Data and Caching

- Daily data is cached incrementally to `data/cache/<SYMBOL>.parquet` with CSV fallback
- `get_vnindex_data()` is cached in Streamlit for 1 hour
- Chart panels load the selected symbol again and render Plotly candles plus signal overlays

## Universes

- `VN30_STOCKS` — 30 symbols
- `VNMID_STOCKS` — mid-cap basket
- `VN100_STOCKS` — merged universe used by all scan sections

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
