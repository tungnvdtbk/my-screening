# VN Stock Screener — Swing D1

Streamlit app for scanning Vietnamese stocks (`VN30` / `VN100`) for breakout and breakout-pullback setups. The app focuses on trend-continuation workflows only — reversal scanners (mean reversion, climax, pin-bar) were removed in favor of higher-conviction breakout entries.

## Current Scanners

- Daily multi-signal scan: Breakout Momentum, NR7, Gap-Up Breakout, Trend Filter
- Swing Filter
- Price Action — Breakout & Pullback
- Pullback V2 — continuation to MA in uptrend
- BCP — Bull Cluster Pullback (3 bull bars + actionable pullback)
- BPE — Watchlist Breakout Pullback (2-bar big-body or 3-bar above MA20)

## Core Features

- VN30 and VN100 universes with sector labels
- VNINDEX market filter in the sidebar
- Incremental daily price cache in `data/cache/`
- Interactive Plotly chart panels with SL/TP overlays
- Capital and risk-per-trade inputs

## Tech Stack

- Python 3.11
- Streamlit
- pandas / numpy
- plotly / matplotlib
- yfinance
- vnstock3
- pyarrow
- openpyxl
- tqdm

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Local Streamlit uses the default port: `http://localhost:8501`.

## Run with Docker

```bash
docker compose up --build
```

Docker runs the app on port `8000`: `http://localhost:8000`.

Detached mode:

```bash
docker compose up -d
```

## Validate

```bash
python test_app.py
python -m py_compile app.py
```

## Project Structure

```text
app.py                                  # Main Streamlit application
requirements.txt                        # Python dependencies
Dockerfile                              # Container image
docker-compose.yml                      # Local container orchestration
test_app.py                             # Unit tests for scanners/helpers
generate_backtest.py                    # Backtest chart generation
data/cache/                             # Daily price cache
data/backtest/                          # Generated backtest images
README.md                               # Start here
SUMMARY.md                              # High-level system summary
CLAUDE.md                               # Contributor-facing codebase notes
```

## Documentation Map

- `README.md` — onboarding and run instructions
- `SUMMARY.md` — current high-level system summary
- `CLAUDE.md` — contributor-oriented architecture notes
- `gap_scanner.md` — Gap-Up Breakout scanner spec
- `nr7_scanner.md` — NR7 scanner spec
- `trendfilter.md` — Trend Filter spec
- `swing_scanner_rules_pro_v_2.md` — Swing Filter spec
- `price_action_scanner_breakout_pullback_v2.md` — Price Action spec
- `vn_pullback_ma_rule_with_score.md` — Pullback V2 spec
- `watchlist_breakout_pullback_scanner.md` — BPE watchlist spec
- `bull_cluster_pullback_scanner.md` — BCP actionable-pullback spec
- `guide.md` — focused reference for the original D1 breakout framework; not the full system spec
- `instruction.md` — generic prompt/reference material; not synced to runtime behavior

## Source of Truth

Runtime behavior lives in `app.py`. When scanner logic changes, update the matching scanner spec and the top-level docs above.
