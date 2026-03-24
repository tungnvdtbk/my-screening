# AI Trading Project v2

A Streamlit-based AI trading dashboard for analyzing Vietnamese stock market data with technical analysis and automated trading signals.

## Tech Stack

- **Python 3.11**
- **Streamlit** — interactive web dashboard
- **yfinance** — stock market data
- **vnstock3** — Vietnamese market data (VNINDEX)
- **SQLAlchemy + SQLite** — trade logging
- **pandas / numpy** — data analysis

## Prerequisites

- Python 3.11+
- pip
- (Optional) Docker & Docker Compose

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

### Local Development

```bash
streamlit run app.py
```

The dashboard will be available at **http://localhost:8501**.

### Docker

```bash
docker-compose up
```

This will:
- Build the image from the Dockerfile
- Map port **8501** to your host
- Auto-restart unless stopped

To run in detached mode:

```bash
docker-compose up -d
```

### Deploy on Koyeb

1. Push this repo to GitHub.

2. Create a new **Web Service** on [Koyeb](https://app.koyeb.com/) and connect your GitHub repo.

3. Select **Dockerfile** as the build method (auto-detected).

4. Koyeb automatically sets the `PORT` environment variable — the Dockerfile reads it.

5. Set the health check path to `/_stcore/health` (HTTP, port 8000).

6. Deploy. Koyeb will build the image and expose your app on a public URL.

## Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker orchestration
├── symbols.json         # Custom stock watchlist
├── .dockerignore        # Docker build exclusions
└── data/                # SQLite database (auto-created)
```

## Features

- **Market Trend Filter** — uses VNINDEX MA50 to determine overall market direction
- **VN30 Stock Analysis** — pre-loaded with 30+ Vietnamese stocks by sector
- **Custom Watchlist** — add/remove symbols via `symbols.json` or the UI
- **Technical Indicators** — MA20, MA50, MA100, RSI(14), volume analysis
- **Trading Signals** — BUY, WATCH, HOLD, TAKE PROFIT, CUT LOSS, NO TRADE
- **Trade Logging** — all signals saved to SQLite (`data/trades.db`)

## Trading Logic

| Signal | Condition |
|---|---|
| **BUY** | Price pulls back into MA20–MA50 zone, RSI < 45, high volume, market UP |
| **WATCH** | Pullback zone, RSI < 50 |
| **TAKE PROFIT** | Price ≥ 20% above entry |
| **CUT LOSS** | Price falls below MA100 (with 7% buffer) |
| **HOLD** | Currently holding position |
| **NO TRADE** | Not in uptrend or market is DOWN |
