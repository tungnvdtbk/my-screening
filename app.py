import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from vnstock3 import Vnstock

# =============================
# CONFIG
# =============================
DATA_PATH = "./data"
engine = create_engine(f"sqlite:///{DATA_PATH}/trades.db")

VN30_STOCKS = {
    # Banking
    "ACB.VN": "Banking", "BID.VN": "Banking", "CTG.VN": "Banking",
    "HDB.VN": "Banking", "LPB.VN": "Banking", "MBB.VN": "Banking",
    "SHB.VN": "Banking", "SSB.VN": "Banking", "STB.VN": "Banking",
    "TCB.VN": "Banking", "TPB.VN": "Banking", "VCB.VN": "Banking",
    "VIB.VN": "Banking", "VPB.VN": "Banking",
    # Real Estate
    "BCM.VN": "Real Estate", "KDH.VN": "Real Estate", "VHM.VN": "Real Estate",
    # Retail
    "MSN.VN": "Retail", "MWG.VN": "Retail", "SAB.VN": "Retail", "VNM.VN": "Retail",
    # Industrial / Production
    "GAS.VN": "Energy", "GVR.VN": "Industrial", "HPG.VN": "Industrial",
    "PLX.VN": "Energy", "POW.VN": "Energy",
    # Technology
    "FPT.VN": "Technology",
    # Insurance
    "BVH.VN": "Insurance",
    # Financial Services
    "SSI.VN": "Financial Services",
    # Aviation
    "VJC.VN": "Aviation",
}

VN30_SYMBOLS = list(VN30_STOCKS.keys())


# =============================
# VNINDEX DATA FETCH (via vnstock)
# =============================
def get_vnindex_data():
    # Try vnstock3 first
    try:
        stock = Vnstock().stock(symbol="VN30", source="VCI")
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df = stock.quote.history(symbol="VNINDEX", start=start, end=end)
        if df is not None and not df.empty:
            df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
            if "time" in df.columns:
                df.index = pd.to_datetime(df["time"])
            elif "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass

    # Fallback to yfinance
    try:
        df = yf.Ticker("^VNINDEX").history(period="6mo")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


# =============================
# SYMBOL MANAGEMENT
# =============================
def load_symbols():
    try:
        with open("symbols.json", "r") as f:
            return json.load(f)
    except:
        return ["HPG.VN", "MBB.VN", "KBC.VN"]


def save_symbols(symbols):
    with open("symbols.json", "w") as f:
        json.dump(symbols, f)


# =============================
# DATA FETCH (SAFE)
# =============================
def get_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period="1y")
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# =============================
# RSI
# =============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# =============================
# VNINDEX FILTER (SAFE)
# =============================
def market_trend():
    df = get_vnindex_data()

    if df is None or df.empty:
        return "UNKNOWN"

    if len(df) < 50:
        return "UNKNOWN"

    df["MA50"] = df["Close"].rolling(50).mean()

    if pd.isna(df["MA50"].iloc[-1]):
        return "UNKNOWN"

    return "UP" if df["Close"].iloc[-1] > df["MA50"].iloc[-1] else "DOWN"


# =============================
# AI CALCULATION
# =============================
def calculate(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()

    price = df["Close"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    ma100 = df["MA100"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    vol = df["Volume"].iloc[-1]
    vol_avg = df["Vol_MA20"].iloc[-1]

    if pd.isna(ma50):
        return None

    has_ma100 = not pd.isna(ma100)

    # Long position: confirm uptrend then define pullback buy zone
    # Uptrend = price above MA100 AND MA50 > MA100 (or price > MA50 if no MA100)
    if has_ma100:
        uptrend = (ma50 > ma100) and (price > ma100)
    else:
        uptrend = price > ma50

    # Pullback buy zone: price dips to MA20-MA50 area while uptrend intact
    zone_low = round(ma50 * 0.98, 2)
    zone_high = round(ma20, 2) if not pd.isna(ma20) else round(ma50 * 1.02, 2)

    # Wider targets for position holding (20-50%)
    target = (round(price * 1.20, 2), round(price * 1.50, 2))
    # Stoploss below MA100 (or MA50) with 7% buffer — position needs room
    stoploss = round((ma100 if has_ma100 else ma50) * 0.93, 2)

    vol_signal = "HIGH" if (not pd.isna(vol_avg) and vol > vol_avg * 1.2) else "NORMAL"

    return price, zone_low, zone_high, target, stoploss, rsi, vol_signal, uptrend


# =============================
# DECISION ENGINE
# =============================
def decision(price, zone_low, zone_high, target, stoploss, rsi, trend, vol_signal, uptrend):
    if trend == "UNKNOWN":
        return "NO DATA"

    if trend == "DOWN":
        return "NO TRADE"

    # Must be in confirmed uptrend for long position
    if not uptrend:
        return "NO TRADE"

    if price <= stoploss:
        return "CUT LOSS"

    # Pullback buy: price dipped into MA20-MA50 support zone
    in_pullback = zone_low <= price <= zone_high

    # Strong buy: pullback + RSI cooling off + volume spike (institutional accumulation)
    if in_pullback and rsi < 45 and vol_signal == "HIGH":
        return "BUY"

    # Watch: pullback happening but no volume confirmation yet
    if in_pullback and rsi < 50:
        return "WATCH"

    # Take partial profit at first target
    if price >= target[0]:
        return "TAKE PROFIT"

    return "HOLD"


# =============================
# SAVE TRADE
# =============================
def save_trade(symbol, price, action):
    try:
        df = pd.DataFrame(
            [[symbol, price, action]],
            columns=["symbol", "price", "action"]
        )
        df.to_sql("trades", engine, if_exists="append", index=False)
    except:
        pass


# =============================
# UI
# =============================
st.set_page_config(layout="wide")
st.title("AI Trading Dashboard")

trend = market_trend()

if trend == "UNKNOWN":
    st.warning("Could not fetch VNINDEX data")
else:
    st.info(f"Market Trend: **{trend}**")

# =============================
# SYMBOL SOURCE SELECTION
# =============================
st.subheader("Watchlist")

source = st.radio(
    "Stock list",
    ["Custom", "VN30"],
    horizontal=True,
)

if source == "VN30":
    active_symbols = VN30_SYMBOLS
else:
    active_symbols = load_symbols()
    col_add, col_rm = st.columns([3, 1])
    with col_add:
        new_symbol = st.text_input("Add symbol (e.g. FPT.VN)")
    with col_rm:
        st.write("")
        st.write("")
        if st.button("Add") and new_symbol and new_symbol not in active_symbols:
            active_symbols.append(new_symbol)
            save_symbols(active_symbols)
            st.rerun()

# =============================
# BUILD SUMMARY DATA
# =============================
rows = []
stock_dfs = {}

with st.spinner("Loading data..."):
    for s in active_symbols:
        df = get_data(s)
        if df is None:
            rows.append({"Symbol": s, "Sector": VN30_STOCKS.get(s, "-"),
                         "Price": None, "MA50": None, "MA100": "-",
                         "RSI": None, "Volume": "-", "Trend": "-",
                         "Pullback Zone": "-", "Target": "-",
                         "Stoploss": None, "Action": "NO DATA"})
            continue

        result = calculate(df)
        if result is None:
            rows.append({"Symbol": s, "Sector": VN30_STOCKS.get(s, "-"),
                         "Price": None, "MA50": None, "MA100": "-",
                         "RSI": None, "Volume": "-", "Trend": "-",
                         "Pullback Zone": "-", "Target": "-",
                         "Stoploss": None, "Action": "INSUFFICIENT DATA"})
            continue

        price, zone_low, zone_high, target, stoploss, rsi, vol_signal, uptrend = result
        action = decision(price, zone_low, zone_high, target, stoploss, rsi, trend, vol_signal, uptrend)

        if action in ["BUY", "TAKE PROFIT", "CUT LOSS"]:
            save_trade(s, price, action)

        stock_dfs[s] = df
        rows.append({
            "Symbol": s,
            "Sector": VN30_STOCKS.get(s, "-"),
            "Price": round(price, 2),
            "MA50": round(df["MA50"].iloc[-1], 2),
            "MA100": round(df["MA100"].iloc[-1], 2) if not pd.isna(df["MA100"].iloc[-1]) else "-",
            "RSI": round(rsi, 2),
            "Volume": vol_signal,
            "Trend": "UP" if uptrend else "DOWN",
            "Pullback Zone": f"{zone_low} - {zone_high}",
            "Target": f"{target[0]} (+20%) - {target[1]} (+50%)",
            "Stoploss": stoploss,
            "Action": action,
        })

# =============================
# FILTERS
# =============================
if rows:
    summary_df = pd.DataFrame(rows)

    col_sector, col_action = st.columns(2)

    sectors = sorted(summary_df["Sector"].dropna().unique())
    with col_sector:
        selected_sectors = st.multiselect("Filter by sector", sectors)

    actions = sorted(summary_df["Action"].dropna().unique())
    with col_action:
        selected_actions = st.multiselect("Filter by action", actions)

    filtered = summary_df
    if selected_sectors:
        filtered = filtered[filtered["Sector"].isin(selected_sectors)]
    if selected_actions:
        filtered = filtered[filtered["Action"].isin(selected_actions)]

    # Sort by action priority: BUY/WATCH first, then HOLD, then rest
    ACTION_ORDER = {"BUY": 0, "WATCH": 1, "CUT LOSS": 2, "TAKE PROFIT": 3,
                    "HOLD": 4, "NO TRADE": 5, "NO DATA": 6, "INSUFFICIENT DATA": 7}
    filtered = filtered.copy()
    filtered["_sort"] = filtered["Action"].map(ACTION_ORDER).fillna(9)
    filtered = filtered.sort_values("_sort").drop(columns=["_sort"])

    # =============================
    # SUMMARY TABLE (grouped by sector)
    # =============================
    if filtered.empty:
        st.info("No stocks match the selected filters.")
    else:
        for sector, group in filtered.groupby("Sector", sort=True):
            st.caption(sector)
            st.dataframe(
                group.drop(columns=["Sector"]),
                use_container_width=True,
                hide_index=True,
            )

# =============================
# CHART VIEWER
# =============================
available = [s for s in filtered["Symbol"].tolist() if s in stock_dfs] if rows and not filtered.empty else []

if available:
    st.subheader("Chart")
    cols = st.columns([3, 1])
    with cols[0]:
        selected = st.selectbox("Select symbol", available, label_visibility="collapsed")
    with cols[1]:
        show_chart = st.button("View Chart")

    if show_chart and selected:
        df = stock_dfs[selected]
        chart_cols = ["Close", "MA50"]
        if "MA100" in df.columns and not df["MA100"].isna().all():
            chart_cols.append("MA100")
        st.line_chart(df[chart_cols])
