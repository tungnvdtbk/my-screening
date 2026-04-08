"""
download_data.py — Bulk download full history for all VN30 symbols.

Run once before backtesting:
    python download_data.py

Data saved to ./data/cache/<SYMBOL>.parquet (git-ignored).
Backtests then read from cache with no API calls.
"""

from __future__ import annotations

import os
import time
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = "./data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

VN30_STOCKS = [
    "ACB.VN", "BID.VN", "CTG.VN", "HDB.VN", "LPB.VN",
    "MBB.VN", "SHB.VN", "SSB.VN", "STB.VN", "TCB.VN",
    "TPB.VN", "VCB.VN", "VIB.VN", "VPB.VN",
    "BCM.VN", "KDH.VN", "VHM.VN",
    "MSN.VN", "MWG.VN", "SAB.VN", "VNM.VN",
    "GAS.VN", "GVR.VN", "HPG.VN",
    "PLX.VN", "POW.VN",
    "FPT.VN",
    "BVH.VN",
    "SSI.VN",
    "VJC.VN",
]

PERIOD = "max"   # pull full available history


def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, symbol.replace(".", "_") + ".parquet")


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def download_symbol(symbol: str) -> tuple[str, bool, int]:
    path = _cache_path(symbol)
    try:
        df = yf.Ticker(symbol).history(period=PERIOD)
        df = _strip_tz(df)
        if df is None or df.empty:
            return symbol, False, 0
        df.to_parquet(path)
        return symbol, True, len(df)
    except Exception as e:
        return symbol, False, 0


def main():
    print(f"Downloading {len(VN30_STOCKS)} VN30 symbols (period={PERIOD}) ...")
    print(f"Cache: {os.path.abspath(CACHE_DIR)}\n")

    ok, fail = [], []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(download_symbol, s): s for s in VN30_STOCKS}
        for fut in as_completed(futures):
            symbol, success, rows = fut.result()
            if success:
                print(f"  OK   {symbol:15s}  {rows} bars")
                ok.append(symbol)
            else:
                print(f"  FAIL {symbol}")
                fail.append(symbol)

    print(f"\nDone: {len(ok)} OK, {len(fail)} failed")
    if fail:
        print(f"Failed: {fail}")


if __name__ == "__main__":
    main()
