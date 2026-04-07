"""
Bull Flag Pullback detector — bundled for Streamlit Cloud compatibility.
Original: C:/code_demo/chart_patterns/chart_patterns/pullback_flag.py
"""

import numpy as np
import pandas as pd

from chart_patterns.pivot_points import find_all_pivot_points
from scipy.stats import linregress


def find_pullback_flag(
    ohlc: pd.DataFrame,
    lookback: int = 20,
    min_points: int = 2,
    pole_lookback: int = 15,
    min_pole_gain: float = 0.02,
    r_max: float = 0.85,
    r_min: float = 0.85,
    lower_ratio_slope: float = 0.5,
    upper_ratio_slope: float = 2.0,
    progress: bool = False,
) -> pd.DataFrame:
    if lookback < 1 or pole_lookback < 1:
        raise ValueError("lookback and pole_lookback must be >= 1")
    if min_points < 2:
        raise ValueError("min_points must be >= 2")

    ohlc["chart_type"]                = ""
    ohlc["pullback_flag_point"]       = np.nan
    ohlc["pullback_flag_highs_idx"]   = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_flag_lows_idx"]    = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_flag_highs"]       = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_flag_lows"]        = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_flag_slmax"]       = np.nan
    ohlc["pullback_flag_slmin"]       = np.nan
    ohlc["pullback_flag_intercmax"]   = np.nan
    ohlc["pullback_flag_intercmin"]   = np.nan
    ohlc["pullback_flag_pole_gain"]   = np.nan

    ohlc = find_all_pivot_points(ohlc)

    min_start = lookback + pole_lookback
    for candle_idx in range(min_start, len(ohlc)):
        pole_end   = candle_idx - lookback
        pole_start = pole_end - pole_lookback

        pole_start_price = ohlc.loc[pole_start, "close"]
        pole_end_price   = ohlc.loc[pole_end,   "close"]
        if pole_start_price <= 0:
            continue
        pole_gain = (pole_end_price - pole_start_price) / pole_start_price
        if pole_gain < min_pole_gain:
            continue

        maxim = minim = xxmax = xxmin = np.array([])
        for i in range(candle_idx - lookback, candle_idx + 1):
            if ohlc.loc[i, "pivot"] == 1:
                minim = np.append(minim, ohlc.loc[i, "low"])
                xxmin = np.append(xxmin, i)
            if ohlc.loc[i, "pivot"] == 2:
                maxim = np.append(maxim, ohlc.loc[i, "high"])
                xxmax = np.append(xxmax, i)

        if xxmax.size < min_points or xxmin.size < min_points:
            continue

        slmin, intercmin, rmin, _, _ = linregress(xxmin, minim)
        slmax, intercmax, rmax, _, _ = linregress(xxmax, maxim)

        if slmin == 0:
            continue
        if (abs(rmax) >= r_max and abs(rmin) >= r_min
                and slmax < 0 and slmin < 0
                and lower_ratio_slope < abs(slmax / slmin) < upper_ratio_slope):
            ohlc.loc[candle_idx, "chart_type"]              = "pullback_flag"
            ohlc.loc[candle_idx, "pullback_flag_point"]     = candle_idx
            ohlc.at[candle_idx,  "pullback_flag_highs"]     = maxim
            ohlc.at[candle_idx,  "pullback_flag_lows"]      = minim
            ohlc.at[candle_idx,  "pullback_flag_highs_idx"] = xxmax
            ohlc.at[candle_idx,  "pullback_flag_lows_idx"]  = xxmin
            ohlc.loc[candle_idx, "pullback_flag_slmax"]     = slmax
            ohlc.loc[candle_idx, "pullback_flag_slmin"]     = slmin
            ohlc.loc[candle_idx, "pullback_flag_intercmax"] = intercmax
            ohlc.loc[candle_idx, "pullback_flag_intercmin"] = intercmin
            ohlc.loc[candle_idx, "pullback_flag_pole_gain"] = pole_gain

    return ohlc
