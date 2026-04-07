"""
Bull Triangle Pullback detector — bundled for Streamlit Cloud compatibility.
Original: C:/code_demo/chart_patterns/chart_patterns/pullback_triangle.py
"""

import numpy as np
import pandas as pd

from chart_patterns.pivot_points import find_all_pivot_points
from scipy.stats import linregress

_VALID_TYPES = {"symmetrical", "ascending", "descending"}


def find_pullback_triangle(
    ohlc: pd.DataFrame,
    lookback: int = 25,
    min_points: int = 2,
    prior_lookback: int = 15,
    min_prior_gain: float = 0.02,
    rlimit: float = 0.85,
    slmax_limit: float = 0.00005,
    slmin_limit: float = 0.00005,
    triangle_type: str = "symmetrical",
    progress: bool = False,
) -> pd.DataFrame:
    if triangle_type not in _VALID_TYPES:
        raise ValueError(f"triangle_type must be one of {_VALID_TYPES}, got '{triangle_type}'")
    if lookback < 1 or prior_lookback < 1:
        raise ValueError("lookback and prior_lookback must be >= 1")
    if min_points < 2:
        raise ValueError("min_points must be >= 2")

    ohlc["chart_type"]                     = ""
    ohlc["pullback_triangle_type"]         = ""
    ohlc["pullback_triangle_point"]        = np.nan
    ohlc["pullback_triangle_high_idx"]     = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_triangle_low_idx"]      = [np.array([]) for _ in range(len(ohlc))]
    ohlc["pullback_triangle_slmax"]        = np.nan
    ohlc["pullback_triangle_slmin"]        = np.nan
    ohlc["pullback_triangle_intercmax"]    = np.nan
    ohlc["pullback_triangle_intercmin"]    = np.nan
    ohlc["pullback_triangle_prior_gain"]   = np.nan

    ohlc = find_all_pivot_points(ohlc)

    min_start = lookback + prior_lookback
    for candle_idx in range(min_start, len(ohlc)):
        prior_end   = candle_idx - lookback
        prior_start = prior_end - prior_lookback

        prior_start_price = ohlc.loc[prior_start, "close"]
        prior_end_price   = ohlc.loc[prior_end,   "close"]
        if prior_start_price <= 0:
            continue
        prior_gain = (prior_end_price - prior_start_price) / prior_start_price
        if prior_gain < min_prior_gain:
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

        if abs(rmax) < rlimit or abs(rmin) < rlimit:
            continue

        matched = False
        if triangle_type == "symmetrical":
            matched = slmax <= -slmax_limit and slmin >= slmin_limit
        elif triangle_type == "ascending":
            matched = abs(slmax) <= slmax_limit and slmin >= slmin_limit
        elif triangle_type == "descending":
            matched = slmax <= -slmax_limit and abs(slmin) <= slmin_limit

        if matched:
            ohlc.loc[candle_idx, "chart_type"]                    = "pullback_triangle"
            ohlc.loc[candle_idx, "pullback_triangle_type"]        = triangle_type
            ohlc.loc[candle_idx, "pullback_triangle_point"]       = candle_idx
            ohlc.at[candle_idx,  "pullback_triangle_high_idx"]    = xxmax
            ohlc.at[candle_idx,  "pullback_triangle_low_idx"]     = xxmin
            ohlc.loc[candle_idx, "pullback_triangle_slmax"]       = slmax
            ohlc.loc[candle_idx, "pullback_triangle_slmin"]       = slmin
            ohlc.loc[candle_idx, "pullback_triangle_intercmax"]   = intercmax
            ohlc.loc[candle_idx, "pullback_triangle_intercmin"]   = intercmin
            ohlc.loc[candle_idx, "pullback_triangle_prior_gain"]  = prior_gain

    return ohlc
