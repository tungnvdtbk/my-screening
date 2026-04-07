import numpy as np
import pandas as pd

from chart_patterns.utils import check_ohlc_names
from typing import Union


def find_pivot_point(ohlc: pd.DataFrame, current_row: int,
                     left_count: int = 3, right_count: int = 3) -> int:
    check_ohlc_names(ohlc)
    if current_row - left_count < 0 or current_row + right_count >= len(ohlc):
        return 0

    pivot_low = pivot_high = 1
    for idx in range(current_row - left_count, current_row + right_count + 1):
        if ohlc.loc[current_row, "low"]  > ohlc.loc[idx, "low"]:
            pivot_low  = 0
        if ohlc.loc[current_row, "high"] < ohlc.loc[idx, "high"]:
            pivot_high = 0

    if pivot_low and pivot_high:
        return 3
    if pivot_low:
        return 1
    if pivot_high:
        return 2
    return 0


def find_pivot_point_position(row: pd.Series) -> float:
    try:
        if row["pivot"] == 1:
            return row["low"]  - 1e-3
        if row["pivot"] == 2:
            return row["high"] + 1e-3
        return np.nan
    except Exception:
        return np.nan


def find_all_pivot_points(ohlc: pd.DataFrame, left_count: int = 3,
                          right_count: int = 3,
                          name_pivot: Union[None, str] = None,
                          progress: bool = False) -> pd.DataFrame:
    if name_pivot is not None:
        ohlc.loc[:, name_pivot]               = ohlc.apply(lambda row: find_pivot_point(ohlc, row.name, left_count, right_count), axis=1)
        ohlc.loc[:, f"{name_pivot}_pos"]      = ohlc.apply(lambda row: find_pivot_point_position(row), axis=1)
    else:
        ohlc.loc[:, "pivot"]     = ohlc.apply(lambda row: find_pivot_point(ohlc, row.name, left_count, right_count), axis=1)
        ohlc.loc[:, "pivot_pos"] = ohlc.apply(lambda row: find_pivot_point_position(row), axis=1)
    return ohlc
