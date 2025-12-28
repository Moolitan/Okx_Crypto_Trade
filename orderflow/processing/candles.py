# orderflow/processing/candles.py
"""Candlestick processing module."""
import pandas as pd
import numpy as np
from typing import Literal


def range_proc(ohlc: pd.DataFrame, type_: Literal["hl", "oc"] = "hl") -> pd.DataFrame:
    """
    Process high/low or open/close price ranges.

    Args:
        ohlc: OHLC DataFrame
            Expected columns:
                - identifier
                - high, low (for type_="hl")
                - open, close (for type_="oc")
        type_: Range type:
            - "hl": high/low range
            - "oc": open/close range

    Returns:
        DataFrame with columns:
            - price
            - identifier

        The DataFrame is indexed by `identifier` and is intended
        for subsequent candlestick wick/body rendering.
    """
    if type_ == "hl":
        seq = pd.concat([ohlc["low"], ohlc["high"]])
    if type_ == "oc":
        seq = pd.concat([ohlc["open"], ohlc["close"]])

    id_seq = pd.concat([ohlc["identifier"], ohlc["identifier"]])
    seq = pd.DataFrame(seq, columns=["price"])
    seq["identifier"] = id_seq
    seq = seq.sort_index()
    seq = seq.set_index("identifier")
    return seq


def candle_proc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process candlestick range data for plotting (wicks and bodies).

    Args:
        df: Output DataFrame from `range_proc`.

    Returns:
        A processed DataFrame suitable for candlestick rendering.
    """
    if df.empty:
        return df

    df = df.sort_values(by=["price"])
    df = df.reset_index()

    if len(df) < 2:
        return df.set_index("identifier") if "identifier" in df.columns else df

    # Duplicate every second row to construct vertical candle segments
    df_dp = df.iloc[1::2].copy()
    df = pd.concat([df, df_dp])
    df = df.sort_index()

    if "identifier" in df.columns:
        df = df.set_index("identifier")

    df = df.sort_values(by=["price"])

    # Insert NaNs to break lines between candle segments
    if len(df) > 2:
        df.iloc[2::3] = np.nan

    return df
