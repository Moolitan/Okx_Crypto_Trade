# orderflow/processing/normalize.py
"""Data normalization module."""
import pandas as pd
import numpy as np


def normalize_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a footprint DataFrame.

    Args:
        df: Raw footprint data (may have missing columns or incorrect dtypes).

    Returns:
        A standardized footprint DataFrame with columns:
            - identifier: str
            - price: float64
            - bid_size: float64
            - ask_size: float64
            - trade_count: int64
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["identifier", "price", "bid_size", "ask_size", "trade_count"])

    of = df.copy()
    for c in ["identifier"]:
        if c in of.columns:
            of[c] = of[c].astype(str)
    for c in ["price", "bid_size", "ask_size"]:
        if c in of.columns:
            of[c] = pd.to_numeric(of[c], errors="coerce").fillna(0.0)
    if "trade_count" in of.columns:
        of["trade_count"] = pd.to_numeric(of["trade_count"], errors="coerce").fillna(0).astype(int)

    return of


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an OHLCV DataFrame.

    Args:
        df: Raw OHLCV data.

    Returns:
        A standardized OHLCV DataFrame with columns:
            - identifier: str
            - open, high, low, close: float64
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["identifier", "open", "high", "low", "close"])

    ohlcv_all = df.copy()
    # store.load_ohlcv typically returns:
    # timestamp, identifier, open, high, low, close, volume, ...
    # Keep only the columns required for plotting.
    needed = ["identifier", "open", "high", "low", "close"]
    for c in needed:
        if c not in ohlcv_all.columns:
            ohlcv_all[c] = np.nan
    ohlc = ohlcv_all[needed].copy()

    ohlc["identifier"] = ohlc["identifier"].astype(str)
    for c in ["open", "high", "low", "close"]:
        ohlc[c] = pd.to_numeric(ohlc[c], errors="coerce")

    return ohlc
