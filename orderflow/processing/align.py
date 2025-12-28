# orderflow/processing/align.py
"""Data alignment module."""
import pandas as pd
from typing import Literal


def align_by_identifier(
    footprint_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    mode: Literal["footprint", "ohlcv"] = "footprint"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by the `identifier` column.

    Args:
        footprint_df: Footprint data.
        ohlcv_df: OHLCV data.
        mode:
            - "footprint": align to footprint identifiers (crop OHLCV to footprint)
            - "ohlcv": align to OHLCV identifiers (crop footprint to OHLCV)

    Returns:
        A tuple of (aligned_footprint_df, aligned_ohlcv_df).
    """
    if footprint_df.empty or ohlcv_df.empty:
        return footprint_df, ohlcv_df

    if mode == "footprint":
        # Force alignment: only keep identifiers that exist in footprint data.
        # Otherwise, OHLCV may contain more bars than footprint,
        # leading to empty or misaligned columns in the plot.
        valid = set(footprint_df["identifier"].astype(str).unique())
        ohlcv_df = ohlcv_df[ohlcv_df["identifier"].astype(str).isin(valid)].copy()
    elif mode == "ohlcv":
        # Align footprint to OHLCV identifiers.
        valid = set(ohlcv_df["identifier"].astype(str).unique())
        footprint_df = footprint_df[footprint_df["identifier"].astype(str).isin(valid)].copy()

    return footprint_df, ohlcv_df
