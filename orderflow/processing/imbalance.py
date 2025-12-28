# orderflow/processing/imbalance.py
"""Imbalance computation and annotation module."""
import pandas as pd
from typing import Optional


def calc_imbalance(
    df: pd.DataFrame,
    valid_identifiers: Optional[set] = None
) -> pd.DataFrame:
    """
    Compute orderflow imbalance.

    Args:
        df: Footprint DataFrame with columns:
            ['identifier', 'price', 'bid_size', 'ask_size']
        valid_identifiers: Optional set of valid identifiers used for filtering.

    Returns:
        DataFrame indexed by `identifier` with additional columns:
            - sum: bid_size + ask_size
            - text: formatted bid/ask volume text
            - size: normalized imbalance value
    """
    df = df.copy()

    if valid_identifiers:
        df = df[df["identifier"].isin(valid_identifiers)]

    if df.empty:
        return df

    df["sum"] = df["bid_size"] + df["ask_size"]

    bids, asks = [], []
    for b, a in zip(
        df["bid_size"].astype(int).astype(str),
        df["ask_size"].astype(int).astype(str),
    ):
        dif = 4 - len(a)
        a = a + (" " * dif)
        dif = 4 - len(b)
        b = (" " * dif) + b
        bids.append(b)
        asks.append(a)

    df["text"] = pd.Series(bids, index=df.index) + "  " + pd.Series(asks, index=df.index)
    df.index = df["identifier"]

    df["size"] = (df["bid_size"] - df["ask_size"].shift().bfill()) / (
        df["bid_size"] + df["ask_size"].shift().bfill()
    )
    df["size"] = df["size"].ffill().bfill()

    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate volume distribution text annotations.

    Args:
        df: Output DataFrame from `calc_imbalance`.

    Returns:
        DataFrame with annotated `text` column representing volume bars.
    """
    df2 = df.copy()
    df2 = df2.drop(["size"], axis=1)
    df2["sum"] = df2["sum"] / df2.groupby(df2.index)["sum"].transform("max")
    df2["text"] = ""
    df2["text"] = ["█" * int(sum_ * 10) for sum_ in df2["sum"]]
    df2["text"] = "                    " + df2["text"]
    return df2
