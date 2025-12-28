# orderflow/processing/metrics.py
"""Metrics computation module."""
import pandas as pd
import numpy as np


def calc_params(
    footprint_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute footprint-related metrics (delta, cumulative delta, ROC, volume).

    Args:
        footprint_df: Footprint data.
        ohlcv_df: OHLCV data.

    Returns:
        DataFrame indexed by `identifier` with columns:
            - value: metric value (normalized)
            - type: metric type ('delta', 'cum_delta', 'roc', 'volume')
            - text: string representation of the raw metric value
    """
    if footprint_df.empty or ohlcv_df.empty:
        return pd.DataFrame(columns=["value", "type", "text"])

    grouped = footprint_df.groupby(footprint_df["identifier"]).sum(numeric_only=True)

    # Delta: ask volume minus bid volume
    delta = grouped["ask_size"] - grouped["bid_size"]
    delta = delta.reindex(ohlcv_df["identifier"], fill_value=0)

    # Cumulative delta (rolling window)
    cum_delta = delta.rolling(window=min(10, len(delta)), min_periods=1).sum()

    # Rate of change of cumulative delta
    roc = cum_delta.diff() / cum_delta.shift(1).replace(0, np.nan) * 100
    roc = roc.fillna(0).round(2)

    # Total traded volume
    volume = grouped["ask_size"] + grouped["bid_size"]
    volume = volume.reindex(ohlcv_df["identifier"], fill_value=0)

    delta_df = pd.DataFrame({"value": delta.values, "type": "delta"}, index=delta.index)
    cum_delta_df = pd.DataFrame({"value": cum_delta.values, "type": "cum_delta"}, index=cum_delta.index)
    roc_df = pd.DataFrame({"value": roc.values, "type": "roc"}, index=roc.index)
    volume_df = pd.DataFrame({"value": volume.values, "type": "volume"}, index=volume.index)

    labels = pd.concat([delta_df, cum_delta_df, roc_df, volume_df])
    labels = labels.sort_index()
    labels["text"] = labels["value"].astype(str)

    # Normalize values for heatmap visualization
    labels["value"] = np.tanh(labels["value"].replace([np.inf, -np.inf], 0))
    return labels
