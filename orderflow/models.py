# orderflow/models.py
"""Data structure definitions (dataclasses)."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FootprintRaw:
    """Raw data loaded from DuckDB (before normalization)."""
    footprint_df: pd.DataFrame
    ohlcv_df: pd.DataFrame


@dataclass
class FootprintProcessed:
    """Processed data ready for plotting."""
    df: pd.DataFrame          # Output of calc_imbalance
    df2: pd.DataFrame         # Output of annotate
    labels: pd.DataFrame      # Output of calc_params
    green_hl: pd.DataFrame
    red_hl: pd.DataFrame
    green_oc: pd.DataFrame
    red_oc: pd.DataFrame
    granularity: float
    tickvals: list
    ticktext: list
    ymin: float
    ymax: float
    xmin: int
    xmax: int
    valid_identifiers: set
    green_id: Optional[pd.Series] = None
    red_id: Optional[pd.Series] = None
