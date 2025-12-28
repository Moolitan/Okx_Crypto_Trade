# orderflow/plotting/plotly_footprint.py
"""Plotly footprint chart construction module."""
from typing import TYPE_CHECKING, Optional
import plotly.graph_objects as go

if TYPE_CHECKING:
    from orderflow.models import FootprintProcessed
    from orderflow.config import PlotConfig


def build_figure(
    processed: "FootprintProcessed",
    config: "PlotConfig"
) -> go.Figure:
    """
    Build a Plotly footprint chart.

    Args:
        processed: Processed footprint data (FootprintProcessed dataclass).
        config: Plot configuration.

    Returns:
        go.Figure
    """
    # TODO: Implement logic migrated from FootprintChart.plot
    pass
