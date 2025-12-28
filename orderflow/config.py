# orderflow/config.py
"""Configuration class definitions."""
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Plot configuration."""
    theme: str = "plotly_dark"
    height: int = 780
    paper_bgcolor: str = "#222"
    plot_bgcolor: str = "#222"
    title: str = "Order Flow Footprint Chart (DuckDB)"
    dragmode: str = "pan"
    margin: dict = None

    def __post_init__(self):
        if self.margin is None:
            self.margin = dict(l=10, r=0, t=40, b=20)
