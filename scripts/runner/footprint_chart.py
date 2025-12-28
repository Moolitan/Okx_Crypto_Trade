# -*- coding: utf-8 -*-
"""
Footprint Chart plotting class - DuckDB version

Changes:
- No longer reads orderflow.json / orderflow_1m_candle.json
- Reads directly from DuckDB via data/okx/store.py (OkxPersistStore):
  - Aggregates `market_trades` into footprint data
    (identifier, price, bid_size, ask_size, trade_count)
  - Reads OHLCV candles from the `ohlcv` table
    (identifier, open, high, low, close)

Assumed project layout:
- orderflow/footprint_chart.py
- data/okx/store.py
Both are top-level folders under the project root (orderflow and data are siblings).
"""

import os
import sys
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Add project root to sys.path (orderflow and data are sibling directories)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Access DuckDB via store.py
try:
    from data.okx.store import OkxPersistStore
except Exception as e:
    raise ImportError(
        "Failed to import data.okx.store.OkxPersistStore. "
        "Please confirm the project structure: <project_root>/data/okx/store.py "
        "and that 'orderflow' and 'data' are sibling directories."
    ) from e

# Import refactored modules
try:
    from orderflow.datasource.duckdb_reader import load_footprint, load_ohlcv
    from orderflow.processing.normalize import normalize_footprint, normalize_ohlcv
    from orderflow.processing.align import align_by_identifier
    from orderflow.processing.imbalance import calc_imbalance, annotate
    from orderflow.processing.candles import range_proc, candle_proc
    from orderflow.processing.metrics import calc_params
except ImportError:
    # Fallback for direct script execution without package context
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "duckdb_reader",
        os.path.join(os.path.dirname(__file__), "datasource", "duckdb_reader.py")
    )
    duckdb_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(duckdb_reader)
    load_footprint = duckdb_reader.load_footprint
    load_ohlcv = duckdb_reader.load_ohlcv

    spec = importlib.util.spec_from_file_location(
        "normalize",
        os.path.join(os.path.dirname(__file__), "processing", "normalize.py")
    )
    normalize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalize)
    normalize_footprint = normalize.normalize_footprint
    normalize_ohlcv = normalize.normalize_ohlcv

    spec = importlib.util.spec_from_file_location(
        "align",
        os.path.join(os.path.dirname(__file__), "processing", "align.py")
    )
    align = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align)
    align_by_identifier = align.align_by_identifier

    spec = importlib.util.spec_from_file_location(
        "imbalance",
        os.path.join(os.path.dirname(__file__), "processing", "imbalance.py")
    )
    imbalance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imbalance)
    calc_imbalance = imbalance.calc_imbalance
    annotate = imbalance.annotate

    spec = importlib.util.spec_from_file_location(
        "candles",
        os.path.join(os.path.dirname(__file__), "processing", "candles.py")
    )
    candles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candles)
    range_proc = candles.range_proc
    candle_proc = candles.candle_proc

    spec = importlib.util.spec_from_file_location(
        "metrics",
        os.path.join(os.path.dirname(__file__), "processing", "metrics.py")
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    calc_params = metrics.calc_params


class FootprintChart:
    """
    Footprint chart plotting class.

    Used to visualize orderflow footprints, showing bid/ask volume
    distribution at each price level.
    """

    def __init__(self, orderflow_data: pd.DataFrame, ohlc_data: pd.DataFrame):
        """
        Args:
            orderflow_data: Footprint data with columns
                ['identifier', 'price', 'bid_size', 'ask_size', 'trade_count']
            ohlc_data: OHLC candle data with at least
                ['identifier', 'open', 'high', 'low', 'close']
        """
        self.orderflow_data = orderflow_data
        self.ohlc_data = ohlc_data
        self.is_processed = False

        # Estimate price granularity (used for plotting ranges)
        if len(orderflow_data) > 1 and "price" in orderflow_data.columns:
            prices = sorted(orderflow_data["price"].dropna().unique().tolist())
            if len(prices) > 1:
                self.granularity = abs(prices[1] - prices[0])
            else:
                self.granularity = 1.0
        else:
            self.granularity = 1.0

        # Processed / intermediate data containers
        self.df = None
        self.df2 = None
        self.labels = None
        self.green_hl = None
        self.red_hl = None
        self.green_oc = None
        self.red_oc = None

        # Filter: only render time buckets (identifiers) that exist in footprint data
        self.valid_identifiers = set()
        if not self.orderflow_data.empty and "identifier" in self.orderflow_data.columns:
            self.valid_identifiers = set(self.orderflow_data["identifier"].astype(str).unique())

    # -----------------------------
    # Entry point: load data from DuckDB
    # -----------------------------
    @classmethod
    def from_duckdb(
        cls,
        symbol: str,
        *,
        bar: str = "1m",
        footprint_limit: int = 120,
        ohlcv_limit: int = 120,
        duckdb_path: Optional[str] = None,
        read_only: bool = True,
    ) -> "FootprintChart":
        """
        Build a FootprintChart instance from DuckDB.

        Args:
            symbol: Trading pair symbol, e.g. 'BTC-USDT-SWAP'
            bar: Candle timeframe (OHLCV table is stored per bar)
            footprint_limit: Number of most recent bars used for footprint aggregation
            ohlcv_limit: Number of most recent OHLCV candles to load
            duckdb_path: DuckDB file path; if None, uses default path from store.py
            read_only: Open DuckDB in read-only mode (recommended)

        Returns:
            FootprintChart instance
        """
        with OkxPersistStore(path=duckdb_path, read_only=read_only) as store:
            # Use the datasource layer
            of = load_footprint(store, symbol, bar, footprint_limit)
            ohlcv_all = load_ohlcv(store, symbol, bar, ohlcv_limit)

        # Normalize columns and dtypes
        of = normalize_footprint(of)
        ohlc = normalize_ohlcv(ohlcv_all)

        # Align by identifier (use footprint identifiers as reference)
        of, ohlc = align_by_identifier(of, ohlc, mode="footprint")

        return cls(of, ohlc)

    # -----------------------------
    # Compatibility wrappers (logic moved to processing modules)
    # -----------------------------
    def calc_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute imbalance (delegates to processing.imbalance; kept for compatibility)."""
        return calc_imbalance(df, self.valid_identifiers)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate volume distribution text (delegates to processing.imbalance)."""
        return annotate(df)

    def range_proc(self, ohlc: pd.DataFrame, type_: str = "hl") -> pd.DataFrame:
        """Process high/low or open/close ranges (delegates to processing.candles)."""
        return range_proc(ohlc, type_)

    def candle_proc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process candlestick data for plotting (delegates to processing.candles)."""
        return candle_proc(df)

    def calc_params(self, of: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Compute metrics (delegates to processing.metrics)."""
        return calc_params(of, ohlc)

    def process_data(self):
        """Run all data processing steps prior to plotting."""
        if self.orderflow_data.empty or self.ohlc_data.empty:
            self.is_processed = False
            return

        self.df = calc_imbalance(self.orderflow_data, self.valid_identifiers)
        self.df2 = annotate(self.df.copy())

        self.green_id = self.ohlc_data.loc[self.ohlc_data["close"] >= self.ohlc_data["open"]]["identifier"]
        self.red_id = self.ohlc_data.loc[self.ohlc_data["close"] < self.ohlc_data["open"]]["identifier"]

        self.high_low = range_proc(self.ohlc_data, type_="hl")
        self.green_hl = self.high_low.loc[self.green_id] if len(self.green_id) > 0 else pd.DataFrame()
        if not self.green_hl.empty:
            self.green_hl = candle_proc(self.green_hl)

        self.red_hl = self.high_low.loc[self.red_id] if len(self.red_id) > 0 else pd.DataFrame()
        if not self.red_hl.empty:
            self.red_hl = candle_proc(self.red_hl)

        self.open_close = range_proc(self.ohlc_data, type_="oc")
        self.green_oc = self.open_close.loc[self.green_id] if len(self.green_id) > 0 else pd.DataFrame()
        if not self.green_oc.empty:
            self.green_oc = candle_proc(self.green_oc)

        self.red_oc = self.open_close.loc[self.red_id] if len(self.red_id) > 0 else pd.DataFrame()
        if not self.red_oc.empty:
            self.red_oc = candle_proc(self.red_oc)

        self.labels = calc_params(self.orderflow_data, self.ohlc_data)
        self.is_processed = True

    def plot_ranges(self, ohlc: pd.DataFrame) -> tuple:
        """Compute plot axis ranges and tick labels."""
        if ohlc.empty:
            return 0, 0, 0, 0, [], []

        ymin = ohlc["high"].max() + 1
        ymax = ymin - int(48 * self.granularity)
        xmax = ohlc.shape[0]
        xmin = max(0, xmax - 9)

        tickvals = ohlc["identifier"].tolist()
        ticktext = ohlc["identifier"].tolist()

        return ymin, ymax, xmin, xmax, tickvals, ticktext

    def plot(self, return_figure: bool = False) -> Optional[go.Figure]:
        """Render the footprint chart using Plotly."""
        if not self.is_processed:
            self.process_data()

        if not self.is_processed or self.orderflow_data.empty or self.ohlc_data.empty:
            error_msg = "No data available (footprint or OHLCV not found in DuckDB)."
            fig = go.Figure()
            fig.add_annotation(
                text=error_msg,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#222", plot_bgcolor="#222", height=600)
            if return_figure:
                return fig
            fig.show()
            return None

        # (Plotting logic unchanged; omitted here for brevity)

        if return_figure:
            return fig

        fig.show()
        return None


if __name__ == "__main__":
    # Simple test entry
    chart = FootprintChart.from_duckdb(
        "BTC-USDT-SWAP", bar="1m", footprint_limit=100, ohlcv_limit=100
    )
    chart.plot()
