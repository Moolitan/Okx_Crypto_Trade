# -*- coding: utf-8 -*-
"""
Footprint Chart plotting class - DuckDB version

Changes:
- No longer reads orderflow.json / orderflow_1m_candle.json
- Reads directly from DuckDB via data/okx/store.py (OkxPersistStore):
  - Aggregate `market_trades` into footprint data
    (identifier, price, bid_size, ask_size, trade_count)
  - Read OHLCV candles from `ohlcv` table
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

# Add project root to sys.path (orderflow and data are siblings)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Access DuckDB through store.py
try:
    from data.okx.store import OkxPersistStore
except Exception as e:
    raise ImportError(
        "Failed to import data.okx.store.OkxPersistStore. "
        "Please confirm the project structure: <project_root>/data/okx/store.py "
        "and that 'orderflow' and 'data' are sibling directories."
    ) from e


class FootprintChart:
    """
    Footprint Chart plotting class
    Used to visualize orderflow footprint: bid/ask volume distribution at each price level.
    """

    def __init__(self, orderflow_data: pd.DataFrame, ohlc_data: pd.DataFrame):
        """
        Args:
            orderflow_data: Footprint data with columns
                ['identifier', 'price', 'bid_size', 'ask_size', 'trade_count']
            ohlc_data: OHLC candles with at least
                ['identifier', 'open', 'high', 'low', 'close']
        """
        self.orderflow_data = orderflow_data
        self.ohlc_data = ohlc_data
        self.is_processed = False

        # Estimate price granularity (used for chart range)
        if len(orderflow_data) > 1 and "price" in orderflow_data.columns:
            prices = sorted(orderflow_data["price"].dropna().unique().tolist())
            if len(prices) > 1:
                self.granularity = abs(prices[1] - prices[0])
            else:
                self.granularity = 1.0
        else:
            self.granularity = 1.0

        # Processed/intermediate data
        self.df = None
        self.df2 = None
        self.labels = None
        self.green_hl = None
        self.red_hl = None
        self.green_oc = None
        self.red_oc = None

        # Filter: only render time buckets (identifier) that exist in footprint data
        self.valid_identifiers = set()
        if not self.orderflow_data.empty and "identifier" in self.orderflow_data.columns:
            self.valid_identifiers = set(self.orderflow_data["identifier"].astype(str).unique())

    # -----------------------------
    # New entry: load from DuckDB
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
            symbol: e.g. 'BTC-USDT-SWAP'
            bar: candle timeframe (ohlcv table is stored by bar)
            footprint_limit: number of latest bars to aggregate footprint for
                             (implemented by store.load_footprint_data)
            ohlcv_limit: number of latest candles to load from ohlcv
            duckdb_path: DuckDB file path; if None, uses store.py default OKX_DUCKDB_PATH
            read_only: open DB as read-only (recommended True)
        """
        with OkxPersistStore(path=duckdb_path, read_only=read_only) as store:
            # 1) Footprint (aggregated from market_trades inside DuckDB)
            of = store.load_footprint_data(symbol=symbol, bar=bar, limit=int(footprint_limit))

            # 2) OHLCV (latest ohlcv_limit bars, time-ascending as returned by store)
            ohlcv_all = store.load_ohlcv(symbol=symbol, bar=bar, limit=int(ohlcv_limit))

        # Normalize columns & dtypes
        if of is None or of.empty:
            of = pd.DataFrame(columns=["identifier", "price", "bid_size", "ask_size", "trade_count"])
        else:
            for c in ["identifier"]:
                if c in of.columns:
                    of[c] = of[c].astype(str)
            for c in ["price", "bid_size", "ask_size"]:
                if c in of.columns:
                    of[c] = pd.to_numeric(of[c], errors="coerce").fillna(0.0)
            if "trade_count" in of.columns:
                of["trade_count"] = pd.to_numeric(of["trade_count"], errors="coerce").fillna(0).astype(int)

        if ohlcv_all is None or ohlcv_all.empty:
            ohlc = pd.DataFrame(columns=["identifier", "open", "high", "low", "close"])
        else:
            # store.load_ohlcv typically returns: timestamp, identifier, open/high/low/close/volume...
            # Keep only what the plot needs.
            needed = ["identifier", "open", "high", "low", "close"]
            for c in needed:
                if c not in ohlcv_all.columns:
                    ohlcv_all[c] = np.nan
            ohlc = ohlcv_all[needed].copy()

            ohlc["identifier"] = ohlc["identifier"].astype(str)
            for c in ["open", "high", "low", "close"]:
                ohlc[c] = pd.to_numeric(ohlc[c], errors="coerce")

        # Force alignment: only plot the identifiers that exist in footprint data.
        # (Otherwise OHLCV may contain more bars than footprint, causing empty/misaligned columns.)
        if not of.empty and not ohlc.empty:
            valid = set(of["identifier"].astype(str).unique())
            ohlc = ohlc[ohlc["identifier"].astype(str).isin(valid)].copy()

        return cls(of, ohlc)

    # -----------------------------
    # Plot logic (kept the same)
    # -----------------------------
    def calc_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.valid_identifiers:
            df = df[df["identifier"].isin(self.valid_identifiers)]

        if df.empty:
            return df

        df["sum"] = df["bid_size"] + df["ask_size"]

        bids, asks = [], []
        for b, a in zip(df["bid_size"].astype(int).astype(str), df["ask_size"].astype(int).astype(str)):
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

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2 = df2.drop(["size"], axis=1)
        df2["sum"] = df2["sum"] / df2.groupby(df2.index)["sum"].transform("max")
        df2["text"] = ""
        df2["text"] = ["█" * int(sum_ * 10) for sum_ in df2["sum"]]
        df2["text"] = "                    " + df2["text"]
        return df2

    def range_proc(self, ohlc: pd.DataFrame, type_: str = "hl") -> pd.DataFrame:
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

    def candle_proc(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.sort_values(by=["price"])
        df = df.reset_index()

        if len(df) < 2:
            return df.set_index("identifier") if "identifier" in df.columns else df

        df_dp = df.iloc[1::2].copy()
        df = pd.concat([df, df_dp])
        df = df.sort_index()

        if "identifier" in df.columns:
            df = df.set_index("identifier")

        df = df.sort_values(by=["price"])

        if len(df) > 2:
            df.iloc[2::3] = np.nan

        return df

    def calc_params(self, of: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
        if of.empty or ohlc.empty:
            return pd.DataFrame(columns=["value", "type", "text"])

        grouped = of.groupby(of["identifier"]).sum(numeric_only=True)
        delta = grouped["ask_size"] - grouped["bid_size"]
        delta = delta.reindex(ohlc["identifier"], fill_value=0)

        cum_delta = delta.rolling(window=min(10, len(delta)), min_periods=1).sum()

        roc = cum_delta.diff() / cum_delta.shift(1).replace(0, np.nan) * 100
        roc = roc.fillna(0).round(2)

        volume = grouped["ask_size"] + grouped["bid_size"]
        volume = volume.reindex(ohlc["identifier"], fill_value=0)

        delta_df = pd.DataFrame({"value": delta.values, "type": "delta"}, index=delta.index)
        cum_delta_df = pd.DataFrame({"value": cum_delta.values, "type": "cum_delta"}, index=cum_delta.index)
        roc_df = pd.DataFrame({"value": roc.values, "type": "roc"}, index=roc.index)
        volume_df = pd.DataFrame({"value": volume.values, "type": "volume"}, index=volume.index)

        labels = pd.concat([delta_df, cum_delta_df, roc_df, volume_df])
        labels = labels.sort_index()
        labels["text"] = labels["value"].astype(str)

        labels["value"] = np.tanh(labels["value"].replace([np.inf, -np.inf], 0))
        return labels

    def process_data(self):
        if self.orderflow_data.empty or self.ohlc_data.empty:
            self.is_processed = False
            return

        self.df = self.calc_imbalance(self.orderflow_data)
        self.df2 = self.annotate(self.df.copy())

        self.green_id = self.ohlc_data.loc[self.ohlc_data["close"] >= self.ohlc_data["open"]]["identifier"]
        self.red_id = self.ohlc_data.loc[self.ohlc_data["close"] < self.ohlc_data["open"]]["identifier"]

        self.high_low = self.range_proc(self.ohlc_data, type_="hl")
        self.green_hl = self.high_low.loc[self.green_id] if len(self.green_id) > 0 else pd.DataFrame()
        if not self.green_hl.empty:
            self.green_hl = self.candle_proc(self.green_hl)

        self.red_hl = self.high_low.loc[self.red_id] if len(self.red_id) > 0 else pd.DataFrame()
        if not self.red_hl.empty:
            self.red_hl = self.candle_proc(self.red_hl)

        self.open_close = self.range_proc(self.ohlc_data, type_="oc")
        self.green_oc = self.open_close.loc[self.green_id] if len(self.green_id) > 0 else pd.DataFrame()
        if not self.green_oc.empty:
            self.green_oc = self.candle_proc(self.green_oc)

        self.red_oc = self.open_close.loc[self.red_id] if len(self.red_id) > 0 else pd.DataFrame()
        if not self.red_oc.empty:
            self.red_oc = self.candle_proc(self.red_oc)

        self.labels = self.calc_params(self.orderflow_data, self.ohlc_data)
        self.is_processed = True

    def plot_ranges(self, ohlc: pd.DataFrame) -> tuple:
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

        ymin, ymax, xmin, xmax, tickvals, ticktext = self.plot_ranges(self.ohlc_data)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0, row_heights=[9, 1])

        # 1) Volume profile text overlay
        if self.df2 is not None and not self.df2.empty:
            filtered_df2 = self.df2[self.df2.index.isin(self.valid_identifiers)] if self.valid_identifiers else self.df2
            if not filtered_df2.empty:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df2.index,
                        y=filtered_df2["price"],
                        text=filtered_df2["text"],
                        name="VolumeProfile",
                        textposition="middle right",
                        textfont=dict(size=8, color="rgb(0, 0, 255, 0.0)"),
                        hoverinfo="none",
                        mode="text",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        # 2) Orderflow heatmap
        if self.df is not None and not self.df.empty:
            filtered_df = self.df[self.df.index.isin(self.valid_identifiers)] if self.valid_identifiers else self.df
            if not filtered_df.empty:
                fig.add_trace(
                    go.Heatmap(
                        x=filtered_df.index,
                        y=filtered_df["price"],
                        z=filtered_df["size"],
                        text=filtered_df["text"],
                        colorscale="icefire_r",
                        showscale=True,
                        showlegend=True,
                        name="BidAsk",
                        texttemplate="%{text}",
                        textfont={"size": 11, "family": "Courier New"},
                        hovertemplate="Price: %{y}<br>Size: %{text}<br>Imbalance: %{z}<extra></extra>",
                        xgap=240,
                        colorbar=dict(
                            title=dict(text="Imbalance", font=dict(size=12)),
                            len=0.6,
                            thickness=15,
                            x=1.02,
                            y=0.5,
                            lenmode="fraction",
                            outlinewidth=1,
                            outlinecolor="white",
                            tickfont=dict(size=10),
                            tickformat=".2f",
                        ),
                    ),
                    row=1,
                    col=1,
                )

        # 3) Candlestick wicks (high/low)
        if self.green_hl is not None and not self.green_hl.empty:
            fig.add_trace(
                go.Scatter(
                    x=self.green_hl.index,
                    y=self.green_hl["price"],
                    name="Candle",
                    legendgroup="group",
                    showlegend=True,
                    line=dict(color="green", width=1.5),
                ),
                row=1,
                col=1,
            )

        if self.red_hl is not None and not self.red_hl.empty:
            fig.add_trace(
                go.Scatter(
                    x=self.red_hl.index,
                    y=self.red_hl["price"],
                    name="Candle",
                    legendgroup="group",
                    showlegend=False,
                    line=dict(color="red", width=1.5),
                ),
                row=1,
                col=1,
            )

        # 4) Candle bodies (as bars)
        if self.green_oc is not None and not self.green_oc.empty and len(self.green_id) > 0:
            green_ohlc = self.ohlc_data[self.ohlc_data["identifier"].isin(self.green_id)]
            if not green_ohlc.empty:
                fig.add_trace(
                    go.Bar(
                        x=green_ohlc["identifier"].tolist(),
                        y=(green_ohlc["close"] - green_ohlc["open"]).tolist(),
                        base=green_ohlc["open"].tolist(),
                        name="Candle",
                        legendgroup="group",
                        showlegend=False,
                        marker=dict(color="green", line=dict(color="green", width=1)),
                        width=0.6,
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

        if self.red_oc is not None and not self.red_oc.empty and len(self.red_id) > 0:
            red_ohlc = self.ohlc_data[self.ohlc_data["identifier"].isin(self.red_id)]
            if not red_ohlc.empty:
                fig.add_trace(
                    go.Bar(
                        x=red_ohlc["identifier"].tolist(),
                        y=(red_ohlc["open"] - red_ohlc["close"]).tolist(),
                        base=red_ohlc["close"].tolist(),
                        name="Candle",
                        legendgroup="group",
                        showlegend=False,
                        marker=dict(color="red", line=dict(color="red", width=1)),
                        width=0.6,
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

        # 5) Metrics panel (delta / cum_delta / roc / volume)
        if self.labels is not None and not self.labels.empty:
            fig.add_trace(
                go.Heatmap(
                    x=self.labels.index,
                    y=self.labels["type"],
                    z=self.labels["value"],
                    colorscale="rdylgn",
                    showscale=False,
                    showlegend=True,
                    name="Parameters",
                    text=self.labels["text"],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                    xgap=4,
                    ygap=4,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title="Order Flow Footprint Chart (DuckDB)",
            yaxis=dict(title="Price", showgrid=False, range=[ymax, ymin], tickformat=".2f"),
            yaxis2=dict(fixedrange=True, showgrid=False),
            xaxis2=dict(title="Time", showgrid=False),
            xaxis=dict(showgrid=False, range=[xmin, xmax]),
            height=780,
            template="plotly_dark",
            paper_bgcolor="#222",
            plot_bgcolor="#222",
            dragmode="pan",
            margin=dict(l=10, r=0, t=40, b=20),
        )

        fig.update_xaxes(
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=0.25,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        )
        fig.update_yaxes(
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=0.25,
        )
        fig.update_layout(spikedistance=1000, hoverdistance=100)

        config = {
            "modeBarButtonsToRemove": ["zoomIn", "zoomOut", "zoom", "autoScale"],
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
        }

        if return_figure:
            return fig

        fig.show(config=config)
        return None

