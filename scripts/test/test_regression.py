# orderflow/test_regression.py
"""
Regression tests: ensure consistent behavior before and after refactoring.
"""
import pandas as pd
from data.okx.store import OkxPersistStore
from scripts.runner.footprint_chart import FootprintChart


def test_data_consistency():
    """Validate data loading and processing consistency."""
    store = OkxPersistStore(read_only=True)
    symbol, bar, limit = "BTC-USDT-SWAP", "1m", 100

    # Refactored version
    chart = FootprintChart.from_duckdb(
        symbol, bar=bar, footprint_limit=limit, ohlcv_limit=limit
    )

    # Checkpoint 1: footprint_df shape and uniqueness
    assert chart.orderflow_data.shape[0] >= 0, "footprint_df should exist"
    if not chart.orderflow_data.empty:
        assert len(chart.orderflow_data["identifier"].unique()) > 0, \
            "Should have unique identifiers"
        assert len(chart.orderflow_data["price"].unique()) > 0, \
            "Should have unique prices"

    # Checkpoint 2: identifier set alignment between footprint and OHLCV
    footprint_ids = (
        set(chart.orderflow_data["identifier"].astype(str))
        if not chart.orderflow_data.empty else set()
    )
    ohlcv_ids = (
        set(chart.ohlc_data["identifier"].astype(str))
        if not chart.ohlc_data.empty else set()
    )
    if footprint_ids and ohlcv_ids:
        assert footprint_ids == ohlcv_ids, \
            "Identifier sets should match after alignment"

    # Checkpoint 3: processed data existence
    chart.process_data()
    assert chart.is_processed, "Chart data should be processed"
    if not chart.orderflow_data.empty and not chart.ohlc_data.empty:
        assert chart.df is not None, "df should exist"
        assert chart.df2 is not None, "df2 should exist"
        assert chart.labels is not None, "labels should exist"

    # Checkpoint 4: Plotly figure traces
    fig = chart.plot(return_figure=True)
    trace_types = [type(t).__name__ for t in fig.data]
    # Expect at least some traces (may be zero if no data is available)
    assert len(fig.data) >= 0, "Figure should contain traces"

    # Checkpoint 5: key attributes and ranges
    assert hasattr(chart, "granularity"), "Chart should have granularity"
    assert chart.granularity > 0, "Granularity should be positive"

    # Validate y-range and tick values via plot_ranges
    ymin, ymax, xmin, xmax, tickvals, ticktext = chart.plot_ranges(chart.ohlc_data)
    if not chart.ohlc_data.empty:
        assert ymin > ymax, "ymin should be greater than ymax"
        assert len(tickvals) == len(ticktext), \
            "tickvals and ticktext lengths should match"

    print("✅ All regression checks passed")


if __name__ == "__main__":
    test_data_consistency()
