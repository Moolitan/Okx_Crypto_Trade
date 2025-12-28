# orderflow/datasource/duckdb_reader.py
"""DuckDB data loading module."""
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from data.okx.store import OkxPersistStore


def load_footprint(
    store: "OkxPersistStore",
    symbol: str,
    bar: str,
    limit: int
) -> pd.DataFrame:
    """
    Load footprint data from DuckDB.

    Args:
        store: OkxPersistStore instance.
        symbol: Trading pair symbol, e.g. 'BTC-USDT-SWAP'.
        bar: Timeframe, e.g. '1m'.
        limit: Number of bars to load.

    Returns:
        DataFrame with columns:
            ['identifier', 'price', 'bid_size', 'ask_size', 'trade_count']

        Column definitions:
            - identifier: str (ISO timestamp)
            - price: float64
            - bid_size: float64
            - ask_size: float64
            - trade_count: int64
    """
    # Footprint data (aggregated from market_trades inside DuckDB)
    of = store.load_footprint_data(symbol=symbol, bar=bar, limit=int(limit))
    return of


def load_ohlcv(
    store: "OkxPersistStore",
    symbol: str,
    bar: str,
    limit: int
) -> pd.DataFrame:
    """
    Load OHLCV data from DuckDB.

    Args:
        store: OkxPersistStore instance.
        symbol: Trading pair symbol.
        bar: Timeframe.
        limit: Number of bars to load.

    Returns:
        DataFrame with columns:
            ['identifier', 'open', 'high', 'low', 'close', 'volume']

        Column definitions:
            - identifier: str (ISO timestamp)
            - open/high/low/close/volume: float64
    """
    # OHLCV data (latest bars, time-ascending as returned by the store)
    ohlcv_all = store.load_ohlcv(symbol=symbol, bar=bar, limit=int(limit))
    return ohlcv_all
