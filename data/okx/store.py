# data/okx/store.py
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd
import numpy as np

from .path import OKX_DUCKDB_PATH

try:
    import duckdb
except ImportError as e:
    duckdb = None
    _duckdb_import_error = e


def _ensure_duckdb():
    if duckdb is None:
        raise RuntimeError(
            f"duckdb is not available: {_duckdb_import_error}. "
            f"Install with: pip install duckdb"
        )


def _to_ts_ms(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to int64 milliseconds (UTC).
    Robust handling for ISO strings, datetime objects, and numeric timestamps.
    """
    if series.empty:
        return series.astype("int64")

    # 1) datetime64
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
        return (dt.astype("int64") // 1_000_000).astype("int64")

    # 2) numeric heuristic: seconds vs ms
    if pd.api.types.is_numeric_dtype(series):
        s = series.fillna(0).astype("int64")
        sample = int(s.iloc[0]) if len(s) > 0 else 0
        if 0 < sample < 100_000_000_000:  # < 1e11 => seconds
            return (s * 1000).astype("int64")
        return s

    # 3) strings/objects
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return (dt.astype("int64") // 1_000_000).fillna(0).astype("int64")


def _ts_ms_to_iso(ts_ms: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts_ms.astype("int64"), unit="ms", utc=True, errors="coerce")
    return dt.astype(str)


def _bar_to_seconds(bar: str) -> int:
    """
    Convert a bar string like '1m', '5m', '1h' to seconds.
    """
    bar = (bar or "1m").strip().lower()
    mapping = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
    return mapping.get(bar, 60)


@dataclass
class OkxPersistStore:
    """
    Persistence/query layer (DuckDB as primary store, optional Parquet export).

    This module should:
    - initialize schema (when not read_only)
    - provide append/upsert and query methods
    """

    path: Optional[str] = None
    parquet_dir: Optional[str] = None
    read_only: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        _ensure_duckdb()
        self.path = str(self.path or OKX_DUCKDB_PATH)

        if not self.read_only:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            if self.parquet_dir:
                os.makedirs(self.parquet_dir, exist_ok=True)

        config = {"allow_unsigned_extensions": "true"}
        self.con = duckdb.connect(self.path, config=config, read_only=self.read_only)

        if not self.read_only:
            self._init_schema()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -----------------------
    # Schema
    # -----------------------

    def _init_schema(self):
        # OHLCV: Primary Key (symbol, bar, ts_ms)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                bar VARCHAR,
                ts_ms BIGINT,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, bar, ts_ms)
            );
            """
        )

        # Funding
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS funding (
                symbol VARCHAR,
                ts_ms BIGINT,
                funding_rate DOUBLE,
                realized_rate DOUBLE,
                PRIMARY KEY (symbol, ts_ms)
            );
            """
        )

        # Fills (append-only; indexed)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                strategy VARCHAR,
                symbol VARCHAR,
                ts_ms BIGINT,
                order_id VARCHAR,
                side VARCHAR,
                price DOUBLE,
                qty DOUBLE,
                fee DOUBLE,
                fee_ccy VARCHAR,
                extra_json VARCHAR
            );
            """
        )
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_fills ON fills(strategy, symbol, ts_ms);")

        # Signals (append-only; indexed)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                strategy VARCHAR,
                symbol VARCHAR,
                ts_ms BIGINT,
                signal_name VARCHAR,
                value DOUBLE,
                decision VARCHAR,
                reason VARCHAR,
                extra_json VARCHAR
            );
            """
        )
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_signals ON signals(strategy, symbol, ts_ms);")

        # Market trades (tick-level; for footprint)
        self._init_market_trades_schema()

    def _init_market_trades_schema(self) -> None:
        """
        Create market_trades table + indexes, enforcing uniqueness on (symbol, trade_id).

        Strategy:
        1) Prefer UNIQUE constraint in CREATE TABLE.
        2) Fallback to UNIQUE INDEX if constraint syntax/enforcement is unsupported by the DuckDB version.
        """
        try:
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS market_trades (
                    symbol VARCHAR,
                    ts_ms BIGINT,
                    trade_id VARCHAR,
                    price DOUBLE,
                    sz DOUBLE,
                    side VARCHAR,
                    UNIQUE(symbol, trade_id)
                );
                """
            )
        except Exception:
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS market_trades (
                    symbol VARCHAR,
                    ts_ms BIGINT,
                    trade_id VARCHAR,
                    price DOUBLE,
                    sz DOUBLE,
                    side VARCHAR
                );
                """
            )

        self.con.execute("CREATE INDEX IF NOT EXISTS idx_mkt_trades_ts ON market_trades(symbol, ts_ms);")
        try:
            self.con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_mkt_trades_uniq ON market_trades(symbol, trade_id);")
        except Exception:
            pass

    # -----------------------
    # OHLCV
    # -----------------------

    def upsert_ohlcv(self, symbol: str, bar: str, df: pd.DataFrame) -> None:
        """
        Upsert OHLCV bars using DuckDB ON CONFLICT.
        Expected columns: timestamp, open, high, low, close, volume
        """
        if df is None or df.empty:
            return

        req = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in req):
            missing = [c for c in req if c not in df.columns]
            raise ValueError(f"OHLCV df missing columns: {missing}")

        tmp = df.copy()
        tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        tmp["symbol"] = symbol
        tmp["bar"] = bar

        insert_df = tmp[["symbol", "bar", "ts_ms", "open", "high", "low", "close", "volume"]]

        with self._lock:
            self.con.register("ohlcv_in", insert_df)
            self.con.execute(
                """
                INSERT INTO ohlcv
                SELECT symbol, bar, ts_ms, open, high, low, close, volume FROM ohlcv_in
                ON CONFLICT (symbol, bar, ts_ms) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume;
                """
            )
            self.con.unregister("ohlcv_in")

    def load_ohlcv(
        self,
        symbol: str,
        bar: str,
        start_ts_ms: Optional[int] = None,
        end_ts_ms: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        q = "SELECT ts_ms, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND bar=?"
        params = [symbol, bar]

        if start_ts_ms is not None:
            q += " AND ts_ms >= ?"
            params.append(int(start_ts_ms))
        if end_ts_ms is not None:
            q += " AND ts_ms <= ?"
            params.append(int(end_ts_ms))

        q += " ORDER BY ts_ms ASC"
        if limit:
            q += f" LIMIT {int(limit)}"

        with self._lock:
            out = self.con.execute(q, params).df()

        if out.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "identifier"])

        out["timestamp"] = _ts_ms_to_iso(out["ts_ms"])
        out["identifier"] = out["timestamp"]
        return out

    # -----------------------
    # Market Trades (tick-level)
    # -----------------------

    def append_market_trades(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Bulk insert market trades (tick-level data).

        Expected columns (flexible):
          - timestamp or ts
          - trade_id
          - px or price
          - sz or size
          - side

        Notes:
          - Rows with missing/empty trade_id are dropped.
          - Rows with invalid timestamps are dropped.
          - Uniqueness enforced by (symbol, trade_id) via constraint/index.
        """
        if df is None or df.empty:
            return

        tmp = df.copy()

        # Timestamp normalization
        if "timestamp" in tmp.columns:
            tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        elif "ts" in tmp.columns:
            tmp["ts_ms"] = _to_ts_ms(tmp["ts"])
        else:
            return  # no usable timestamp column

        # Standardize aliases
        tmp.rename(columns={"px": "price", "size": "sz"}, inplace=True)

        tmp["symbol"] = symbol

        req_cols = ["symbol", "ts_ms", "trade_id", "price", "sz", "side"]
        for c in req_cols:
            if c not in tmp.columns:
                tmp[c] = None

        insert_df = tmp[req_cols].copy()

        # Drop invalid ts
        insert_df = insert_df[insert_df["ts_ms"].notna()]
        insert_df["ts_ms"] = insert_df["ts_ms"].astype("int64")
        insert_df = insert_df[insert_df["ts_ms"] > 0]

        # Drop invalid trade_id
        insert_df["trade_id"] = insert_df["trade_id"].astype(str)
        insert_df = insert_df[insert_df["trade_id"].notna()]
        insert_df = insert_df[insert_df["trade_id"].str.strip() != ""]
        insert_df = insert_df[insert_df["trade_id"].str.lower() != "none"]
        insert_df = insert_df[insert_df["trade_id"].str.lower() != "nan"]

        if insert_df.empty:
            return

        insert_df["side"] = insert_df["side"].astype(str).str.lower()

        with self._lock:
            self.con.register("trades_in", insert_df)
            try:
                self.con.execute(
                    """
                    INSERT INTO market_trades
                    SELECT * FROM trades_in
                    ON CONFLICT (symbol, trade_id) DO NOTHING;
                    """
                )
            except Exception:
                self.con.execute(
                    """
                    INSERT INTO market_trades
                    SELECT t.*
                    FROM trades_in t
                    LEFT JOIN market_trades m
                      ON m.symbol = t.symbol AND m.trade_id = t.trade_id
                    WHERE m.trade_id IS NULL;
                    """
                )
            self.con.unregister("trades_in")

    # -----------------------
    # Footprint aggregation
    # -----------------------

    def load_footprint_data(
        self,
        symbol: str,
        bar: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Compute Footprint (price-by-volume) data inside DuckDB.

        Returns DataFrame columns:
          - identifier (ISO timestamp of the bucket)
          - price
          - bid_size  (aggressive sells)
          - ask_size  (aggressive buys)
          - trade_count
        """
        bar_seconds = _bar_to_seconds(bar)
        interval_ms = int(bar_seconds) * 1000

        query = f"""
        WITH max_ts AS (
            SELECT MAX(ts_ms) AS m
            FROM market_trades
            WHERE symbol = ?
        ),
        timeframe_trades AS (
            SELECT
                (ts_ms // {interval_ms}) * {interval_ms} AS bucket_ts,
                price,
                side,
                sz
            FROM market_trades
            WHERE symbol = ?
              AND ts_ms > (SELECT m FROM max_ts) - ({int(limit)} * {interval_ms})
        )
        SELECT
            bucket_ts,
            price,
            SUM(CASE WHEN side = 'sell' THEN sz ELSE 0 END) AS bid_size,
            SUM(CASE WHEN side = 'buy'  THEN sz ELSE 0 END) AS ask_size,
            COUNT(*) AS trade_count
        FROM timeframe_trades
        GROUP BY bucket_ts, price
        ORDER BY bucket_ts DESC, price ASC
        """

        with self._lock:
            df = self.con.execute(query, [symbol, symbol]).df()

        if df.empty:
            return pd.DataFrame(columns=["identifier", "price", "bid_size", "ask_size", "trade_count"])

        df["identifier"] = _ts_ms_to_iso(df["bucket_ts"])
        df = df.drop(columns=["bucket_ts"])
        return df
