# data/okx/store.py
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional

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

    # 1) Already datetime64
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
        return (dt.astype("int64") // 1_000_000).astype("int64")

    # 2) Numeric (heuristic detection: seconds vs ms)
    if pd.api.types.is_numeric_dtype(series):
        s = series.fillna(0).astype("int64")
        sample = int(s.iloc[0]) if len(s) > 0 else 0

        # Heuristic:
        # - seconds around year 2286 ~ 1e10
        # - milliseconds around year 1973 ~ 1e11
        # If < 1e11, assume seconds
        if 0 < sample < 100_000_000_000:
            return (s * 1000).astype("int64")
        return s

    # 3) Strings / Objects (ISO parsing)
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return (dt.astype("int64") // 1_000_000).fillna(0).astype("int64")


def _ts_ms_to_iso(ts_ms: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts_ms.astype("int64"), unit="ms", utc=True, errors="coerce")
    return dt.astype(str)


@dataclass
class OkxPersistStore:
    """
    Persistence layer (DuckDB as primary store, optional Parquet export).
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

    def _init_schema(self):
        # OHLCV: Primary Key effectively (symbol, bar, ts_ms)
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

        # Market Trades (for Footprint Chart)
        self._init_market_trades_schema()

    def _init_market_trades_schema(self) -> None:
        """
        Create market_trades table + indexes, enforcing uniqueness on (symbol, trade_id).

        Strategy:
        1) Prefer UNIQUE constraint in CREATE TABLE.
        2) Fallback to UNIQUE INDEX if constraint syntax/enforcement is unsupported.
        """
        # Try table with UNIQUE constraint first
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
            # Fallback: create table without constraint
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

        # Time-based index for fast aggregations
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_mkt_trades_ts ON market_trades(symbol, ts_ms);")

        # Fallback uniqueness enforcement via UNIQUE INDEX (if supported)
        try:
            self.con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_mkt_trades_uniq ON market_trades(symbol, trade_id);"
            )
        except Exception:
            pass

    # -----------------------
    # OHLCV (Optimized Upsert)
    # -----------------------

    def upsert_ohlcv(self, symbol: str, bar: str, df: pd.DataFrame) -> None:
        """
        Upsert OHLCV bars using DuckDB ON CONFLICT upsert.
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

        if self.parquet_dir:
            self._export_ohlcv_parquet(symbol=symbol, bar=bar)

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
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        out["timestamp"] = _ts_ms_to_iso(out["ts_ms"])
        out["identifier"] = out["timestamp"]
        return out

    def _export_ohlcv_parquet(self, symbol: str, bar: str):
        out_dir = os.path.join(self.parquet_dir, "ohlcv")
        os.makedirs(out_dir, exist_ok=True)
        with self._lock:
            self.con.execute(
                f"""
                COPY (
                    SELECT
                        symbol,
                        bar,
                        ts_ms,
                        open, high, low, close, volume,
                        strftime(to_timestamp(ts_ms/1000), '%Y-%m-%d') AS date
                    FROM ohlcv
                    WHERE symbol='{symbol}' AND bar='{bar}'
                )
                TO '{out_dir}'
                (FORMAT PARQUET, PARTITION_BY (symbol, bar, date), OVERWRITE_OR_IGNORE 1);
                """
            )

    # -----------------------
    # Market Trades (Unique by symbol + trade_id)
    # -----------------------

    def append_market_trades(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Bulk insert market trades (tick-level data).

        Expected columns (flexible):
          - timestamp or ts
          - trade_id
          - px or price
          - sz or size
          - side  (e.g., 'buy'/'sell')

        Data hygiene:
          - Rows with missing/empty trade_id are dropped (uniqueness requires a real id).
          - Rows with missing timestamps are dropped (cannot bucket/aggregate).

        Uniqueness:
          - Enforced by (symbol, trade_id) via UNIQUE constraint or UNIQUE INDEX (depending on DuckDB version).
          - Preferred insert path: ON CONFLICT DO NOTHING.
          - Fallback insert path: anti-join to avoid duplicates.
        """
        if df is None or df.empty:
            return

        tmp = df.copy()

        # Support different timestamp column names
        if "timestamp" in tmp.columns:
            tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        elif "ts" in tmp.columns:
            tmp["ts_ms"] = _to_ts_ms(tmp["ts"])
        else:
            return  # No usable timestamp column

        # Standardize common column aliases
        col_map = {"px": "price", "size": "sz"}
        tmp.rename(columns=col_map, inplace=True)

        tmp["symbol"] = symbol

        req_cols = ["symbol", "ts_ms", "trade_id", "price", "sz", "side"]
        for c in req_cols:
            if c not in tmp.columns:
                tmp[c] = None

        insert_df = tmp[req_cols].copy()

        # ---- Data hygiene / filtering ----
        # 1) Drop rows with invalid timestamps (0 is produced by _to_ts_ms when parsing fails)
        insert_df = insert_df[insert_df["ts_ms"].notna()]
        insert_df["ts_ms"] = insert_df["ts_ms"].astype("int64")
        insert_df = insert_df[insert_df["ts_ms"] > 0]

        # 2) Drop rows with missing/empty trade_id
        # (UNIQUE with NULL doesn't behave as "dedup"; most SQL engines allow multiple NULLs)
        insert_df["trade_id"] = insert_df["trade_id"].astype(str)
        insert_df = insert_df[insert_df["trade_id"].notna()]
        insert_df = insert_df[insert_df["trade_id"].str.strip() != ""]
        insert_df = insert_df[insert_df["trade_id"].str.lower() != "none"]
        insert_df = insert_df[insert_df["trade_id"].str.lower() != "nan"]

        if insert_df.empty:
            return

        # Optional: normalize side values to lower-case for consistent aggregation
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
                # Compatibility fallback: anti-join insert
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

    def load_footprint_data(
        self,
        symbol: str,
        bar_seconds: int = 60,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Core method: compute Footprint (price-by-volume) data directly inside DuckDB.

        Args:
            symbol: Trading pair / instrument symbol
            bar_seconds: Bar interval in seconds (1m = 60)
            limit: Number of most recent bars to load

        Returns:
            DataFrame with columns:
              - identifier (ISO timestamp for the bar bucket)
              - price
              - bid_size  (aggressive sells)
              - ask_size  (aggressive buys)
              - trade_count
        """
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

    # -----------------------
    # Funding history
    # -----------------------

    def append_funding_history(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        tmp = df.copy()
        tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        tmp["symbol"] = symbol
        if "realized_rate" not in tmp.columns:
            tmp["realized_rate"] = None

        insert_df = tmp[["symbol", "ts_ms", "funding_rate", "realized_rate"]]

        with self._lock:
            self.con.register("fund_in", insert_df)
            self.con.execute(
                """
                INSERT INTO funding
                SELECT symbol, ts_ms, funding_rate, realized_rate FROM fund_in
                ON CONFLICT (symbol, ts_ms) DO UPDATE SET
                    funding_rate = EXCLUDED.funding_rate,
                    realized_rate = EXCLUDED.realized_rate;
                """
            )
            self.con.unregister("fund_in")

        if self.parquet_dir:
            # Optional export logic can be added here
            pass

    # -----------------------
    # Fills / Signals (Append Only)
    # -----------------------

    def append_fills(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        tmp = df.copy()
        tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        if "extra_json" not in tmp.columns:
            tmp["extra_json"] = None

        cols = ["strategy", "symbol", "ts_ms", "order_id", "side", "price", "qty", "fee", "fee_ccy", "extra_json"]
        for c in cols:
            if c not in tmp.columns:
                tmp[c] = None

        with self._lock:
            self.con.register("fills_in", tmp[cols])
            self.con.execute("INSERT INTO fills SELECT * FROM fills_in;")
            self.con.unregister("fills_in")

    def append_signals(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        tmp = df.copy()
        tmp["ts_ms"] = _to_ts_ms(tmp["timestamp"])
        if "extra_json" not in tmp.columns:
            tmp["extra_json"] = None

        cols = ["strategy", "symbol", "ts_ms", "signal_name", "value", "decision", "reason", "extra_json"]
        for c in cols:
            if c not in tmp.columns:
                tmp[c] = None

        with self._lock:
            self.con.register("sig_in", tmp[cols])
            self.con.execute("INSERT INTO signals SELECT * FROM sig_in;")
            self.con.unregister("sig_in")
