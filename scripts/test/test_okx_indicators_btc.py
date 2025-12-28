# scripts/test/test_okx_indicators_btc.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---- Make project root importable (fix: ModuleNotFoundError: No module named 'data')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.okx.store import OkxPersistStore
from data.okx.indicators import QuantitativeIndicator

# OKX SDK (you already use okx.MarketData elsewhere)
import okx.MarketData as MarketData


def fetch_okx_ohlcv(symbol: str, bar: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetch real OHLCV from OKX via official python sdk wrapper.

    Returns df with columns:
      timestamp, open, high, low, close, volume
    where timestamp is int ms (as returned by OKX).
    """
    api = MarketData.MarketAPI(flag="0")  # "0" = live, "1" = demo (depends on your OKX setup)
    resp = api.get_candlesticks(instId=symbol, bar=bar, limit=str(limit))

    # OKX SDK response usually looks like {"code":"0","data":[[ts, o, h, l, c, vol, ...], ...]}
    if not isinstance(resp, dict) or resp.get("code") != "0":
        raise RuntimeError(f"OKX get_candlesticks failed: {resp}")

    rows = resp.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # OKX returns newest-first; make it ascending
    rows = list(reversed(rows))

    out = pd.DataFrame(rows)
    if out.shape[1] < 6:
        raise RuntimeError(f"Unexpected kline row format, got columns={out.shape[1]} row0={rows[0]}")

    out = out.iloc[:, :6]
    out.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # normalize dtypes
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return out


def ensure_duckdb_has_data(
    store: OkxPersistStore,
    symbol: str,
    bar: str,
    limit: int = 300,
) -> pd.DataFrame:
    """
    Load from DuckDB; if empty, fetch from OKX and upsert, then reload.
    Hard-fail if still empty (no fake fallback).
    """
    df = store.load_ohlcv(symbol=symbol, bar=bar, limit=limit)
    if df is not None and not df.empty:
        return df

    print(">>> DuckDB is empty for this symbol/bar. Fetching from OKX and persisting...")
    live = fetch_okx_ohlcv(symbol=symbol, bar=bar, limit=max(limit, 200))
    if live.empty:
        raise RuntimeError("OKX returned empty OHLCV; cannot run real-data test.")

    # NOTE: upsert expects 'timestamp' column; can be int ms or ISO str; your _to_ts_ms handles both.
    store.upsert_ohlcv(symbol=symbol, bar=bar, df=live)

    df2 = store.load_ohlcv(symbol=symbol, bar=bar, limit=limit)
    if df2 is None or df2.empty:
        raise RuntimeError(
            "After upsert, DuckDB is still empty. Check DB path, table schema, or symbol/bar keys."
        )
    return df2


def main():
    symbol = os.getenv("OKX_SYMBOL", "BTC-USDT")
    bar = os.getenv("OKX_BAR", "1m")
    db_path = os.getenv("OKX_DUCKDB", "./data/okx.duckdb")
    limit = int(os.getenv("OKX_LIMIT", "300"))

    print("\n==============================")
    print("Testing: data/okx/indicators.py (QuantitativeIndicator) [REAL DATA ONLY]")
    print(f"Symbol: {symbol}, Bar: {bar}, DuckDB: {db_path}, Limit: {limit}")
    print("==============================\n")

    with OkxPersistStore(path=db_path, read_only=False) as store:
        # 1) Ensure real OHLCV exists in DuckDB
        ohlcv = ensure_duckdb_has_data(store, symbol=symbol, bar=bar, limit=limit)

    # 2) Compute indicators
    qi = QuantitativeIndicator()
    feat = qi.compute_all(ohlcv, sentiment_data=None)

    # 3) Print tails
    print("--- Raw OHLCV (tail 5) ---")
    print(ohlcv.tail(5).to_string(index=False))

    cols_to_show = ["timestamp", "close", "MA_7", "MA_25", "RSI", "MACD_dif", "MACD_dea", "MACD_hist"]
    cols_exist = [c for c in cols_to_show if c in feat.columns]

    print("\n--- OHLCV + Indicators (tail 10) ---")
    print(feat[cols_exist].tail(10).to_string(index=False))

    # 4) Minimal assertions (real-data sanity)
    assert "RSI" in feat.columns, "RSI column missing"
    assert "MACD_dif" in feat.columns and "MACD_dea" in feat.columns and "MACD_hist" in feat.columns, "MACD columns missing"
    assert len(feat) >= 30, "Need at least ~30 bars for stable indicator sanity checks"

    # RSI should not be all NaN with real BTC data unless data is pathological
    rsi_valid = feat["RSI"].dropna()
    assert len(rsi_valid) > 0, "RSI is all NaN. Check timestamp ordering / close values / RSI implementation."

    print("\n>>> PASS: Real-data indicators computed successfully.\n")


if __name__ == "__main__":
    main()
