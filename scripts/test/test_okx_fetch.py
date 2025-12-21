# scripts/test/test_okx_fetch_live_check.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

# Add project root to sys.path so `import data...` works when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.okx.fetch import OkxFetcher  # noqa: E402


def ms_to_times(ms_str: str) -> dict:
    """Convert epoch ms string to UTC + Asia/Shanghai time strings."""
    try:
        ms = int(ms_str)
        dt_utc = pd.to_datetime(ms, unit="ms", utc=True)
        dt_cn = dt_utc.tz_convert("Asia/Shanghai")
        return {
            "ms": ms,
            "utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "cn": dt_cn.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return {"ms": None, "utc": None, "cn": None}


def main() -> None:
    inst_id = "BTC-USDT-SWAP"
    bar = "1m"

    # flag="0" live, flag="1" sim
    f = OkxFetcher(flag="0", sleep_s=0.05)

    # Fetch last 30 minutes
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=30)

    print("=== OKX LIVE CHECK ===")
    print(f"inst_id={inst_id} bar={bar}")
    print(f"range_utc: {start.strftime('%Y-%m-%d %H:%M:%S')} -> {end.strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------
    # 1) OHLCV
    # -------------------------
    df = f.fetch_ohlcv(
        inst_id=inst_id,
        bar=bar,
        start=start,
        end=end,
        limit=300,
        max_pages=50,
    )

    print("\n=== [1] LIVE OHLCV (persisted fields) ===")
    print(f"rows={len(df)} columns={list(df.columns)}")

    if df.empty:
        print("OHLCV is EMPTY. If ticker works but candles are empty, it's usually parameter/SDK mismatch.")
        return

    # Add dual timezone view for manual verification
    df_view = df.copy()
    df_view["timestamp_utc"] = pd.to_datetime(df_view["timestamp"], utc=True)
    df_view["timestamp_cn"] = (
        df_view["timestamp_utc"]
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    df_view["timestamp_utc"] = df_view["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print("\n--- last 10 bars (UTC + CN) ---")
    print(
        df_view[["timestamp_utc", "timestamp_cn", "open", "high", "low", "close", "volume"]]
        .tail(10)
        .to_string(index=False)
    )

    last_candle = df.iloc[-1].to_dict()
    print("\n--- last candle (persisted) ---")
    print(last_candle)

    # -------------------------
    # 2) Ticker
    # -------------------------
    print("\n=== [2] LIVE Ticker snapshot ===")
    try:
        t = f._fetch_ticker_one(inst_id)
    except Exception as e:
        print("ticker fetch failed:", repr(e))
        return

    if not isinstance(t, dict) or not t:
        print("ticker empty/unexpected:", t)
        return

    ts_info = ms_to_times(t.get("ts", "0"))
    snap = {
        "instId": t.get("instId"),
        "last": t.get("last"),
        "bidPx": t.get("bidPx"),
        "askPx": t.get("askPx"),
        "markPx": t.get("markPx"),
        "idxPx": t.get("idxPx"),
        "ts_ms": ts_info["ms"],
        "ts_utc": ts_info["utc"],
        "ts_cn": ts_info["cn"],
    }
    print(snap)

    # Cross-check candle close vs ticker last
    try:
        candle_close = float(df["close"].iloc[-1])
        ticker_last = float(t.get("last", 0) or 0)
        diff = ticker_last - candle_close
        diff_pct = diff / candle_close * 100 if candle_close != 0 else float("nan")
        print("\n=== [3] Cross-check (candle_close vs ticker_last) ===")
        print(
            {
                "candle_close": candle_close,
                "ticker_last": ticker_last,
                "diff": diff,
                "diff_pct": f"{diff_pct:.6f}%",
                "note": "Small diff is normal (candle close vs live last sampled at different instants).",
            }
        )
    except Exception as e:
        print("cross-check failed:", repr(e))

    # -------------------------
    # 3) Orderbook top1
    # -------------------------
    print("\n=== [4] LIVE Orderbook top1 ===")
    try:
        ob = f.get_orderbook_cached(inst_id, depth=1, ttl_s=1.0)
        best_bid = ob["bids"][0] if ob.get("bids") else None
        best_ask = ob["asks"][0] if ob.get("asks") else None
        print({"timestamp": ob.get("timestamp"), "best_bid": best_bid, "best_ask": best_ask})
    except Exception as e:
        print("orderbook failed:", repr(e))



if __name__ == "__main__":
    main()
