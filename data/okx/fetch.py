# data/okx/fetch.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import time

import pandas as pd

from .core import OkxBaseMixin
from .state import OkxRuntimeState  # <-- from the state.py I gave you

# Optional Rubik (analytics / long-short ratio, etc.)
try:
    import okx.Rubik as Rubik  # type: ignore
except Exception:
    Rubik = None


# ----------------------------
# Time helpers
# ----------------------------
def _to_ms(ts: pd.Timestamp | datetime) -> int:
    """Convert a pandas Timestamp or datetime to epoch milliseconds (UTC)."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def _now_ms() -> int:
    """Current epoch milliseconds (UTC)."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_str(ms: int) -> str:
    """Convert epoch milliseconds to UTC string '%Y-%m-%d %H:%M:%S'."""
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Fetch config (OHLCV)
# ----------------------------
@dataclass
class OkxFetchConfig:
    bar: str                 # e.g. '1m', '5m', '1H', '1D'
    limit: int = 300         # per-request limit
    max_pages: int = 2000    # safety guard


class OkxFetcher(OkxBaseMixin):
    """
    OKX Fetcher:
    - Persisted data (disk): OHLCV only (via your Finstore layer)
      OHLCV DataFrame columns MUST be:
      ['timestamp','open','high','low','close','volume']  (timestamp UTC string)

    - Non-persisted data (in-memory, runtime): stored in self.state
      (rankings, orderbook, best bid/ask, mark/index/last, spreads, funding snapshots, OI, L/S ratio, etc.)
    """

    def __init__(
        self,
        flag: str = "0",
        sleep_s: float = 0.05,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ) -> None:
        super().__init__(flag=flag)
        self.sleep_s = sleep_s
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # In-memory runtime state (NOT persisted)
        self.state = OkxRuntimeState()

        # Optional Rubik API
        self.rubik_api = None
        if Rubik is not None:
            try:
                self.rubik_api = Rubik.RubikAPI(flag=self.flag)
            except Exception:
                self.rubik_api = None

    # ============================================================
    # Internal: robust OKX call wrapper (works even if core.py doesn't provide it)
    # ============================================================
    def _call_with_retry(
        self,
        fn: Callable[..., Any],
        ctx: str,
        **kwargs: Any,
    ) -> Any:
        """
        Call an OKX SDK method with retries + OKX response validation.
        Returns resp["data"] (list/dict) when code == "0".
        """
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = fn(**kwargs)

                # Many OKX SDK calls return dict: {"code":"0","data":[...],...}
                # Some might already return data; handle both.
                if isinstance(resp, dict):
                    resp = self._require_ok(resp, ctx)
                    return resp.get("data", [])
                return resp

            except Exception as e:
                last_err = e
                # exponential backoff
                sleep_time = self.backoff_factor * (2 ** attempt)
                time.sleep(max(self.sleep_s, sleep_time))

        raise RuntimeError(f"{ctx}: failed after {self.max_retries} retries. last_err={last_err}")

    # ============================================================
    # 1) OHLCV (persisted)
    # ============================================================
    def fetch_ohlcv(
    self,
    inst_id: str,
    bar: str,
    start: datetime | pd.Timestamp,
    end: Optional[datetime | pd.Timestamp] = None,
    limit: int = 300,
    max_pages: int = 2000,
    ) -> pd.DataFrame:
        """
        Fetch OKX OHLCV (candles) and paginate backwards using `before`
        until [start, end] is covered.

        Returns DataFrame columns:
        ['timestamp','open','high','low','close','volume']
        timestamp is '%Y-%m-%d %H:%M:%S' (UTC)
        """
        start_ms = _to_ms(start)
        end_ms = _to_ms(end) if end is not None else _now_ms()

        rows: List[list] = []
        pages = 0

        # IMPORTANT: Many OKX SDK versions behave badly with after=""
        # So we DO NOT pass 'after' at all.
        # First request: do not pass before either -> let OKX return latest candles.
        before: Optional[int] = None

        while pages < max_pages:
            pages += 1

            kwargs: Dict[str, Any] = {
                "instId": inst_id,
                "bar": bar,
                "limit": str(limit),
            }
            if before is not None:
                kwargs["before"] = str(before)

            data = self._call_with_retry(
                fn=self.mkt_api.get_candlesticks,
                ctx=f"OKX get_candlesticks instId={inst_id} bar={bar}",
                **kwargs,
            )

            if not data:
                break

            rows.extend(data)

            # OKX candles: each row like [ts, o, h, l, c, vol, ...]
            try:
                ts_list = [int(r[0]) for r in data if r and r[0] is not None]
            except Exception:
                break

            oldest_ms = min(ts_list)

            # If we've reached the requested start time, stop
            if oldest_ms <= start_ms:
                break

            # Guard infinite loops
            if before is not None and oldest_ms >= before:
                break

            # Next page: go further back in time
            before = oldest_ms

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Normalize: [ts, o, h, l, c, vol]
        norm: List[List[Any]] = []
        for r in rows:
            if not r or len(r) < 6:
                continue
            try:
                norm.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
            except Exception:
                continue

        df = pd.DataFrame(norm, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Dedup + ascending
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Clip to [start, end]
        df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]

        # Format timestamp as UTC string
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        return df.reset_index(drop=True)


    # ============================================================
    # 2) Rankings (NOT persisted): top gainers / top volume
    # ============================================================
    def get_top_gainers_swap_usdt(self, limit: int = 20) -> pd.DataFrame:
        """
        Query OKX immediately: SWAP perpetual top gainers (USDT-margined).
        Returns DataFrame: ['symbol','change_24h','price']
        """
        data = self._call_with_retry(
            fn=self.mkt_api.get_tickers,
            ctx="OKX get_tickers instType=SWAP",
            instType="SWAP",
        )

        rows: List[Dict[str, Any]] = []
        for t in data or []:
            inst = t.get("instId", "")
            if not inst.endswith("USDT-SWAP"):
                continue
            try:
                last_price = float(t.get("last", 0) or 0)
                open_24h = float(t.get("open24h", 0) or 0)
                change = (last_price - open_24h) / open_24h if open_24h > 0 else 0.0
                rows.append({"symbol": inst, "change_24h": change, "price": last_price})
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "change_24h", "price"])
        return df.sort_values("change_24h", ascending=False).head(limit).reset_index(drop=True)

    def get_top_volume_swap_usdt(self, limit: int = 20) -> pd.DataFrame:
        """
        Query OKX immediately: SWAP perpetual top volume (USDT-margined).
        Returns DataFrame: ['symbol','volume_24h','price']
        """
        data = self._call_with_retry(
            fn=self.mkt_api.get_tickers,
            ctx="OKX get_tickers instType=SWAP",
            instType="SWAP",
        )

        rows: List[Dict[str, Any]] = []
        for t in data or []:
            inst = t.get("instId", "")
            if not inst.endswith("USDT-SWAP"):
                continue
            try:
                vol_24h_usdt = float(t.get("volCcy24h", 0) or 0)
                last_price = float(t.get("last", 0) or 0)
                rows.append({"symbol": inst, "volume_24h": vol_24h_usdt, "price": last_price})
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "volume_24h", "price"])
        return df.sort_values("volume_24h", ascending=False).head(limit).reset_index(drop=True)

    def get_top_gainers_cached(self, limit: int = 20) -> pd.DataFrame:
        cache = self.state.top_gainers
        if cache.is_fresh() and isinstance(cache.value, pd.DataFrame):
            return cache.value.head(limit)
        df = self.get_top_gainers_swap_usdt(limit=limit)
        cache.set(df)
        return df

    def get_top_volume_cached(self, limit: int = 20) -> pd.DataFrame:
        cache = self.state.top_volume
        if cache.is_fresh() and isinstance(cache.value, pd.DataFrame):
            return cache.value.head(limit)
        df = self.get_top_volume_swap_usdt(limit=limit)
        cache.set(df)
        return df

    # ============================================================
    # 3) Orderbook (NOT persisted): snapshot + cached wrapper
    # ============================================================
    def fetch_orderbook(self, inst_id: str, depth: int = 10) -> Dict[str, Any]:
        """
        Query OKX immediately: orderbook snapshot.
        Returns dict:
        {'bids': [[price,size],...], 'asks': [[price,size],...], 'timestamp': '...'}
        """
        data = self._call_with_retry(
            fn=self.mkt_api.get_order_books,
            ctx=f"OKX get_order_books instId={inst_id}",
            instId=inst_id,
            sz=str(depth),
        )

        if not data:
            return {"bids": [], "asks": [], "timestamp": "1970-01-01 00:00:00"}

        ob = data[0]

        def parse_depth(items: List[List[str]]) -> List[List[float]]:
            out: List[List[float]] = []
            for item in items or []:
                try:
                    out.append([float(item[0]), float(item[1])])
                except Exception:
                    continue
            return out

        ts_ms = int(ob.get("ts", 0) or 0)
        return {
            "bids": parse_depth(ob.get("bids", [])),
            "asks": parse_depth(ob.get("asks", [])),
            "timestamp": _ms_to_str(ts_ms) if ts_ms else "1970-01-01 00:00:00",
        }

    def get_orderbook_cached(self, inst_id: str, depth: int = 10, ttl_s: float = 1.5) -> Dict[str, Any]:
        """
        Cached orderbook (uses RuntimeState bounded LRU cache).
        """
        cache = self.state.ob_cache(inst_id, ttl_s=ttl_s)
        if cache.is_fresh() and isinstance(cache.value, dict):
            return cache.value

        ob = self.fetch_orderbook(inst_id=inst_id, depth=depth)
        cache.set(ob)
        self.state.last_update_by_key[f"orderbook:{inst_id}"] = time.time()
        return ob

    # ============================================================
    # 4) Best bid/ask + spread (NOT persisted): cached
    # ============================================================
    @staticmethod
    def _compute_spread_bps(bid: float, ask: float) -> float:
        """
        Spread in basis points using mid price.
        """
        if bid <= 0 or ask <= 0:
            return float("inf")
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return float("inf")
        return (ask - bid) / mid * 10_000.0

    def _fetch_ticker_one(self, inst_id: str) -> Dict[str, Any]:
        """
        Fetch single ticker if SDK supports get_ticker; otherwise fallback to get_tickers+filter.
        Returns the matched ticker dict or {}.
        """
        # Try single-instrument endpoint first (more efficient)
        if hasattr(self.mkt_api, "get_ticker"):
            data = self._call_with_retry(
                fn=self.mkt_api.get_ticker,
                ctx=f"OKX get_ticker instId={inst_id}",
                instId=inst_id,
            )
            # get_ticker typically returns list with one dict
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict):
                return data
            return {}

        # Fallback: scan from all tickers (SWAP)
        data = self._call_with_retry(
            fn=self.mkt_api.get_tickers,
            ctx="OKX get_tickers instType=SWAP (fallback for get_ticker)",
            instType="SWAP",
        )
        for t in data or []:
            if t.get("instId") == inst_id:
                return t
        return {}

    def get_best_bid_ask_cached(self, inst_id: str, ttl_s: float = 1.0) -> Dict[str, Any]:
        """
        Cached best bid/ask snapshot.
        Returns: {'bid': float, 'ask': float, 'timestamp': str}
        """
        cache = self.state.best_bid_ask_cache(inst_id, ttl_s=ttl_s)
        if cache.is_fresh() and isinstance(cache.value, dict):
            return cache.value

        t = self._fetch_ticker_one(inst_id)

        bid = float(t.get("bidPx", 0) or 0)
        ask = float(t.get("askPx", 0) or 0)
        ts_ms = int(t.get("ts", 0) or 0)

        # Fallback to orderbook top-of-book if ticker doesn't provide bid/ask
        if bid <= 0 or ask <= 0:
            ob = self.get_orderbook_cached(inst_id, depth=1, ttl_s=min(ttl_s, 1.5))
            bid = float(ob["bids"][0][0]) if ob.get("bids") else 0.0
            ask = float(ob["asks"][0][0]) if ob.get("asks") else 0.0
            # orderbook timestamp already string
            value = {"bid": bid, "ask": ask, "timestamp": ob.get("timestamp", "1970-01-01 00:00:00")}
        else:
            value = {"bid": bid, "ask": ask, "timestamp": _ms_to_str(ts_ms) if ts_ms else "1970-01-01 00:00:00"}

        cache.set(value)
        self.state.last_update_by_key[f"bbo:{inst_id}"] = time.time()

        # Update spread cache together
        spread_cache = self.state.spread_bps_cache(inst_id, ttl_s=ttl_s)
        spread_cache.set(self._compute_spread_bps(value["bid"], value["ask"]))
        return value

    def get_spread_bps_cached(self, inst_id: str, ttl_s: float = 1.0) -> float:
        """
        Cached spread (bps). Will also refresh BBO if needed.
        """
        cache = self.state.spread_bps_cache(inst_id, ttl_s=ttl_s)
        if cache.is_fresh() and cache.value is not None:
            try:
                return float(cache.value)
            except Exception:
                pass

        bbo = self.get_best_bid_ask_cached(inst_id, ttl_s=ttl_s)
        spread = self._compute_spread_bps(float(bbo["bid"]), float(bbo["ask"]))
        cache.set(spread)
        return spread

    # ============================================================
    # 5) Mark / Index / Last + Basis (NOT persisted): cached
    # ============================================================
    def get_mark_index_last_cached(self, inst_id: str, ttl_s: float = 1.0) -> Dict[str, Any]:
        """
        Cached mark/index/last snapshot.
        Returns: {'mark': float, 'index': float, 'last': float, 'timestamp': str}
        """
        cache = self.state.mark_index_last_cache(inst_id, ttl_s=ttl_s)
        if cache.is_fresh() and isinstance(cache.value, dict):
            return cache.value

        t = self._fetch_ticker_one(inst_id)

        last_px = float(t.get("last", 0) or 0)
        mark_px = float(t.get("markPx", 0) or 0)
        idx_px = float(t.get("idxPx", 0) or 0)
        ts_ms = int(t.get("ts", 0) or 0)

        value = {
            "mark": mark_px,
            "index": idx_px,
            "last": last_px,
            "timestamp": _ms_to_str(ts_ms) if ts_ms else "1970-01-01 00:00:00",
        }

        cache.set(value)
        self.state.last_update_by_key[f"mil:{inst_id}"] = time.time()
        return value

    def get_basis_cached(self, inst_id: str, ttl_s: float = 1.0, basis_type: str = "mark_minus_index") -> float:
        """
        Cached basis:
        - 'mark_minus_index' (default)
        - 'last_minus_index'
        """
        mil = self.get_mark_index_last_cached(inst_id, ttl_s=ttl_s)
        mark_px = float(mil.get("mark", 0) or 0)
        idx_px = float(mil.get("index", 0) or 0)
        last_px = float(mil.get("last", 0) or 0)

        if basis_type == "last_minus_index":
            return last_px - idx_px if idx_px != 0 else 0.0
        return mark_px - idx_px if idx_px != 0 else 0.0

    # ============================================================
    # 6) Funding (history optional persisted) + (snapshot cached if you want)
    # ============================================================
    def fetch_funding_rate_history(self, inst_id: str, limit: int = 100) -> pd.DataFrame:
        """
        Funding rate history (time series).
        Returns DataFrame: ['timestamp','symbol','funding_rate','realized_rate']
        """
        data = self._call_with_retry(
            fn=self.mkt_api.get_funding_rate_history,
            ctx=f"OKX get_funding_rate_history instId={inst_id}",
            instId=inst_id,
            limit=str(limit),
        )

        rows: List[Dict[str, Any]] = []
        for item in data or []:
            try:
                funding_time = int(item.get("fundingTime"))
                rows.append({
                    "timestamp": _ms_to_str(funding_time),
                    "symbol": item.get("instId", inst_id),
                    "funding_rate": float(item.get("fundingRate")),
                    "realized_rate": float(item.get("realizedRate", 0) or 0),
                })
            except Exception:
                continue

        return pd.DataFrame(rows)

    def funding_rate_as_indicator_df(self, inst_id: str, limit: int = 100) -> pd.DataFrame:
        """
        Convert funding rate history to indicator long format:
        ['timestamp','indicator_name','indicator_value']
        """
        fr = self.fetch_funding_rate_history(inst_id=inst_id, limit=limit)
        if fr.empty:
            return pd.DataFrame(columns=["timestamp", "indicator_name", "indicator_value"])
        out = fr[["timestamp", "funding_rate"]].copy()
        out["indicator_name"] = "funding_rate"
        out = out.rename(columns={"funding_rate": "indicator_value"})
        return out[["timestamp", "indicator_name", "indicator_value"]]

    # Optional: funding snapshot cache (if your SDK has a direct endpoint, wire it here)
    # For now we keep it generic: you can set it from your own fetch if needed.
    def set_funding_snapshot(self, inst_id: str, rate: float, next_time_ms: int, ttl_s: float = 60.0) -> None:
        """
        Manually set funding snapshot in runtime state (NOT persisted).
        Useful if you fetch it elsewhere and want strategy access.
        """
        cache = self.state.funding_cache(inst_id, ttl_s=ttl_s)
        cache.set({"rate": float(rate), "next_time": _ms_to_str(next_time_ms)})
        self.state.last_update_by_key[f"funding:{inst_id}"] = time.time()

    # ============================================================
    # 7) Long/Short ratio (NOT persisted): cached
    # ============================================================
    def fetch_long_short_ratio_series(self, ccy: str, period: str = "5m") -> pd.DataFrame:
        """
        Query OKX Rubik immediately: elite trader long/short account ratio (time series).
        Returns DataFrame: ['timestamp','ccy','ratio']
        """
        if self.rubik_api is None:
            return pd.DataFrame(columns=["timestamp", "ccy", "ratio"])

        data = self._call_with_retry(
            fn=self.rubik_api.get_long_short_account_ratio,
            ctx=f"OKX rubik get_long_short_account_ratio ccy={ccy} period={period}",
            ccy=ccy,
            period=period,
        )

        rows: List[Dict[str, Any]] = []
        for item in data or []:
            try:
                ts = int(item.get("ts"))
                rows.append({"timestamp": _ms_to_str(ts), "ccy": ccy, "ratio": float(item.get("ratio"))})
            except Exception:
                continue
        return pd.DataFrame(rows)

    def get_long_short_ratio_cached(self, inst_id: str, period: str = "5m", ttl_s: float = 30.0) -> float:
        """
        Latest long/short ratio snapshot (NOT persisted).
        """
        ccy = inst_id.split("-")[0]
        cache = self.state.lsr_cache(ccy, ttl_s=ttl_s)
        if cache.is_fresh() and cache.value is not None:
            try:
                return float(cache.value)
            except Exception:
                pass

        df = self.fetch_long_short_ratio_series(ccy=ccy, period=period)
        ratio = 1.0
        if not df.empty and "ratio" in df.columns:
            try:
                ratio = float(df["ratio"].iloc[0])  # newest-first typical
            except Exception:
                ratio = 1.0

        cache.set(ratio)
        self.state.last_update_by_key[f"lsr:{ccy}"] = time.time()
        return ratio

    # ============================================================
    # 8) Maintenance helper (call in your loop)
    # ============================================================
    def cleanup_state(self) -> Dict[str, int]:
        """
        Cleanup very stale cache entries from runtime state.
        Suggestion: call every 30-60 seconds.
        """
        return self.state.cleanup()
