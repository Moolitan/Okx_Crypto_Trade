# data/okx/data.py
import time
import logging
from typing import Any, Dict, List, Optional
from collections import deque

import pandas as pd

# ✅ Unified imports:
# Prefer package-relative imports; fall back to local imports for direct script execution
try:
    from .core import OkxBaseMixin
    from .store import OkxPersistStore
except Exception:
    from core import OkxBaseMixin
    from store import OkxPersistStore

logger = logging.getLogger("OkxMarketDataBus")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class MarketDataBus(OkxBaseMixin):
    """
    Unified OKX market data bus.

    Responsibilities:
    - Collect incoming market data
    - Cache data in memory (OHLCV, order book, trades, sentiment, etc.)
    - Persist data to storage (DuckDB via OkxPersistStore)
    - Expose read-only interfaces for downstream consumers (e.g., indicators, strategies)

    Notes:
    - In-memory buffers: K-lines, order book, trades, sentiment
    - Persistence layer: DuckDB (OkxPersistStore)
    - ❌ No runtime state (RuntimeState). Runtime/strategy state should live in an external
      state.py or in the strategy process itself.
    """

    def __init__(
        self,
        exchange: str = "okx",
        base_symbol: str = "",
        flag: str = "0",
        sleep_s: float = 0.05,
        max_klines_mem: int = 5000,
        max_trades_mem: int = 5000,
        persist_path: str = "./data/okx.duckdb",
        bar: str = "1m",
    ):
        super().__init__(flag=flag, sleep_s=sleep_s)

        self.exchange = exchange
        self.base_symbol = base_symbol
        # Keep bar interval directly on the bus (not a runtime state object)
        self.bar = bar

        self.now_ts: float = time.time()
        self.last_update_ts: float = self.now_ts

        # K-line buffer
        self._kline_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        self._kline_buffer: deque = deque(maxlen=max_klines_mem)

        # Basic ticker data
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None

        self.order_book: Dict[str, List] = {"bids": [], "asks": []}
        self.trades: deque = deque(maxlen=max_trades_mem)

        # Derivatives-related data
        self.funding_rate: Optional[float] = None
        self.next_funding_time: Optional[int] = None

        # Sentiment / Rubik metrics
        self.sentiment_data: Dict[str, Any] = {
            "long_short_ratio": None,
            "top_long_short_ratio": None,
            "top_pos_long_short_ratio": None,
            "oi_token": None,
            "oi_usd": None,
            "taker_vol_buy": None,
            "taker_vol_sell": None,
        }

        # Market-wide (low-frequency) data
        self.market_tickers = pd.DataFrame()
        self.top_gainers_24h: List[Dict] = []
        self.top_losers_24h: List[Dict] = []
        self.top_volume_24h: List[Dict] = []
        self.funding_rates: Dict[str, float] = {}

        # Persistence backend
        self.store = OkxPersistStore(path=persist_path)

        logger.info(
            f"MarketDataBus initialized exchange={exchange}, symbol={base_symbol}, bar={bar}"
        )

    def _touch(self):
        self.now_ts = time.time()
        self.last_update_ts = self.now_ts

    # -------------------------
    # Update Interfaces (Write)
    # -------------------------

    def update_kline(self, kline_row: Dict[str, Any], *, is_closed: Optional[bool] = None):
        """
        Ingest a single K-line (candlestick) record.

        Expected keys:
        - timestamp
        - open
        - high
        - low
        - close
        - volume
        """
        if not kline_row or "timestamp" not in kline_row:
            return

        cleaned = {k: kline_row.get(k) for k in self._kline_cols}
        self._kline_buffer.append(cleaned)
        self._touch()

        # If you want to persist only on bar close, keep this hook
        if is_closed is True:
            self.persist_ohlcv_latest()

    def update_ticker(self, price: float, bid: float = None, ask: float = None):
        if price is not None:
            self.last_price = float(price)
        if bid is not None:
            self.bid_price = float(bid)
        if ask is not None:
            self.ask_price = float(ask)
        self._touch()

    def update_order_book(self, bids: List, asks: List):
        try:
            self.order_book["bids"] = [[float(p), float(s)] for p, s, *_ in bids]
            self.order_book["asks"] = [[float(p), float(s)] for p, s, *_ in asks]
            self._touch()
        except Exception as e:
            logger.error(f"Error updating order book: {e}")

    def update_trade(self, trade: Dict[str, Any]):
        if trade:
            self.trades.append(trade)
            self._touch()

    def update_sentiment(self, data_dict: Dict[str, Any]):
        if not data_dict:
            return
        self.sentiment_data.update(data_dict)
        self._touch()

    # -------------------------
    # Read Interfaces (For Indicators)
    # -------------------------

    def get_klines_df(self) -> pd.DataFrame:
        """
        Convert the in-memory K-line buffer to a pandas DataFrame.
        This is the primary data source for indicator calculations.
        """
        if not self._kline_buffer:
            return pd.DataFrame(columns=self._kline_cols)

        df = pd.DataFrame(list(self._kline_buffer), columns=self._kline_cols)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_latest_kline(self) -> Optional[Dict[str, Any]]:
        return self._kline_buffer[-1] if self._kline_buffer else None

    def get_mid_price(self) -> Optional[float]:
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return self.last_price

    def get_order_book_imbalance(self, depth: int = 5) -> float:
        if not self.order_book["bids"] or not self.order_book["asks"]:
            return 0.0

        bids = self.order_book["bids"][:depth]
        asks = self.order_book["asks"][:depth]

        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        denom = bid_vol + ask_vol

        return 0.0 if denom == 0 else (bid_vol - ask_vol) / denom

    # -------------------------
    # Persistence
    # -------------------------

    def persist_ohlcv_latest(self, *, bar: Optional[str] = None):
        """
        Upsert OHLCV data from the current in-memory buffer into DuckDB.
        Deduplication is handled by primary keys at the storage layer.
        """
        if not self.store:
            return

        df = self.get_klines_df()
        if df.empty:
            return

        use_bar = bar or self.bar or "1m"
        self.store.upsert_ohlcv(symbol=self.base_symbol, bar=use_bar, df=df)

    # -------------------------
    # Snapshot
    # -------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a lightweight snapshot of the latest market state.
        Useful for logging, monitoring, or strategy-level debugging.
        """
        snap = {
            "exchange": self.exchange,
            "symbol": self.base_symbol,
            "ts": self.now_ts,
            "bar": self.bar,
            "price": self.last_price,
            "funding": self.funding_rate,
            "ob_imbalance": self.get_order_book_imbalance(),
        }
        snap.update(self.sentiment_data)
        return snap

    def close(self):
        logger.info("Closing MarketDataBus...")
        if self.store:
            self.store.close()
