# data/okx/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import time
from collections import OrderedDict


# ============================================================
# TTL cache primitive
# ============================================================
@dataclass
class TTLValue:
    """
    A single cached value with TTL (time-to-live).

    Used for:
    - market snapshots
    - rankings
    - derived indicators
    - anything that should expire automatically

    value:
        Cached payload (dict / DataFrame / float / etc.)

    updated_at:
        Epoch seconds when last updated

    ttl_s:
        Seconds before the value is considered stale
    """
    value: Any = None
    updated_at: float = 0.0
    ttl_s: float = 5.0

    def is_fresh(self) -> bool:
        return (time.time() - self.updated_at) < self.ttl_s

    def set(self, v: Any) -> None:
        self.value = v
        self.updated_at = time.time()


# ============================================================
# Bounded LRU + TTL dictionary
# ============================================================
class LRUTTLDict:
    """
    Bounded mapping: key -> TTLValue, with LRU eviction.

    Design goals:
    - prevent memory growth
    - safe for long-running strategies
    - suitable for ~10 symbols, second-level updates

    Used heavily by:
    - market making
    - arbitrage
    - execution-layer caching
    """
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._data: "OrderedDict[str, TTLValue]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def get_or_create(self, key: str, ttl_s: float) -> TTLValue:
        """
        Get existing TTLValue or create a new one.
        Automatically updates LRU order and TTL.
        """
        if key in self._data:
            tv = self._data.pop(key)
            tv.ttl_s = ttl_s  # allow dynamic TTL adjustment
            self._data[key] = tv  # move to MRU
            return tv

        tv = TTLValue(ttl_s=ttl_s)
        self._data[key] = tv

        # Evict least-recently-used if over capacity
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

        return tv

    def cleanup(self, stale_factor: float = 3.0) -> int:
        """
        Remove very stale entries.

        A value is removed if:
            now - updated_at > ttl_s * stale_factor

        Called periodically from main loop.
        """
        now = time.time()
        to_delete = []

        for k, tv in self._data.items():
            if tv.updated_at <= 0:
                continue
            if (now - tv.updated_at) > (tv.ttl_s * stale_factor):
                to_delete.append(k)

        for k in to_delete:
            self._data.pop(k, None)

        return len(to_delete)


# ============================================================
# Runtime State for Strategies
# ============================================================
@dataclass
class OkxRuntimeState:
    """
    In-memory (NON-persisted) runtime state for strategy decisions.

    Designed for:
    - ~10 symbols
    - second-level loops
    - trend / mean-reversion / momentum / arbitrage / market making
    """

    # --------------------------------------------------------
    # Universe / symbol meta
    # --------------------------------------------------------

    active_symbols: set[str] = field(default_factory=set)
    # used by:
    # - all strategies (limit universe size)
    # - market making (fixed symbol pool)
    # - arbitrage (legs must be active)

    symbol_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # tickSize, lotSize, ctVal, etc.
    # used by:
    # - market making (valid quote prices)
    # - arbitrage (position sizing across legs)
    # - all execution logic

    disabled_until: Dict[str, float] = field(default_factory=dict)
    # symbol -> epoch seconds
    # used by:
    # - market making (illiquid / bad book)
    # - arbitrage (broken leg)
    # - all strategies (data errors / cooldowns)


    # --------------------------------------------------------
    # Rankings / global snapshots
    # --------------------------------------------------------

    top_gainers: TTLValue = field(default_factory=lambda: TTLValue(ttl_s=10))
    # used by:
    # - momentum (primary signal)
    # - trend (strong-symbol selection)
    # - mean reversion (inverse usage)

    top_volume: TTLValue = field(default_factory=lambda: TTLValue(ttl_s=10))
    # used by:
    # - market making (liquidity filter)
    # - momentum / trend (volume confirmation)
    # - arbitrage (execution safety)


    # --------------------------------------------------------
    # Quotes / prices
    # --------------------------------------------------------

    best_bid_ask: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # inst_id -> {bid, ask, ts}
    # used by:
    # - market making (core input)
    # - arbitrage (leg execution)
    # - all strategies (slippage-aware execution)

    mark_index_last: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # inst_id -> {mark, index, last, ts}
    # used by:
    # - arbitrage (basis / mispricing)
    # - market making (fair price)
    # - trend / momentum (price validation)
    # - mean reversion (deviation detection)

    spread_bps: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # inst_id -> float
    # used by:
    # - market making (quote width decision)
    # - arbitrage (cost filter)
    # - all strategies (execution filter)


    # --------------------------------------------------------
    # Orderbook
    # --------------------------------------------------------

    orderbook_by_symbol: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=30))
    # inst_id -> orderbook snapshot
    # used by:
    # - market making (depth / imbalance / queue)
    # - arbitrage (leg depth)
    # - short-term momentum / mean reversion (optional microstructure)


    # --------------------------------------------------------
    # Derivatives-specific metrics
    # --------------------------------------------------------

    funding_snapshot: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # inst_id -> {rate, next_time}
    # used by:
    # - arbitrage (funding carry trades)
    # - market making (inventory skew)
    # - momentum / trend (crowdedness)
    # - mean reversion (extreme funding reversal)

    open_interest: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # inst_id -> {oi, ts}
    # used by:
    # - momentum / trend (confirmation)
    # - arbitrage (risk / crowding)
    # - market making (liquidity proxy)


    # --------------------------------------------------------
    # Long / short ratio
    # --------------------------------------------------------

    long_short_by_ccy: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))
    # ccy -> ratio
    # used by:
    # - mean reversion (primary signal)
    # - momentum / trend (crowding filter)
    # - arbitrage (risk filter)
    # - market making (inventory bias)


    # --------------------------------------------------------
    # Execution state (NON-TTL)
    # --------------------------------------------------------

    positions_by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # used by:
    # - market making (inventory)
    # - arbitrage (leg tracking)
    # - all strategies (position awareness)

    open_orders_by_symbol: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # used by:
    # - market making (quote lifecycle)
    # - arbitrage (partial fills)
    # - limit-order execution

    cooldown_until: Dict[str, float] = field(default_factory=dict)
    # symbol -> epoch seconds
    # used by:
    # - all strategies (anti-overtrading)
    # - market making (cancel/replace throttling)


    # --------------------------------------------------------
    # Health / infra
    # --------------------------------------------------------

    consecutive_failures: Dict[str, int] = field(default_factory=dict)
    # endpoint -> count
    # used by:
    # - all strategies (API instability detection)

    circuit_breaker_until: float = 0.0
    # used by:
    # - all strategies (global kill switch)

    last_update_by_key: Dict[str, float] = field(default_factory=dict)
    # e.g. 'orderbook:BTC-USDT-SWAP' -> epoch seconds
    # used by:
    # - market making (stale data detection)
    # - arbitrage (leg synchronization)
    # - debugging / monitoring


    # --------------------------------------------------------
    # Cache slot helpers
    # --------------------------------------------------------

    def ob_cache(self, inst_id: str, ttl_s: float = 1.5) -> TTLValue:
        """Orderbook cache slot (used mainly by market making / arbitrage)."""
        return self.orderbook_by_symbol.get_or_create(inst_id, ttl_s=ttl_s)

    def lsr_cache(self, ccy: str, ttl_s: float = 30.0) -> TTLValue:
        """Long/short ratio cache slot (mean reversion / crowding)."""
        return self.long_short_by_ccy.get_or_create(ccy, ttl_s=ttl_s)

    def best_bid_ask_cache(self, inst_id: str, ttl_s: float = 1.0) -> TTLValue:
        return self.best_bid_ask.get_or_create(inst_id, ttl_s=ttl_s)

    def mark_index_last_cache(self, inst_id: str, ttl_s: float = 1.0) -> TTLValue:
        return self.mark_index_last.get_or_create(inst_id, ttl_s=ttl_s)

    def spread_bps_cache(self, inst_id: str, ttl_s: float = 1.0) -> TTLValue:
        return self.spread_bps.get_or_create(inst_id, ttl_s=ttl_s)

    def funding_cache(self, inst_id: str, ttl_s: float = 60.0) -> TTLValue:
        return self.funding_snapshot.get_or_create(inst_id, ttl_s=ttl_s)

    def oi_cache(self, inst_id: str, ttl_s: float = 120.0) -> TTLValue:
        return self.open_interest.get_or_create(inst_id, ttl_s=ttl_s)


    # --------------------------------------------------------
    # Maintenance
    # --------------------------------------------------------

    def cleanup(self) -> Dict[str, int]:
        """
        Cleanup very stale entries from bounded caches.

        Suggested:
            call every 30–60 seconds in main loop
        """
        return {
            "orderbook": self.orderbook_by_symbol.cleanup(3.0),
            "bbo": self.best_bid_ask.cleanup(3.0),
            "mark_index_last": self.mark_index_last.cleanup(3.0),
            "spread": self.spread_bps.cleanup(3.0),
            "funding": self.funding_snapshot.cleanup(5.0),
            "oi": self.open_interest.cleanup(5.0),
            "long_short": self.long_short_by_ccy.cleanup(5.0),
        }
