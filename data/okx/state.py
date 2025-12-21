# data/okx/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import time
from collections import OrderedDict


# ----------------------------
# TTL cache primitive
# ----------------------------
@dataclass
class TTLValue:
    """
    A single cached value with TTL (time-to-live).
    - value: cached payload (dict/df/float/etc.)
    - updated_at: epoch seconds when updated
    - ttl_s: seconds before considered stale
    """
    value: Any = None
    updated_at: float = 0.0
    ttl_s: float = 5.0

    def is_fresh(self) -> bool:
        return (time.time() - self.updated_at) < self.ttl_s

    def set(self, v: Any) -> None:
        self.value = v
        self.updated_at = time.time()


# ----------------------------
# LRU dict (bounded)
# ----------------------------
class LRUTTLDict:
    """
    Bounded mapping: key -> TTLValue, with LRU eviction.
    - Keeps at most max_items keys.
    - Evicts least-recently-used key when over capacity.
    """
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._data: "OrderedDict[str, TTLValue]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def get_or_create(self, key: str, ttl_s: float) -> TTLValue:
        if key in self._data:
            tv = self._data.pop(key)
            tv.ttl_s = ttl_s  # allow dynamic ttl adjustment
            self._data[key] = tv  # move to MRU
            return tv

        # create new
        tv = TTLValue(ttl_s=ttl_s)
        self._data[key] = tv

        # evict if over capacity
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)  # LRU
        return tv

    def cleanup(self, stale_factor: float = 3.0) -> int:
        """
        Remove entries that are very stale: now - updated_at > ttl_s * stale_factor
        Returns number of removed entries.
        """
        now = time.time()
        to_delete = []
        for k, tv in self._data.items():
            if tv.updated_at <= 0:
                # never populated -> safe to keep (or delete if you want)
                continue
            if (now - tv.updated_at) > (tv.ttl_s * stale_factor):
                to_delete.append(k)

        for k in to_delete:
            self._data.pop(k, None)
        return len(to_delete)


# ----------------------------
# Runtime State for strategies
# ----------------------------
@dataclass
class OkxRuntimeState:
    """
    In-memory (non-persisted) runtime state for strategy decisions.

    Designed for:
    - ~10 symbols
    - second-level updates
    - safe memory usage (bounded LRU caches + cleanup)
    """

    # --- Universe / symbol meta ---
    active_symbols: set[str] = field(default_factory=set)
    symbol_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # tickSize, lotSize, ctVal, etc.
    disabled_until: Dict[str, float] = field(default_factory=dict)        # symbol -> epoch seconds

    # --- Rankings (global snapshots) ---
    top_gainers: TTLValue = field(default_factory=lambda: TTLValue(ttl_s=10))
    top_volume: TTLValue = field(default_factory=lambda: TTLValue(ttl_s=10))

    # --- Quotes (best bid/ask, mark/index/last, spread) ---
    # Keep these as cached dicts to reduce repeated computation in strategies.
    best_bid_ask: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))   # inst_id -> {'bid':x,'ask':y,'ts':...}
    mark_index_last: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))# inst_id -> {'mark':..,'index':..,'last':..,'ts':..}
    spread_bps: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))     # inst_id -> float

    # --- Orderbook snapshots (bounded) ---
    # For 10 symbols, set max_items ~ 30 (allows rotations, but prevents growth).
    orderbook_by_symbol: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=30))

    # --- Derivatives snapshots ---
    funding_snapshot: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))  # inst_id -> {'rate':..,'next_time':..}
    open_interest: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))     # inst_id -> {'oi':..,'ts':..}

    # --- Long/Short ratio snapshots (by ccy) ---
    long_short_by_ccy: LRUTTLDict = field(default_factory=lambda: LRUTTLDict(max_items=50))

    # --- Execution state (non-TTL, event/loop driven) ---
    positions_by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    open_orders_by_symbol: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    cooldown_until: Dict[str, float] = field(default_factory=dict)  # symbol -> epoch seconds

    # --- Health / infra ---
    consecutive_failures: Dict[str, int] = field(default_factory=dict)   # endpoint -> count
    circuit_breaker_until: float = 0.0                                  # global breaker
    last_update_by_key: Dict[str, float] = field(default_factory=dict)   # 'orderbook:BTC...' -> epoch seconds

    # ------------- small helpers -------------
    def is_symbol_enabled(self, inst_id: str) -> bool:
        """Return False if symbol is disabled until some future time."""
        until = self.disabled_until.get(inst_id, 0.0)
        return time.time() >= until

    def disable_symbol(self, inst_id: str, seconds: float, reason: str = "") -> None:
        """Disable a symbol for `seconds` (useful for stale data / bad liquidity / errors)."""
        self.disabled_until[inst_id] = time.time() + seconds
        if reason:
            self.last_update_by_key[f"disabled_reason:{inst_id}"] = time.time()

    # ------------- cache slot getters -------------
    def ob_cache(self, inst_id: str, ttl_s: float = 1.5) -> TTLValue:
        """Orderbook cache slot for symbol."""
        return self.orderbook_by_symbol.get_or_create(inst_id, ttl_s=ttl_s)

    def lsr_cache(self, ccy: str, ttl_s: float = 30.0) -> TTLValue:
        """Long/short ratio cache slot for currency."""
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

    # ------------- maintenance -------------
    def cleanup(self) -> Dict[str, int]:
        """
        Cleanup very stale entries from bounded caches.
        Call e.g. every 30-60 seconds in your main loop.
        """
        removed = {
            "orderbook_by_symbol": self.orderbook_by_symbol.cleanup(stale_factor=3.0),
            "best_bid_ask": self.best_bid_ask.cleanup(stale_factor=3.0),
            "mark_index_last": self.mark_index_last.cleanup(stale_factor=3.0),
            "spread_bps": self.spread_bps.cleanup(stale_factor=3.0),
            "funding_snapshot": self.funding_snapshot.cleanup(stale_factor=5.0),
            "open_interest": self.open_interest.cleanup(stale_factor=5.0),
            "long_short_by_ccy": self.long_short_by_ccy.cleanup(stale_factor=5.0),
        }
        return removed
