# exchanges/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ExchangeBase(ABC):
    """
    交易所市场数据 + 交易通道抽象（策略依赖它获取行情/下单等）。
    """

    @abstractmethod
    def get_ticker_last(self, inst_id: str) -> float:
        """获取最新成交价 last"""
        raise NotImplementedError

    @abstractmethod
    def get_candles(self, inst_id: str, bar: str, limit: int) -> List[List[str]]:
        """获取K线原始数据（OKX SDK返回的 data 列表结构）"""
        raise NotImplementedError

    @abstractmethod
    def get_top_gainers(self, inst_type: str, suffix: str, top_n: int) -> List[str]:
        """获取涨幅榜 TopN（返回 instId 列表）"""
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, inst_id: str, td_mode: str, side: str, pos_side: str, sz: str) -> Dict[str, Any]:
        """下市价单"""
        raise NotImplementedError

    @abstractmethod
    def try_place_oco_tpsl(
        self,
        inst_id: str,
        td_mode: str,
        side: str,
        pos_side: str,
        sz: str,
        tp_trigger_px: str,
        sl_trigger_px: str,
    ) -> bool:
        """尝试下OCO止盈止损（不支持则返回 False）"""
        raise NotImplementedError
