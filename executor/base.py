# exchanges/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ExchangeBase(ABC):
    """
    交易所市场数据 + 交易通道抽象（策略依赖它获取行情/下单等）。
    """



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
