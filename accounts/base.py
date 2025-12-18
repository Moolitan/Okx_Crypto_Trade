# accounts/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class AccountBase(ABC):
    """
    账户抽象：杠杆、余额、持仓
    """
    def __init__(self, leverage: int = 10):
        super().__init__()
        self.account_balance: float = 0.0      # USDT 可用余额
        self.account_equity: float = 0.0    # USDT 总权益（含持仓、浮盈亏）
        self.leverage: int = leverage          # 默认杠杆倍数
        self.position_inst_id: str = ""        # 当前持仓 instId（你策略里“单币种持仓”会很有用）

    @abstractmethod
    def set_leverage(self, inst_id: str, lever: str, mgn_mode: str, pos_side: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_usdt_free(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def get_account_equity(self) -> float:
        """返回 USDT 总权益（含持仓）"""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, inst_id: str, pos_side: str) -> Tuple[float, float]:
        """
        返回 (availPos, avgPx)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_all_positions(self, inst_type: str, simple: bool = False) -> Any:
        """
        返回所有持仓列表（raw data or simple list）
        """
        raise NotImplementedError
    
    @abstractmethod
    def print_positions_summary(self) -> None:
        """
        打印当前持仓汇总信息
        """
        raise NotImplementedError
