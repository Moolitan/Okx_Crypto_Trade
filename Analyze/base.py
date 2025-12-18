# strategies/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from Trade.base import ExchangeBase
from Accounts.base import AccountBase

class StrategyBase(ABC):
    """
    策略抽象：负责“选币、开仓、监控、平仓”等状态机。
    """
    def __init__(self, exchange: ExchangeBase, account: AccountBase):
        self.exchange = exchange
        self.account = account
        self.running = True

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError
