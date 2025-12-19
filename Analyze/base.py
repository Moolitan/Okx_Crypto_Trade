# Analyze/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# 如果 StrategyBase 和 AccountBase 在其他地方定义，请确保路径正确
# 这里假设它们在项目根目录的 Trade 和 Accounts 包中
# from Trade.base import ExchangeBase
# from Accounts.base import AccountBase

class StrategyBase(ABC):
    """
    策略抽象：负责“选币、开仓、监控、平仓”等状态机。
    """
    def __init__(self, operator: 'ExchangeOperator', account: Any):
        # 注意：这里我们将 exchange 换成了 operator，更符合现在的架构
        self.operator = operator
        self.account = account
        self.running = True

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class ExchangeOperator(ABC):
    """
    交易所操作抽象基类 (Market Data / Public Data Layer)
    职责：负责对交易所获取需求信息，获取成交量，获取涨幅榜等公共数据。
    """

    # --- 1. 市场扫描 (Discovery) ---

    @abstractmethod
    def get_top_gainers(self, limit: int = 20) -> List[Dict]:
        """
        获取涨幅榜
        :return: [{'symbol': 'BTC-USDT-SWAP', 'change_24h': 0.05, 'price': 98000}, ...]
        """
        pass

    @abstractmethod
    def get_top_volume(self, limit: int = 20) -> List[Dict]:
        """
        获取成交额榜（寻找热点币）
        :return: [{'symbol': 'ETH-USDT-SWAP', 'volume_24h': 100000000, 'price': 3000}, ...]
        """
        pass

    # --- 2. 深度与流动性 (Liquidity) ---

    @abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        获取盘口信息 (Bids/Asks)
        :return: {'bids': [[price, size], ...], 'asks': [[price, size], ...], 'ts': 123456789}
        """
        pass

    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        获取最新成交历史
        :return: [{'price': 100, 'sz': 1, 'side': 'buy', 'ts': 123456}, ...]
        """
        pass

    # --- 3. 需求与情绪 (Demand/Sentiment) ---

    @abstractmethod
    def get_funding_rate_history(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        获取资金费率历史
        """
        pass

    @abstractmethod
    def get_open_interest(self, symbol: str) -> float:
        """
        获取合约持仓量 (Open Interest)
        """
        pass

    @abstractmethod
    def get_long_short_ratio(self, symbol: str) -> float:
        """
        获取多空比
        """
        pass

    # --- 4. 基础信息 (Metadata & Market Data) ---
    
    @abstractmethod
    def get_instrument_info(self, symbol: str) -> Dict:
        """
        获取合约面值、最小下单数量、价格精度等
        """
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """
        获取指定币种的【实时聚合行情】(Ticker)
        :return: {'symbol': '...', 'last': 100.0, 'vol_24h': 1000, ...}
        """
        pass

    @abstractmethod
    def get_kline(self, symbol: str, bar: str = '15m', limit: int = 100) -> List[List]:
        """
        获取【K线数据】
        :return: [[ts, open, high, low, close, vol, ...], ...] (按时间正序排列)
        """
        pass
