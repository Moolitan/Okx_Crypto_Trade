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
    

class ExchangeOperator(ABC):
    """
    交易所操作抽象基类 (Market Data / Public Data Layer)
    职责：负责对交易所获取需求信息，获取成交量，获取涨幅榜的币的信息。
    """

    # --- 1. 市场扫描 (Discovery) ---

    @abstractmethod
    def get_top_gainers(self, limit: int = 20) -> List[Dict]:
        """
        获取涨幅榜
        :return: [{'symbol': 'BTC-USDT', 'change_24h': 0.05, 'price': 98000}, ...]
        """
        pass

    @abstractmethod
    def get_top_volume(self, limit: int = 20) -> List[Dict]:
        """
        获取成交额榜（寻找热点币）
        """
        pass

    # --- 2. 深度与流动性 (Liquidity) ---

    @abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        获取盘口信息 (Bids/Asks)
        用来计算冲击成本，或者观察是否有压盘/托盘
        """
        pass

    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        获取最新成交历史
        用来计算主动买入/卖出量 (Taker Buy/Sell Volume)
        """
        pass

    # --- 3. 需求与情绪 (Demand/Sentiment) ---

    @abstractmethod
    def get_funding_rate_history(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        获取资金费率历史
        判断做多/做空的情绪拥挤度
        """
        pass

    @abstractmethod
    def get_open_interest(self, symbol: str) -> float:
        """
        获取合约持仓量 (Open Interest)
        判断资金是流入还是流出
        """
        pass

    @abstractmethod
    def get_long_short_ratio(self, symbol: str) -> float:
        """
        获取多空比 (可选)
        """
        pass

    # --- 4. 基础信息 (Metadata) ---
    
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
        
        这是最常用的接口，通常包含以下所有信息：
        - 最新成交价 (Last Price)
        - 24小时成交量 (Volume 24h)
        - 24小时最高/最低 (High/Low)
        - 24小时涨跌幅 (Change %)
        - 买一/卖一价 (Best Bid/Ask)
        
        :param symbol: 交易对，如 'BTC-USDT-SWAP'
        :return: 字典格式的行情快照
        """
        pass

    @abstractmethod
    def get_kline(self, symbol: str, bar: str = '15m', limit: int = 100) -> List[List]:
        """
        获取【K线数据】 (Candlestick/OHLCV)
        
        这是计算技术指标（MA, RSI, MACD, Bollinger Bands）的基础。
        
        :param symbol: 交易对
        :param bar: 时间粒度 (e.g., '1m', '15m', '1H', '4H', '1D')
        :param limit:以此推算要获取多少根K线
        :return: [[ts, open, high, low, close, vol, volCcy], ...]
        """
        pass


    def get_current_price(self, symbol: str) -> float:
        """
        只获取当前最新价格 (float)
        """
        ticker = self.get_ticker(symbol)
        # 假设返回字典里 'last' 是最新价，具体取决于实现
        return float(ticker.get('last', 0.0))

    def get_current_volume(self, symbol: str) -> float:
        """
        获取24小时成交量
        """
        ticker = self.get_ticker(symbol)
        return float(ticker.get('vol24h', 0.0))