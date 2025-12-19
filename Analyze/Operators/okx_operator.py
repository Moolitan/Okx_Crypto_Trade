# Analyze/operators/okx_operator.py

# 1. 导入抽象基类 (契约)
from ..base import ExchangeOperator

# 2. 导入核心连接 (基础)
from .core import OkxBaseMixin

# 3. 导入具体功能实现 (组件)
# 注意：你需要确保 Analyze/operators/ 目录下存在以下文件
# 如果某些文件尚未创建，请先注释掉对应的 import 和类继承
try:
    from .impl_top_gainers import TopGainersImpl
    from .impl_top_volume import TopVolumeImpl
    from .impl_orderbook import OrderbookImpl
    from .impl_recent_trades import RecentTradesImpl
    from .impl_funding_rate import FundingRateImpl
    from .impl_open_interest import OpenInterestImpl
    from .impl_ls_ratio import LongShortRatioImpl
    from .impl_instrument import InstrumentImpl
    from .impl_ticker import TickerImpl
    from .impl_kline import KlineImpl
except ImportError as e:
    print(f"Warning: Some implementation modules are missing: {e}")
    # 为了防止代码完全崩溃，这里定义空类作为占位符（可选）
    # 在实际开发中，建议直接报错提示缺文件
    TopGainersImpl = object
    TopVolumeImpl = object
    OrderbookImpl = object
    RecentTradesImpl = object
    FundingRateImpl = object
    OpenInterestImpl = object
    LongShortRatioImpl = object
    InstrumentImpl = object
    TickerImpl = object
    KlineImpl = object


class OkxOperator(
    # --- A. 市场发现组件 ---
    TopGainersImpl,
    TopVolumeImpl,
    
    # --- B. 深度与流动性组件 ---
    OrderbookImpl,
    RecentTradesImpl,
    
    # --- C. 情绪数据组件 ---
    FundingRateImpl,
    OpenInterestImpl,
    LongShortRatioImpl,
    
    # --- D. 基础数据组件 ---
    InstrumentImpl,
    TickerImpl,
    KlineImpl,
    
    # --- E. 核心连接 ---
    OkxBaseMixin,
    
    # --- F. 抽象基类 (接口约束) ---
    ExchangeOperator
):
    """
    OKX 交易所全功能操作类。
    
    继承顺序说明 (MRO):
    1. 具体的 Impl Mixin 类 (提供具体功能方法，如 get_kline)
    2. OkxBaseMixin (提供 self.mkt_api 和 self._require_ok)
    3. ExchangeOperator (提供类型检查和接口约束)
    """

    def __init__(self, flag: str = "0"):
        """
        :param flag: "0" 实盘, "1" 模拟盘
        """
        # 初始化基础连接 (这会创建 self.mkt_api)
        OkxBaseMixin.__init__(self, flag=flag)
        
        # 初始化基类 (通常为空，但为了多重继承的正确性建议调用)
        ExchangeOperator.__init__(self)

    def __repr__(self):
        return f"<OkxOperator flag={self.flag}>"

    # 如果需要覆盖 Mixin 中的方法，或者添加不属于接口的特殊逻辑，可以在此编写
    # 例如：
    # def get_kline(self, ...):
    #     print("Calling OKX kline...")
    #     return super().get_kline(...)