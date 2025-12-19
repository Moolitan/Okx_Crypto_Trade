import okx.Rubik as Rubik
from .core import OkxBaseMixin

class LongShortRatioImpl(OkxBaseMixin):
    """
    注意：多空比数据属于 Rubik (Trading Data) 模块，
    因此我们需要单独初始化 RubikAPI，而不仅仅是使用 core.py 里的 mkt_api。
    """
    
    def get_long_short_ratio(self, symbol: str) -> float:
        """
        获取精英交易员多空持仓人数比 (Long/Short Ratio)
        """
        try:
            # 临时初始化 RubikAPI (或者你可以考虑将其放入 core.py 的 __init__)
            # 这里为了保持 mixin 的独立性，且不修改 core.py，选择局部初始化
            # 实际上建议在 core.py 统一管理所有 API 实例
            rubik_api = Rubik.RubikAPI(flag=self.flag)
            
            # period: 周期，可选 "5m", "15m", "1H" 等
            # OKX 接口名通常为 get_long_short_account_ratio
            # 注意：币对参数通常是 "BTC" 而不是 "BTC-USDT-SWAP"
            ccy = symbol.split('-')[0] # 提取币种，如 BTC
            
            resp = rubik_api.get_long_short_account_ratio(ccy=ccy, period="5m")
            
            if resp.get("code") == "0" and resp.get("data"):
                # 返回最新的一个数据点
                latest = resp["data"][0] 
                return float(latest.get("ratio", 1.0))
                
        except Exception as e:
            print(f"Warning: Failed to get L/S ratio for {symbol}: {e}")
            
        return 1.0 # 默认值，表示多空平衡