from typing import List
from .core import OkxBaseMixin

class KlineImpl(OkxBaseMixin):
    def get_kline(self, symbol: str, bar: str = '15m', limit: int = 100) -> List[List]:
        """
        获取 K 线数据
        :return: [[ts, open, high, low, close, vol, volCcy], ...]
        """
        # OKX 接口参数: bar (e.g. 1m, 15m, 1H, 4H, 1D)
        resp = self.mkt_api.get_candlesticks(instId=symbol, bar=bar, limit=str(limit))
        raw_data = self._require_ok(resp, f"get_kline({symbol})").get("data", [])
        
        cleaned_data = []
        for candle in raw_data:
            # OKX 原始格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            cleaned_data.append([
                int(candle[0]),      # timestamp
                float(candle[1]),    # open
                float(candle[2]),    # high
                float(candle[3]),    # low
                float(candle[4]),    # close
                float(candle[5]),    # volume (张)
                float(candle[6])     # volCcy (币)
            ])
            
        # OKX 返回的数据通常是倒序的（最新的在第一个），习惯上我们按时间正序排列
        return cleaned_data[::-1]