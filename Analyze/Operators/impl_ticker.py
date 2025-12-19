from typing import Dict
from .core import OkxBaseMixin

class TickerImpl(OkxBaseMixin):
    def get_ticker(self, symbol: str) -> Dict:
        """
        获取单个币种的实时行情
        """
        resp = self.mkt_api.get_ticker(instId=symbol)
        data = self._require_ok(resp, f"get_ticker({symbol})").get("data", [])
        
        if not data:
            return {}
            
        t = data[0]
        return {
            "symbol": t.get("instId"),
            "last": float(t.get("last", 0)),
            "high_24h": float(t.get("high24h", 0)),
            "low_24h": float(t.get("low24h", 0)),
            "vol_24h": float(t.get("vol24h", 0)),        # 24h 成交量 (张)
            "vol_ccy_24h": float(t.get("volCcy24h", 0)), # 24h 成交额 (币/USDT)
            "bid": float(t.get("bidPx", 0)),
            "ask": float(t.get("askPx", 0)),
            "ts": int(t.get("ts", 0))
        }