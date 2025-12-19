from typing import List, Dict
from .core import OkxBaseMixin

class RecentTradesImpl(OkxBaseMixin):
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        获取最新成交历史
        """
        resp = self.mkt_api.get_trades(instId=symbol, limit=str(limit))
        data = self._require_ok(resp, f"get_recent_trades({symbol})").get("data", [])
        
        trades = []
        for t in data:
            trades.append({
                "price": float(t.get("px")),
                "sz": float(t.get("sz")),
                "side": t.get("side"),  # 'buy' or 'sell'
                "ts": int(t.get("ts"))
            })
            
        return trades