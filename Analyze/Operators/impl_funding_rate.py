from typing import List, Dict
from .core import OkxBaseMixin

class FundingRateImpl(OkxBaseMixin):
    def get_funding_rate_history(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        获取历史资金费率
        """
        resp = self.mkt_api.get_funding_rate_history(instId=symbol, limit=str(limit))
        data = self._require_ok(resp, f"get_funding_rate_history({symbol})").get("data", [])
        
        results = []
        for item in data:
            results.append({
                "symbol": item.get("instId"),
                "funding_rate": float(item.get("fundingRate")),
                "realized_rate": float(item.get("realizedRate", 0)),
                "funding_time": int(item.get("fundingTime"))
            })
        return results