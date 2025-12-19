from typing import List, Dict
from .core import OkxBaseMixin

class TopVolumeImpl(OkxBaseMixin):
    def get_top_volume(self, limit: int = 20) -> List[Dict]:
        """
        获取 SWAP 永续合约成交额榜 (USDT本位)
        """
        resp = self.mkt_api.get_tickers(instType="SWAP")
        data = self._require_ok(resp, "get_top_volume").get("data", [])
        
        candidates = []
        for t in data:
            inst_id = t.get("instId", "")
            if not inst_id.endswith("USDT-SWAP"):
                continue
            
            try:
                # volCcy24h 通常代表以计价货币(USDT)计算的成交额
                vol_24h_usdt = float(t.get("volCcy24h", 0))
                
                candidates.append({
                    "symbol": inst_id,
                    "volume_24h": vol_24h_usdt,
                    "price": float(t.get("last", 0))
                })
            except (ValueError, TypeError):
                continue
        
        # 排序：成交额从高到低
        candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
        return candidates[:limit]