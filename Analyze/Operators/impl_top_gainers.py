from typing import List, Dict
from .core import OkxBaseMixin

class TopGainersImpl(OkxBaseMixin):
    def get_top_gainers(self, limit: int = 20) -> List[Dict]:
        """
        获取 SWAP 永续合约涨幅榜 (USDT本位)
        """
        # 获取所有 SWAP 的 Ticker
        resp = self.mkt_api.get_tickers(instType="SWAP")
        data = self._require_ok(resp, "get_top_gainers").get("data", [])
        
        candidates = []
        for t in data:
            inst_id = t.get("instId", "")
            # 过滤：只看 USDT 永续
            if not inst_id.endswith("USDT-SWAP"):
                continue
            
            try:
                # 计算涨跌幅
                # 优先寻找 API 直接提供的 24h 涨跌幅字段 (不同版本SDK字段可能不同)
                # 常见字段: sodUtc0 (UTC 0点开盘价), open24h (24h前价格)
                last_price = float(t.get("last", 0))
                open_24h = float(t.get("open24h", 0))
                
                # 如果有 24h 开盘价且大于 0，计算涨跌幅
                if open_24h > 0:
                    change = (last_price - open_24h) / open_24h
                else:
                    change = 0.0
                
                candidates.append({
                    "symbol": inst_id,
                    "change_24h": change,
                    "price": last_price
                })
            except (ValueError, TypeError):
                continue
                
        # 排序：从高到低
        candidates.sort(key=lambda x: x["change_24h"], reverse=True)
        return candidates[:limit]