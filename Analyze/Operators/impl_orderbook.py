from typing import Dict
from .core import OkxBaseMixin

class OrderbookImpl(OkxBaseMixin):
    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        获取订单簿深度
        """
        # OKX SDK: sz 参数控制返回深度，最大通常支持 400
        resp = self.mkt_api.get_order_books(instId=symbol, sz=str(depth))
        data = self._require_ok(resp, f"get_orderbook({symbol})").get("data", [])
        
        if not data:
            return {"bids": [], "asks": [], "ts": 0}
            
        ob = data[0]
        
        # 格式转换：OKX 返回的是字符串数组 [['price', 'size', ...], ...]
        # 我们转为 [[float_price, float_size], ...]
        def parse_depth(items):
            return [[float(item[0]), float(item[1])] for item in items]
            
        return {
            "bids": parse_depth(ob.get("bids", [])),
            "asks": parse_depth(ob.get("asks", [])),
            "ts": int(ob.get("ts", 0))
        }