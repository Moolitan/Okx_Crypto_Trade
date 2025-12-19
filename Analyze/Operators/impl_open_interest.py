from .core import OkxBaseMixin

class OpenInterestImpl(OkxBaseMixin):
    def get_open_interest(self, symbol: str) -> float:
        """
        获取当前合约持仓量 (以币为单位，例如 BTC 数量)
        """
        resp = self.mkt_api.get_open_interest(instId=symbol)
        data = self._require_ok(resp, f"get_open_interest({symbol})").get("data", [])
        
        if data:
            # oiCcy: 持仓量（币）
            # oi: 持仓量（张）
            # 这里的选择取决于你的策略习惯，通常用币数(oiCcy)更直观，但部分币种可能只有张数
            return float(data[0].get("oiCcy") or data[0].get("oi") or 0.0)
        return 0.0