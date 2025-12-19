from typing import Dict
from .core import OkxBaseMixin

class InstrumentImpl(OkxBaseMixin):
    def get_instrument_info(self, symbol: str) -> Dict:
        """
        获取合约详细信息 (面值, 最小下单量, 价格精度等)
        """
        # instType="SWAP" 
        resp = self.mkt_api.get_instruments(instType="SWAP", instId=symbol)
        data = self._require_ok(resp, f"get_instrument_info({symbol})").get("data", [])
        
        if not data:
            return {}
            
        info = data[0]
        return {
            "symbol": info.get("instId"),
            "contract_val": float(info.get("ctVal", 0)),      # 合约面值
            "contract_val_ccy": info.get("ctValCcy"),         # 面值单位 (e.g. USD)
            "tick_sz": float(info.get("tickSz", 0)),          # 价格最小跳动
            "lot_sz": float(info.get("lotSz", 0)),            # 下单数量最小跳动
            "min_sz": float(info.get("minSz", 0)),            # 最小下单数量
            "lever_max": float(info.get("lever", 1))          # 最大杠杆
        }