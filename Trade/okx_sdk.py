# exchanges/okx_sdk.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
import okx.MarketData as MarketData
import okx.Trade as Trade

from .base import ExchangeBase

load_dotenv(".env")


def require_ok(resp: dict, ctx: str) -> dict:
    if not isinstance(resp, dict):
        raise RuntimeError(f"{ctx}: response not dict: {resp}")
    if resp.get("code") != "0":
        raise RuntimeError(f"{ctx}: {resp.get('msg')} (code={resp.get('code')})")
    return resp


def calc_24h_pct_from_ticker(t: dict) -> Optional[float]:
    """
    返回百分比值，例如 12.3 表示 +12.3%
    """
    try:
        last = float(t.get("last")) if t.get("last") is not None else None
        open24h = float(t.get("open24h")) if t.get("open24h") is not None else None
        if last is not None and open24h is not None and open24h > 0:
            return (last - open24h) / open24h * 100.0
    except Exception:
        pass

    for k in ("chg24h", "percentage", "changePercentage"):
        if t.get(k) is None:
            continue
        try:
            v = float(t[k])
            return v * 100.0 if abs(v) <= 2 else v
        except Exception:
            continue
    return None


class OkxSdkExchange(ExchangeBase):
    """
    OKX SDK 实现：行情 + 下单
    """

    def __init__(self, flag: str = "0"):
        k = os.getenv("OKX_API_KEY")
        s = os.getenv("OKX_SECRET_KEY")
        p = os.getenv("OKX_PASSPHRASE")
        if not all([k, s, p]):
            raise EnvironmentError("Missing OKX credentials env vars in .env")

        self.flag = flag
        self.mkt = MarketData.MarketAPI(flag=flag)
        self.trd = Trade.TradeAPI(k, s, p, False, flag)

    def get_ticker_last(self, inst_id: str) -> float:
        resp = require_ok(self.mkt.get_ticker(instId=inst_id), f"get_ticker({inst_id})")
        return float(resp["data"][0]["last"])

    def get_candles(self, inst_id: str, bar: str, limit: int) -> List[List[str]]:
        resp = require_ok(
            self.mkt.get_candlesticks(instId=inst_id, bar=bar, limit=str(limit)),
            f"get_candlesticks({inst_id},{bar},{limit})"
        )
        return resp.get("data", [])

    def get_top_gainers(self, inst_type: str, suffix: str, top_n: int) -> List[str]:
        """
        inst_type: "SWAP"
        suffix: "-USDT-SWAP"
        """
        resp = require_ok(self.mkt.get_tickers(instType=inst_type), f"get_tickers({inst_type})")
        data = resp.get("data", [])

        candidates = []
        for t in data:
            inst = t.get("instId")
            if not inst or not inst.endswith(suffix):
                continue
            pct = calc_24h_pct_from_ticker(t)
            if pct is None:
                continue
            candidates.append((inst, pct))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [inst for inst, _ in candidates[:top_n]]

    def place_market_order(self, inst_id: str, td_mode: str, side: str, pos_side: str, sz: str) -> Dict[str, Any]:
        resp = require_ok(
            self.trd.place_order(
                instId=inst_id,
                tdMode=td_mode,
                side=side,
                posSide=pos_side,
                ordType="market",
                sz=sz,
            ),
            f"place_order({inst_id},{side},{pos_side},{sz})"
        )
        return resp

    def try_place_oco_tpsl(
        self,
        inst_id: str,
        td_mode: str,
        side: str,
        pos_side: str,
        sz: str,
        tp_trigger_px: str,
        sl_trigger_px: str,
    ) -> bool:
        """
        兼容：有的 SDK 叫 place_algo_order，有的没有。
        """
        if not hasattr(self.trd, "place_algo_order"):
            return False

        try:
            resp = require_ok(
                self.trd.place_algo_order(
                    instId=inst_id,
                    tdMode=td_mode,
                    side=side,
                    posSide=pos_side,
                    ordType="oco",
                    sz=sz,
                    tpTriggerPx=tp_trigger_px,
                    tpOrdPx="-1",
                    slTriggerPx=sl_trigger_px,
                    slOrdPx="-1",
                    reduceOnly="true",
                ),
                f"place_algo_order_oco({inst_id})"
            )
            return True
        except Exception:
            return False
