# accounts/okx_account.py
from __future__ import annotations
import os
from typing import Tuple, Optional

from dotenv import load_dotenv
import okx.Account as Account

from .base import AccountBase

load_dotenv(".env")


def require_ok(resp: dict, ctx: str) -> dict:
    if not isinstance(resp, dict):
        raise RuntimeError(f"{ctx}: response not dict: {resp}")
    if resp.get("code") != "0":
        raise RuntimeError(f"{ctx}: {resp.get('msg')} (code={resp.get('code')})")
    return resp


class OkxAccount(AccountBase):
    def __init__(self, *, flag: str = "0", leverage: int = 10):
        # 先初始化 Base 的通用状态字段
        super().__init__(leverage=leverage)

        k = os.getenv("OKX_API_KEY")
        s = os.getenv("OKX_SECRET_KEY")
        p = os.getenv("OKX_PASSPHRASE")
        if not all([k, s, p]):
            raise EnvironmentError("Missing OKX credentials env vars in .env")

        self.flag = flag
        self.acc = Account.AccountAPI(k, s, p, False, flag)

        # 可选：启动时刷新一次余额
        try:
            self.get_usdt_free()
        except Exception:
            pass

    def set_leverage(self, inst_id: str, lever: str, mgn_mode: str, pos_side: str) -> None:
        # 下杠杆/改杠杆
        require_ok(
            self.acc.set_leverage(instId=inst_id, lever=lever, mgnMode=mgn_mode, posSide=pos_side),
            f"set_leverage({inst_id})"
        )
        # 更新本地状态
        try:
            self.leverage = int(float(lever))
        except Exception:
            # 如果 lever 传的是奇怪格式，就保留原值
            pass

    def get_usdt_free(self) -> float:
        # 获取 账户 USDT 可用余额
        resp = require_ok(self.acc.get_account_balance(), "get_account_balance")
        data = resp.get("data", [])
        if not data:
            self.account_balance = 0.0
            return 0.0

        details = data[0].get("details") or data[0].get("balData") or []
        usdt_free = 0.0
        for d in details:
            if d.get("ccy") == "USDT":
                for k in ("availEq", "eqAvl", "cashBal", "eq"):
                    if d.get(k) is not None:
                        try:
                            usdt_free = float(d[k])
                            break
                        except Exception:
                            continue
            if usdt_free:
                break

        self.account_balance = usdt_free
        return usdt_free

    def get_position(self, inst_id: str, pos_side: str) -> Tuple[float, float]:
        # 获取指定币种持仓信息
        resp = require_ok(self.acc.get_positions(instId=inst_id), f"get_positions({inst_id})")
        avail, avgpx = 0.0, 0.0
        for p in resp.get("data", []):
            if p.get("posSide") != pos_side:
                continue
            avail = float(p.get("availPos", "0") or 0)
            avgpx = float(p.get("avgPx", "0") or 0)
            break

        # 更新“当前持仓 instId”状态（单币种策略适用）
        if avail > 0:
            self.position_inst_id = inst_id
        else:
            # 如果这个 inst_id 正好是当前记录的持仓，就清空
            if self.position_inst_id == inst_id:
                self.position_inst_id = ""

        return avail, avgpx
