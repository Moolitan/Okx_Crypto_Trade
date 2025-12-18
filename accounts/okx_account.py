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
            self.get_account_equity()
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
    
    def get_account_equity(self) -> float:
        # 获取 USDT 总权益（含已开仓、浮盈亏）
        resp = require_ok(self.acc.get_account_balance(), "get_account_balance")
        data = resp.get("data", [])
        if not data:
            self.account_equity = 0.0
            return 0.0

        details = data[0].get("details") or data[0].get("balData") or []
        equity = 0.0
        for d in details:
            if d.get("ccy") == "USDT":
                if d.get("eq") is not None:
                    try:
                        equity = float(d["eq"])
                    except Exception:
                        equity = 0.0
                break

        self.account_equity = equity
        return equity


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
    
    def get_all_positions(self, inst_type: str = None, simple: bool = False) -> list[dict]:
            """
            获取当前账户的所有持仓信息。
            
            :param inst_type: 可选筛选，例如 'SWAP'(永续合约), 'FUTURES'(交割合约)
            :param simple: 如果为 True，只返回包含核心字段的精简列表；如果为 False，返回原始详细数据
            :return: 持仓信息列表
            """
            kwargs = {}
            if inst_type:
                kwargs['instType'] = inst_type

            # 调用 API
            resp = require_ok(self.acc.get_positions(**kwargs), "get_all_positions")
            raw_data = resp.get("data", [])

            # 如果开启了 simple 模式，进行数据清洗
            if simple:
                clean_list = []
                for p in raw_data:
                    clean_list.append({
                        "instId": p.get("instId"),      # 币种名
                        "posSide": p.get("posSide"),    # 方向: long/short/net
                        "pos": p.get("pos"),            # 持仓数量
                        "avgPx": p.get("avgPx"),        # 开仓均价
                        "upl": p.get("upl"),            # 未实现盈亏
                        "uplRatio": p.get("uplRatio")   # 收益率
                    })
                return clean_list
            return raw_data
    
    def print_positions_summary(self):
        """
        (辅助方法) 打印当前所有持仓的简报，用于调试
        """
        positions = self.get_all_positions()
        if not positions:
            print(">>> 当前无持仓")
            return

        print(f">>> 当前持仓列表 (共 {len(positions)} 个):")
        print(f"{'合约(InstId)':<20} {'方向':<6} {'持仓量(张/币)':<12} {'均价':<10} {'未实现盈亏(UPL)':<15}")
        print("-" * 70)
        
        for p in positions:
            inst_id = p.get('instId')
            pos_side = p.get('posSide') # long, short, net
            pos_sz = p.get('pos')       # 持仓数量
            avg_px = p.get('avgPx')
            upl = p.get('upl')          # 未实现盈亏
            
            print(f"{inst_id:<20} {pos_side:<6} {pos_sz:<12} {avg_px:<10} {upl:<15}")
        print("-" * 70)

