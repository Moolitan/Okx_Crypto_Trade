# -*- coding: utf-8 -*-
"""
OKX 1m 高频动量策略（SDK版）
- 动态筛选：USDT 永续涨幅榜 TopN
- 信号：放量(>= M*20均量) + 阳线 + 站上VWAP
- 交易：逐仓/杠杆，单币种持仓，市价开多
- 风控：TP/SL 优先尝试 OCO（若SDK支持），否则程序内手动止盈止损
"""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import okx.MarketData as MarketData
import okx.Trade as Trade
import okx.Account as Account

load_dotenv(".env")


# -----------------------------
# 参数区（按你前面那份策略）
# -----------------------------
TIME_FRAME = "1m"
CANDLE_LIMIT = 50
TOP_N_GAINERS = 20

LEVERAGE = 5
RISK_MODE = "isolated"          # OKX: isolated / cross
POS_SIDE = "long"               # 本策略只做多

TARGET_PRICE_RISE_PCT = 0.02    # +2%
HARD_STOP_LOSS_PCT = 0.01       # -1%

VOLUME_MULTIPLIER = 5
VOLUME_SMA_LENGTH = 20

MAIN_LOOP_SLEEP_SEC = 60
COOLDOWN_SEC = 300
TICK_SLEEP_SEC = 0.2            # 监控持仓时的轮询间隔

FLAG = "0"                      # 0=实盘 1=模拟盘（按你示例）


# -----------------------------
# 工具函数：OKX SDK 初始化
# -----------------------------
def okx_init() -> Tuple[MarketData.MarketAPI, Trade.TradeAPI, Account.AccountAPI]:
    k = os.getenv("OKX_API_KEY")
    s = os.getenv("OKX_SECRET_KEY")
    p = os.getenv("OKX_PASSPHRASE")
    if not all([k, s, p]):
        raise EnvironmentError("Missing OKX credentials env vars in .env")

    mkt = MarketData.MarketAPI(flag=FLAG)
    trd = Trade.TradeAPI(k, s, p, False, FLAG)
    acc = Account.AccountAPI(k, s, p, False, FLAG)
    return mkt, trd, acc


def require_ok(resp: dict, ctx: str = "okx call") -> dict:
    if not isinstance(resp, dict):
        raise RuntimeError(f"{ctx}: response not dict: {resp}")
    if resp.get("code") != "0":
        raise RuntimeError(f"{ctx}: {resp.get('msg')} (code={resp.get('code')})")
    return resp


# -----------------------------
# 指标计算
# -----------------------------
def ohlcv_from_okx(bars_data: List[List[str]]) -> pd.DataFrame:
    """
    OKX get_candlesticks 返回 data: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm] (字符串)
    我们只取前 6 个并转 float。
    """
    rows = []
    for b in bars_data[::-1]:  # OKX 多数是倒序，翻转成时间正序
        rows.append({
            "Timestamp": int(b[0]),
            "Open": float(b[1]),
            "High": float(b[2]),
            "Low": float(b[3]),
            "Close": float(b[4]),
            "Volume": float(b[5]),
        })
    return pd.DataFrame(rows)


def calculate_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vwap_series = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return float(vwap_series.iloc[-1])


def calculate_volume_sma(df: pd.DataFrame, length: int) -> float:
    if df.empty or len(df) < length:
        return np.nan
    return float(df["Volume"].rolling(window=length).mean().iloc[-1])


# -----------------------------
# 市场数据：涨幅榜 TopN（USDT-SWAP）
# -----------------------------
def calc_24h_pct_from_ticker(t: dict) -> Optional[float]:
    """
    OKX ticker 字段在不同接口/版本略有差异，这里做多种兼容：
    优先用 open24h / last 计算，否则尝试 chg24h / sodUtc0 等字段（若存在）。
    返回百分比数值，例如 12.3 表示 +12.3%
    """
    try:
        last = float(t.get("last")) if t.get("last") is not None else None
        open24h = float(t.get("open24h")) if t.get("open24h") is not None else None
        if last is not None and open24h is not None and open24h > 0:
            return (last - open24h) / open24h * 100.0
    except Exception:
        pass

    # 兜底：如果接口直接给了 24h 涨跌幅（有些字段叫 chg24h 或 percentage）
    for k in ("chg24h", "percentage", "changePercentage"):
        if t.get(k) is None:
            continue
        try:
            # 有的返回是小数（0.12），有的返回百分比（12）
            v = float(t[k])
            return v * 100.0 if abs(v) <= 2 else v
        except Exception:
            continue

    return None


def get_top_gainers_universe(mkt: MarketData.MarketAPI, top_n: int) -> List[str]:
    """
    用 OKX SDK 拉 SWAP tickers，过滤 USDT 永续，按 24h 涨幅排序取 TopN
    """
    # OKX: mkt.get_tickers(instType="SWAP")
    resp = require_ok(mkt.get_tickers(instType="SWAP"), "get_tickers")
    data = resp.get("data", [])

    candidates = []
    for t in data:
        inst = t.get("instId")
        if not inst or not inst.endswith("-USDT-SWAP"):
            continue
        pct = calc_24h_pct_from_ticker(t)
        if pct is None:
            continue
        candidates.append((inst, pct))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [inst for inst, _ in candidates[:top_n]]


# -----------------------------
# 信号检测
# -----------------------------
def check_buy_signal(mkt: MarketData.MarketAPI, inst_id: str) -> Tuple[bool, str]:
    try:
        bars = require_ok(
            mkt.get_candlesticks(instId=inst_id, bar=TIME_FRAME, limit=str(CANDLE_LIMIT)),
            f"get_candlesticks({inst_id})"
        )
        df = ohlcv_from_okx(bars["data"])

        if len(df) < (VOLUME_SMA_LENGTH + 1):
            return False, "数据不足"

        current = df.iloc[-1]
        hist = df.iloc[:-1]

        latest_vwap = calculate_vwap(hist)
        volume_sma = calculate_volume_sma(hist, VOLUME_SMA_LENGTH)

        if np.isnan(latest_vwap) or np.isnan(volume_sma) or volume_sma <= 0:
            return False, "指标不可用"

        is_volume_spike = float(current["Volume"]) >= (VOLUME_MULTIPLIER * volume_sma)
        is_rising = float(current["Close"]) > float(current["Open"])
        is_above_vwap = float(current["Close"]) > float(latest_vwap)

        if is_volume_spike and is_rising and is_above_vwap:
            return True, f"{float(current['Close'])}"
        else:
            reason = (
                f"VSpike={is_volume_spike} "
                f"({float(current['Volume']):.4f}/{VOLUME_MULTIPLIER*volume_sma:.4f}), "
                f"Rising={is_rising}, AboveVWAP={is_above_vwap}"
            )
            return False, reason

    except Exception as e:
        return False, f"检查信号异常: {e}"


# -----------------------------
# 账户/持仓
# -----------------------------
def ensure_leverage(acc: Account.AccountAPI, inst_id: str):
    # OKX set_leverage: instId, lever, mgnMode, posSide
    # 本策略只做 long，但你也可以把 short 一起设置
    r = acc.set_leverage(instId=inst_id, lever=str(LEVERAGE), mgnMode=RISK_MODE, posSide=POS_SIDE)
    require_ok(r, f"set_leverage({inst_id})")


def get_usdt_free(acc: Account.AccountAPI) -> float:
    """
    用 get_account_balance 拿 USDT 可用（字段结构可能不同，这里按常见 data/ details/ eqAvl 做兼容）
    """
    resp = require_ok(acc.get_account_balance(), "get_account_balance")
    data = resp.get("data", [])
    if not data:
        return 0.0

    # OKX 常见结构：data[0]["details"] 里按币种
    details = data[0].get("details") or data[0].get("balData") or []
    for d in details:
        if d.get("ccy") == "USDT":
            # 可能字段：availEq / cashBal / eq / eqAvl
            for k in ("availEq", "eqAvl", "cashBal", "eq"):
                if d.get(k) is not None:
                    try:
                        return float(d[k])
                    except Exception:
                        continue
    return 0.0


def get_position(acc: Account.AccountAPI, inst_id: str, pos_side: str) -> Tuple[float, float]:
    """
    返回 (availPos, avgPx)；没有持仓则 (0, 0)
    """
    resp = require_ok(acc.get_positions(instId=inst_id), f"get_positions({inst_id})")
    for p in resp.get("data", []):
        if p.get("posSide") != pos_side:
            continue
        avail = float(p.get("availPos", "0") or 0)
        avgpx = float(p.get("avgPx", "0") or 0)
        return avail, avgpx
    return 0.0, 0.0


# -----------------------------
# 下单/平仓/TP-SL
# -----------------------------
@dataclass
class RiskPlan:
    tp: float
    sl: float


def build_risk_plan(entry_price: float) -> RiskPlan:
    tp = entry_price * (1 + TARGET_PRICE_RISE_PCT)
    sl = entry_price * (1 - HARD_STOP_LOSS_PCT)
    return RiskPlan(tp=tp, sl=sl)


def place_market_long(trd: Trade.TradeAPI, inst_id: str, sz: float):
    r = trd.place_order(
        instId=inst_id,
        tdMode=RISK_MODE,
        side="buy",
        posSide="long",
        ordType="market",
        sz=str(sz),
    )
    require_ok(r, f"place_order_market_buy({inst_id})")
    return r


def close_market_long(trd: Trade.TradeAPI, inst_id: str, sz: float):
    r = trd.place_order(
        instId=inst_id,
        tdMode=RISK_MODE,
        side="sell",
        posSide="long",
        ordType="market",
        sz=str(sz),
    )
    require_ok(r, f"place_order_market_sell({inst_id})")
    return r


def try_place_oco_tpsl(trd: Trade.TradeAPI, inst_id: str, sz: float, plan: RiskPlan) -> bool:
    """
    尝试用 OKX Algo OCO 托管 TP/SL。
    不同 SDK 版本函数名/参数可能不同，所以这里：
    - 如果对象有 place_algo_order，就按常见字段提交
    - 否则返回 False，让主循环用“程序内止盈止损”兜底
    """
    if not hasattr(trd, "place_algo_order"):
        return False

    try:
        # 常见 OKX OCO 字段：ordType="oco", tpTriggerPx, tpOrdPx, slTriggerPx, slOrdPx
        # tpOrdPx/slOrdPx = "-1" 通常表示触发后走市价
        r = trd.place_algo_order(
            instId=inst_id,
            tdMode=RISK_MODE,
            side="sell",
            posSide="long",
            ordType="oco",
            sz=str(sz),
            tpTriggerPx=str(plan.tp),
            tpOrdPx="-1",
            slTriggerPx=str(plan.sl),
            slOrdPx="-1",
            reduceOnly="true",
        )
        require_ok(r, f"place_algo_order_oco({inst_id})")
        return True
    except Exception as e:
        print(f"   ! OCO 提交失败，将降级为程序内止盈止损: {e}")
        return False


# -----------------------------
# 主策略类
# -----------------------------
class MomentumBot:
    def __init__(self):
        self.mkt, self.trd, self.acc = okx_init()
        self.current_inst: Optional[str] = None
        self.entry_price: float = 0.0
        self.risk: Optional[RiskPlan] = None
        self.oco_ok: bool = False
        self.running = True

    def _select_and_enter(self):
        universe = get_top_gainers_universe(self.mkt, TOP_N_GAINERS)
        if not universe:
            print("标的池为空，稍后重试")
            time.sleep(COOLDOWN_SEC)
            return

        for inst in universe:
            ok, info = check_buy_signal(self.mkt, inst)
            print(f"[{inst}] signal={ok} {info}")
            if not ok:
                continue

            # 信号触发：开仓
            price = float(info)
            ensure_leverage(self.acc, inst)

            # 这里的 sz（下单数量）你要自己定：
            # - 你对冲网格代码用 MIN_UNIT=0.1 这种方式，说明你习惯用“固定张数/单位”
            # - 下面默认：用 10 USDT * LEVERAGE 的名义近似换算（但缺少合约面值信息时不严谨）
            # 为了不误下单，这里更安全：先用一个很小的固定 size
            sz = 1  # ✅建议你先用 1 张/1单位小额跑通，再做动态仓位

            print(f"   >>> 触发开多: {inst} @ {price:.6f}, sz={sz}")
            place_market_long(self.trd, inst, sz)

            self.current_inst = inst
            self.entry_price = price
            self.risk = build_risk_plan(price)

            # 尝试挂 OCO
            self.oco_ok = try_place_oco_tpsl(self.trd, inst, sz, self.risk)

            print(f"   风控: TP={self.risk.tp:.6f}, SL={self.risk.sl:.6f}, OCO={'ON' if self.oco_ok else 'OFF(程序内)'}")
            return

    def _monitor_position(self):
        if not self.current_inst:
            return

        inst = self.current_inst

        # 查持仓
        avail, avgpx = get_position(self.acc, inst, "long")
        if avail <= 0:
            print(f"   ✓ 持仓已消失/已平仓: {inst}")
            self.current_inst = None
            self.entry_price = 0.0
            self.risk = None
            self.oco_ok = False
            print(f"   冷却 {COOLDOWN_SEC}s")
            time.sleep(COOLDOWN_SEC)
            return

        # 如果 OCO 已托管，就只需要等持仓消失
        if self.oco_ok:
            time.sleep(TICK_SLEEP_SEC)
            return

        # 否则程序内手动止盈止损
        # 拉最新价
        try:
            tick = require_ok(self.mkt.get_ticker(instId=inst), f"get_ticker({inst})")
            last = float(tick["data"][0]["last"])
        except Exception as e:
            print(f"ticker err: {e}")
            time.sleep(TICK_SLEEP_SEC)
            return

        assert self.risk is not None
        if last >= self.risk.tp:
            print(f"   ✓ 达到止盈，市价平仓: last={last:.6f} >= TP={self.risk.tp:.6f}")
            close_market_long(self.trd, inst, avail)
        elif last <= self.risk.sl:
            print(f"   ✓ 触发止损，市价平仓: last={last:.6f} <= SL={self.risk.sl:.6f}")
            close_market_long(self.trd, inst, avail)

        time.sleep(TICK_SLEEP_SEC)

    def run(self):
        print("-------------------------------------------------------")
        print(" OKX 高频动量策略（OKX SDK版）启动：单币种持仓 / 逐仓5X")
        print("-------------------------------------------------------")

        while self.running:
            try:
                if self.current_inst:
                    self._monitor_position()
                else:
                    self._select_and_enter()
                    time.sleep(MAIN_LOOP_SLEEP_SEC)

            except KeyboardInterrupt:
                self.running = False
                print("Ctrl-C 退出")
            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(2)


if __name__ == "__main__":
    bot = MomentumBot()
    bot.run()
