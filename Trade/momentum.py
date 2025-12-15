# strategies/momentum_v1.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from ..Analyze.base import StrategyBase


@dataclass
class RiskPlan:
    tp: float
    sl: float


def ohlcv_from_okx(bars_data: List[List[str]]) -> pd.DataFrame:
    rows = []
    for b in bars_data[::-1]:
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


class MomentumV1(StrategyBase):
    """
    你现在这套：
    - TopN 涨幅榜（USDT 永续）
    - 1m 放量 + 阳线 + 站上 VWAP
    - 单币种持仓
    - TP/SL：优先 OCO，否则程序内止盈止损
    """

    def __init__(
        self,
        exchange,
        account,
        *,
        time_frame: str = "1m",
        candle_limit: int = 50,
        top_n: int = 20,
        leverage: int = 5,
        td_mode: str = "isolated",
        pos_side: str = "long",
        target_rise_pct: float = 0.02,
        hard_sl_pct: float = 0.01,
        vol_multiplier: float = 5.0,
        vol_sma_len: int = 20,
        loop_sleep_sec: int = 60,
        cooldown_sec: int = 300,
        tick_sleep_sec: float = 0.2,
        swap_suffix: str = "-USDT-SWAP",
    ):
        super().__init__(exchange, account)
        self.time_frame = time_frame
        self.candle_limit = candle_limit
        self.top_n = top_n
        self.leverage = leverage
        self.td_mode = td_mode
        self.pos_side = pos_side

        self.target_rise_pct = target_rise_pct
        self.hard_sl_pct = hard_sl_pct
        self.vol_multiplier = vol_multiplier
        self.vol_sma_len = vol_sma_len

        self.loop_sleep_sec = loop_sleep_sec
        self.cooldown_sec = cooldown_sec
        self.tick_sleep_sec = tick_sleep_sec
        self.swap_suffix = swap_suffix

        # 状态
        self.current_inst: Optional[str] = None
        self.risk: Optional[RiskPlan] = None
        self.oco_ok: bool = False
        self.entry_price: float = 0.0

        # 下单数量（先保守固定；后续你可以做成“按余额动态计算”）
        self.fixed_sz = "1"

    def _build_risk(self, entry: float) -> RiskPlan:
        return RiskPlan(
            tp=entry * (1 + self.target_rise_pct),
            sl=entry * (1 - self.hard_sl_pct),
        )

    def _check_buy_signal(self, inst_id: str) -> Tuple[bool, str]:
        bars = self.exchange.get_candles(inst_id, self.time_frame, self.candle_limit)
        df = ohlcv_from_okx(bars)

        if len(df) < (self.vol_sma_len + 1):
            return False, "数据不足"

        current = df.iloc[-1]
        hist = df.iloc[:-1]

        vwap = calculate_vwap(hist)
        vol_sma = calculate_volume_sma(hist, self.vol_sma_len)
        if np.isnan(vwap) or np.isnan(vol_sma) or vol_sma <= 0:
            return False, "指标不可用"

        is_spike = float(current["Volume"]) >= (self.vol_multiplier * vol_sma)
        is_rising = float(current["Close"]) > float(current["Open"])
        is_above = float(current["Close"]) > float(vwap)

        if is_spike and is_rising and is_above:
            return True, f"{float(current['Close'])}"
        return False, f"VSpike={is_spike}, Rising={is_rising}, AboveVWAP={is_above}"

    def _enter(self, inst: str, price: float):
        # 杠杆
        self.account.set_leverage(inst, str(self.leverage), self.td_mode, self.pos_side)

        # 开多市价
        self.exchange.place_market_order(
            inst_id=inst,
            td_mode=self.td_mode,
            side="buy",
            pos_side="long",
            sz=self.fixed_sz,
        )

        self.current_inst = inst
        self.entry_price = price
        self.risk = self._build_risk(price)

        # 尝试 OCO
        self.oco_ok = self.exchange.try_place_oco_tpsl(
            inst_id=inst,
            td_mode=self.td_mode,
            side="sell",
            pos_side="long",
            sz=self.fixed_sz,
            tp_trigger_px=str(self.risk.tp),
            sl_trigger_px=str(self.risk.sl),
        )

        print(f"   风控: TP={self.risk.tp:.6f}, SL={self.risk.sl:.6f}, OCO={'ON' if self.oco_ok else 'OFF(程序内)'}")

    def _close(self, inst: str, sz: str):
        self.exchange.place_market_order(
            inst_id=inst,
            td_mode=self.td_mode,
            side="sell",
            pos_side="long",
            sz=sz,
        )

    def _monitor(self):
        assert self.current_inst is not None

        inst = self.current_inst
        avail, avgpx = self.account.get_position(inst, "long")
        if avail <= 0:
            print(f"   ✓ 已平仓: {inst}")
            self.current_inst = None
            self.risk = None
            self.oco_ok = False
            self.entry_price = 0.0
            print(f"   冷却 {self.cooldown_sec}s")
            time.sleep(self.cooldown_sec)
            return

        if self.oco_ok:
            time.sleep(self.tick_sleep_sec)
            return

        # 程序内 TP/SL
        last = self.exchange.get_ticker_last(inst)
        assert self.risk is not None
        if last >= self.risk.tp:
            print(f"   ✓ 达到止盈，平仓: last={last:.6f} >= TP={self.risk.tp:.6f}")
            self._close(inst, str(avail))
        elif last <= self.risk.sl:
            print(f"   ✓ 触发止损，平仓: last={last:.6f} <= SL={self.risk.sl:.6f}")
            self._close(inst, str(avail))

        time.sleep(self.tick_sleep_sec)

    def _select_and_enter(self):
        universe = self.exchange.get_top_gainers(inst_type="SWAP", suffix=self.swap_suffix, top_n=self.top_n)
        if not universe:
            print("标的池为空，稍后重试")
            time.sleep(self.cooldown_sec)
            return

        for inst in universe:
            ok, info = self._check_buy_signal(inst)
            print(f"[{inst}] signal={ok} {info}")
            if not ok:
                continue

            price = float(info)
            print(f"   >>> 触发开多: {inst} @ {price:.6f}, sz={self.fixed_sz}")
            self._enter(inst, price)
            return

    def run(self) -> None:
        print("-------------------------------------------------------")
        print(" OKX 高频动量策略（模块化版）启动：单币种持仓 / 逐仓杠杆")
        print("-------------------------------------------------------")
        while self.running:
            try:
                if self.current_inst:
                    self._monitor()
                else:
                    self._select_and_enter()
                    time.sleep(self.loop_sleep_sec)
            except KeyboardInterrupt:
                self.running = False
                print("Ctrl-C 退出")
            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(2)
