import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union


class QuantitativeIndicator:
    """
    Indicator engine for strategy use.

    Design:
    - Does NOT inherit MarketDataBus (keeps data layer and compute layer separated)
    - Holds an optional reference to a data provider (e.g., MarketDataBus) and pulls OHLCV on demand
    - Uses explicit attributes (self.ma_fast, self.rsi_n, ...) instead of a params dict
    - Each indicator method returns a Series/DataFrame (and can optionally write back to df)
    """

    def __init__(
        self,
        bus: Optional[Any] = None,
        *,
        ma_fast: int = 7,
        ma_slow: int = 25,
        ma_trend: int = 99,
        boll_n: int = 20,
        boll_k: float = 2.0,
        rsi_n: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        kdj_n: int = 9,
        kdj_m1: int = 3,
        kdj_m2: int = 3,
        wr_n: int = 14,
        roc_n: int = 12,
        cci_n: int = 14,
        supertrend_n: int = 10,
        supertrend_f: float = 3.0,
        atr_n: int = 14,
        sr_n: int = 30,
        vr_n: int = 26,
        psy_n: int = 12,
        mtm_n: int = 12,
        stoch_rsi_n: int = 14,
    ):
        self.bus = bus

        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_trend = ma_trend

        self.boll_n = boll_n
        self.boll_k = boll_k

        self.rsi_n = rsi_n

        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.kdj_n = kdj_n
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2

        self.wr_n = wr_n
        self.roc_n = roc_n
        self.cci_n = cci_n

        self.supertrend_n = supertrend_n
        self.supertrend_f = supertrend_f

        self.atr_n = atr_n
        self.sr_n = sr_n
        self.vr_n = vr_n
        self.psy_n = psy_n
        self.mtm_n = mtm_n
        self.stoch_rsi_n = stoch_rsi_n

    # -------------------------
    # Data access helpers
    # -------------------------

    def _get_df(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Resolve OHLCV DataFrame:
        - If df is provided, use it
        - Otherwise pull from self.bus.get_klines_df()
        """
        if df is not None:
            return df
        if self.bus is None:
            raise ValueError("No DataFrame provided and self.bus is None.")
        if not hasattr(self.bus, "get_klines_df"):
            raise TypeError("bus must provide get_klines_df().")
        return self.bus.get_klines_df()

    @staticmethod
    def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # -------------------------
    # Trend indicators
    # -------------------------

    def ma(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        write_back: bool = True,
    ) -> pd.DataFrame:
        """Simple moving averages (fast/slow)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        f = fast or self.ma_fast
        s = slow or self.ma_slow

        out = pd.DataFrame(index=df.index)
        out[f"MA_{f}"] = df["close"].rolling(f).mean()
        out[f"MA_{s}"] = df["close"].rolling(s).mean()

        if write_back:
            df[out.columns] = out
            return df
        return out

    def ema(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        write_back: bool = True,
    ) -> pd.DataFrame:
        """Exponential moving averages (fast/slow)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        f = fast or self.ma_fast
        s = slow or self.ma_slow

        out = pd.DataFrame(index=df.index)
        out[f"EMA_{f}"] = df["close"].ewm(span=f, adjust=False).mean()
        out[f"EMA_{s}"] = df["close"].ewm(span=s, adjust=False).mean()

        if write_back:
            df[out.columns] = out
            return df
        return out

    def boll(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        k: Optional[float] = None,
        write_back: bool = True,
    ) -> pd.DataFrame:
        """Bollinger Bands (mid/up/low)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.boll_n
        k = k or self.boll_k

        mid = df["close"].rolling(n).mean()
        std = df["close"].rolling(n).std()

        out = pd.DataFrame(index=df.index)
        out["BOLL_mid"] = mid
        out["BOLL_up"] = mid + k * std
        out["BOLL_low"] = mid - k * std

        if write_back:
            df[out.columns] = out
            return df
        return out

    def atr(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "ATR",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Average True Range (ATR)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.atr_n

        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / n, adjust=False).mean()
        atr.name = col_name

        if write_back:
            df[col_name] = atr
            return df
        return atr

    def supertrend_bands(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        factor: Optional[float] = None,
        write_back: bool = True,
        upper_name: str = "SuperTrend_Upper",
        lower_name: str = "SuperTrend_Lower",
    ) -> pd.DataFrame:
        """
        SuperTrend bands only (upper/lower).
        Note: trend direction / flip logic is intentionally not implemented.
        """
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.supertrend_n
        factor = factor or self.supertrend_f

        atr = self.atr(df, n=n, write_back=False)  # Series
        hl2 = (df["high"] + df["low"]) / 2

        out = pd.DataFrame(index=df.index)
        out[upper_name] = hl2 + factor * atr
        out[lower_name] = hl2 - factor * atr

        if write_back:
            df[out.columns] = out
            return df
        return out

    def macd(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
        write_back: bool = True,
    ) -> pd.DataFrame:
        """MACD (dif, dea, hist)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        fast = fast or self.macd_fast
        slow = slow or self.macd_slow
        signal = signal or self.macd_signal

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = 2 * (dif - dea)

        out = pd.DataFrame(index=df.index)
        out["MACD_dif"] = dif
        out["MACD_dea"] = dea
        out["MACD_hist"] = hist

        if write_back:
            df[out.columns] = out
            return df
        return out

    # -------------------------
    # Oscillators
    # -------------------------

    def rsi(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "RSI",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Relative Strength Index (RSI)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.rsi_n

        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)

        ema_up = up.ewm(com=n - 1, adjust=False).mean()
        ema_down = down.ewm(com=n - 1, adjust=False).mean()

        rs = ema_up / ema_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi.name = col_name

        if write_back:
            df[col_name] = rsi
            return df
        return rsi

    def stoch_rsi(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        rsi_n: Optional[int] = None,
        stoch_n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "StochRSI",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Stochastic RSI."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        rsi_n = rsi_n or self.rsi_n
        stoch_n = stoch_n or self.stoch_rsi_n

        rsi = self.rsi(df, n=rsi_n, write_back=False, col_name="_tmp_RSI")
        min_rsi = rsi.rolling(stoch_n).min()
        max_rsi = rsi.rolling(stoch_n).max()
        denom = (max_rsi - min_rsi).replace(0, np.nan)

        st = (rsi - min_rsi) / denom
        st.name = col_name

        if write_back:
            df[col_name] = st
            return df
        return st

    def kdj(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        m1: Optional[int] = None,
        m2: Optional[int] = None,
        write_back: bool = True,
        k_name: str = "K",
        d_name: str = "D",
        j_name: str = "J",
    ) -> pd.DataFrame:
        """KDJ indicator (K, D, J)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.kdj_n
        m1 = m1 or self.kdj_m1
        m2 = m2 or self.kdj_m2

        low_min = df["low"].rolling(n).min()
        high_max = df["high"].rolling(n).max()
        denom = (high_max - low_min).replace(0, np.nan)
        rsv = (df["close"] - low_min) / denom * 100

        k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
        d = k.ewm(alpha=1 / m2, adjust=False).mean()
        j = 3 * k - 2 * d

        out = pd.DataFrame(index=df.index)
        out[k_name] = k
        out[d_name] = d
        out[j_name] = j

        if write_back:
            df[out.columns] = out
            return df
        return out

    def wr(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "WR",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Williams %R."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.wr_n

        high_max = df["high"].rolling(n).max()
        low_min = df["low"].rolling(n).min()
        denom = (high_max - low_min).replace(0, np.nan)

        wr = -100 * (high_max - df["close"]) / denom
        wr.name = col_name

        if write_back:
            df[col_name] = wr
            return df
        return wr

    def roc(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "ROC",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Rate of Change (ROC)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.roc_n

        roc = df["close"].diff(n) / df["close"].shift(n) * 100
        roc.name = col_name

        if write_back:
            df[col_name] = roc
            return df
        return roc

    def mtm(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "MTM",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Momentum (MTM)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.mtm_n

        mtm = df["close"] - df["close"].shift(n)
        mtm.name = col_name

        if write_back:
            df[col_name] = mtm
            return df
        return mtm

    def cci(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "CCI",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Commodity Channel Index (CCI)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.cci_n

        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(n).mean()
        md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
        cci = (tp - ma) / (0.015 * md.replace(0, np.nan))
        cci.name = col_name

        if write_back:
            df[col_name] = cci
            return df
        return cci

    # -------------------------
    # Volume indicators
    # -------------------------

    def obv(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        write_back: bool = True,
        col_name: str = "OBV",
    ) -> Union[pd.Series, pd.DataFrame]:
        """On-Balance Volume (OBV)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()

        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        obv.name = col_name

        if write_back:
            df[col_name] = obv
            return df
        return obv

    def vr(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "VR",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Volume Ratio (VR)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.vr_n

        change = df["close"].diff()
        vol = df["volume"]

        av = np.where(change > 0, vol, 0.0)
        bv = np.where(change < 0, vol, 0.0)
        cv = np.where(change == 0, vol, 0.0)

        av_s = pd.Series(av, index=df.index).rolling(n).sum()
        bv_s = pd.Series(bv, index=df.index).rolling(n).sum()
        cv_s = pd.Series(cv, index=df.index).rolling(n).sum()

        denom = (bv_s + cv_s / 2).replace(0, np.nan)
        vr = (av_s + cv_s / 2) / denom * 100
        vr.name = col_name

        if write_back:
            df[col_name] = vr
            return df
        return vr

    # -------------------------
    # Misc
    # -------------------------

    def psy(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        col_name: str = "PSY",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Psychological Line (PSY)."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.psy_n

        is_up = (df["close"].diff() > 0).astype(int)
        psy = is_up.rolling(n).sum() / n * 100
        psy.name = col_name

        if write_back:
            df[col_name] = psy
            return df
        return psy

    def support_resistance(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        n: Optional[int] = None,
        write_back: bool = True,
        res_name: str = "Res_Line",
        sup_name: str = "Sup_Line",
    ) -> pd.DataFrame:
        """Rolling resistance/support lines."""
        df = self._ensure_ohlcv(self._get_df(df)).copy()
        n = n or self.sr_n

        out = pd.DataFrame(index=df.index)
        out[res_name] = df["high"].rolling(n).max()
        out[sup_name] = df["low"].rolling(n).min()

        if write_back:
            df[out.columns] = out
            return df
        return out

    def sentiment_features(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        sentiment: Optional[Dict[str, Any]] = None,
        write_back: bool = True,
    ) -> pd.DataFrame:
        """
        Add sentiment features by broadcasting latest sentiment values to the full df.
        If sentiment is not provided, it will be taken from self.bus.sentiment_data when available.
        """
        df = self._ensure_ohlcv(self._get_df(df)).copy()

        if sentiment is None:
            if self.bus is None or not hasattr(self.bus, "sentiment_data"):
                sentiment = {}
            else:
                sentiment = self.bus.sentiment_data or {}

        out = pd.DataFrame(index=df.index)
        if sentiment.get("long_short_ratio") is not None:
            out["Global_L_S"] = sentiment["long_short_ratio"]
        if sentiment.get("top_long_short_ratio") is not None:
            out["Top_Acc_L_S"] = sentiment["top_long_short_ratio"]
        if sentiment.get("top_pos_long_short_ratio") is not None:
            out["Top_Pos_L_S"] = sentiment["top_pos_long_short_ratio"]
        if sentiment.get("oi_token") is not None:
            out["OI"] = sentiment["oi_token"]

        if write_back:
            df[out.columns] = out
            return df
        return out
