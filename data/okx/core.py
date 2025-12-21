# data/okx/core.py
import time
from typing import Any, Callable, Dict, Optional, TypeVar

import okx.MarketData as MarketData


T = TypeVar("T")

class OkxBaseMixin:
    """
    OKX API 基类（只管：初始化 + 响应检查 + 统一重试/限速）
    """

    def __init__(self, flag: str = "0", sleep_s: float = 0.05):
        """
        :param flag: "0" live, "1" sim.
        :param sleep_s: 每次请求后小睡一下，降低触发限速概率
        """
        self.flag = flag
        self.sleep_s = sleep_s
        try:
            self.mkt_api = MarketData.MarketAPI(flag=self.flag)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OKX MarketData API: {e}")

    def _require_ok(self, resp: Dict[str, Any], ctx: str) -> Dict[str, Any]:
        if not isinstance(resp, dict):
            raise RuntimeError(f"{ctx}: Response is not a dict: {resp}")

        code = resp.get("code")
        if code != "0":
            msg = resp.get("msg", "Unknown Error")
            raise RuntimeError(f"{ctx} failed: {msg} (code={code})")

        return resp

    def _extract_data(self, resp: Dict[str, Any], ctx: str) -> Any:
        resp = self._require_ok(resp, ctx)
        return resp.get("data", [])

    def _call_with_retry(
        self,
        fn: Callable[..., Dict[str, Any]],
        ctx: str,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """
        统一重试封装：
        - 遇到 OKX 限速/偶发错误时退避重试
        - OKX 限速文档：50011（rate limit reached） :contentReference[oaicite:1]{index=1}
        """
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = fn(**kwargs)
                # 小睡一下，降低触发 rate limit
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return self._extract_data(resp, ctx)
            except Exception as e:
                last_err = e
                # 简单指数退避
                time.sleep(backoff_base * (2 ** attempt))

        raise RuntimeError(f"{ctx} failed after {max_retries} retries. last_err={last_err}")
