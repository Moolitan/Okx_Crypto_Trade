# data/okx/core.py
import time
import logging
from typing import Any, Callable, Dict, Optional

# Try importing the OKX SDK. If it's not installed, keep running in a degraded mode (useful for local tests/mocks).
try:
    import okx.MarketData as MarketData
except ImportError:
    MarketData = None

logger = logging.getLogger("OkxCore")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OkxBaseMixin:
    """
    OKX API base mixin.
    Responsibilities:
    - Initialize SDK client
    - Validate responses
    - Provide unified retry + simple rate limiting
    """

    def __init__(self, flag: str = "0", sleep_s: float = 0.05, api_client: Any = None):
        """
        :param flag: "0" for live trading, "1" for demo trading
        :param sleep_s: proactive sleep (seconds) after each request to ease rate limiting
        :param api_client: optional injected mock client for testing
        """
        self.flag = flag
        self.sleep_s = sleep_s

        if api_client is not None:
            self.mkt_api = api_client
            return

        if MarketData is None:
            logger.warning("okx.MarketData module not found. API calls will fail unless mocked.")
            self.mkt_api = None
            return

        try:
            # debug=False to silence verbose SDK prints/logs
            self.mkt_api = MarketData.MarketAPI(flag=self.flag, debug=False)
        except Exception as e:
            logger.error(f"Failed to initialize OKX MarketData API: {e}")
            self.mkt_api = None

    def _require_ok(self, resp: Dict[str, Any], ctx: str) -> Dict[str, Any]:
        """Validate OKX standard response schema: {code: '0', data: [...], msg: ''}"""
        if not isinstance(resp, dict):
            raise RuntimeError(f"{ctx}: Invalid response type: {type(resp)}. Content: {resp}")

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
        Unified retry wrapper.
        Handles transient network issues and common OKX rate limiting (e.g., code 50011 / HTTP 429).
        """
        if self.mkt_api is None:
            raise RuntimeError("MarketAPI not initialized.")

        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = fn(**kwargs)

                # Light client-side throttling
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)

                return self._extract_data(resp, ctx)

            except Exception as e:
                last_err = e
                err_str = str(e)

                is_rate_limit = ("50011" in err_str) or ("Too Many Requests" in err_str) or ("429" in err_str)
                wait_time = backoff_base * (2 ** attempt)

                if is_rate_limit:
                    # Add a small extra buffer on rate limit backoff
                    wait_time += 1.0
                    logger.warning(f"Rate limit hit in {ctx}. Retrying in {wait_time:.2f}s...")
                else:
                    logger.debug(
                        f"{ctx} attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time:.2f}s..."
                    )

                time.sleep(wait_time)

        logger.error(f"{ctx} failed permanently after {max_retries} retries. Last error: {last_err}")
        raise RuntimeError(f"{ctx} failed: {last_err}")
