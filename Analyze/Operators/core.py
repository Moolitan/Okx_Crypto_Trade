# Analyze/operators/core.py
import os
from dotenv import load_dotenv
import okx.MarketData as MarketData

# 加载 .env 文件
load_dotenv(".env")

class OkxBaseMixin:
    """
    OKX 基础连接 Mixin 类。
    职责：
    1. 初始化 MarketData API 连接。
    2. 提供统一的 API 响应检查 (_require_ok)。
    """
    def __init__(self, flag: str = "0"):
        """
        :param flag: "0" for live trading, "1" for simulation.
        """
        self.flag = flag
        
        # 检查必要的环境变量，虽然 MarketData 可能不需要 Key，但为了严谨通常建议检查
        # 如果你的 MarketData 不需要鉴权，可以注释掉这部分检查
        # if not os.getenv("OKX_API_KEY"):
        #     raise EnvironmentError("Missing OKX_API_KEY in .env")

        # 初始化 OKX MarketData API
        # 注意：MarketData 通常是公共数据，不需要私钥，除非访问受限接口
        try:
            self.mkt_api = MarketData.MarketAPI(flag=self.flag)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OKX MarketData API: {e}")

    def _require_ok(self, resp: dict, ctx: str) -> dict:
        """
        统一处理 OKX SDK 返回的错误码。
        """
        if not isinstance(resp, dict):
            raise RuntimeError(f"{ctx}: Response is not a dict: {resp}")
        
        code = resp.get("code")
        if code != "0":
            msg = resp.get("msg", "Unknown Error")
            # 可以在这里增加重试逻辑或特定错误处理
            raise RuntimeError(f"{ctx} Failed: {msg} (code={code})")
            
        return resp