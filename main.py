# main.py
from exchanges import OkxSdkExchange
from accounts import OkxAccount
from strategies import MomentumV1

def main():
    FLAG = "0"  # 0=实盘, 1=模拟盘（按你SDK）
    # exchange = OkxSdkExchange(flag=FLAG)
    account = OkxAccount(flag=FLAG)
    usdt_free = account.get_usdt_free()
    equity = account.get_account_equity()
    print(f"账户 USDT 总权益: {equity}")
    print(f"账户 USDT 可用余额: {usdt_free}")

    # 换策略只要替换这里：比如 MomentumV2(), MeanReversion(), GridStrategy()...
    # strategy = MomentumV1(
    #     exchange=exchange,
    #     account=account,
    #     td_mode="isolated",
    #     leverage=5,
    #     top_n=20,
    #     target_rise_pct=0.02,
    #     hard_sl_pct=0.01,
    # )

    # strategy.run()

if __name__ == "__main__":
    main()
