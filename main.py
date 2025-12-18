# main.py
from Trade import OkxSdkExchange
from Accounts import OkxAccount
from Trade import MomentumV1
from pprint import pprint
def main():
    FLAG = "0"  # 0=实盘, 1=模拟盘（按你SDK）
    # exchange = OkxSdkExchange(flag=FLAG)
    account = OkxAccount(flag=FLAG)
    usdt_free = account.get_usdt_free()
    equity = account.get_account_equity()
    # leverage = account.get_leverage()
    account.print_positions_summary()
    positions = account.get_all_positions("SWAP", simple=True)
    # print(f"账户杠杆倍数: {leverage}")
    print(f"账户 USDT 总权益: {equity}")
    print(f"账户 USDT 可用余额: {usdt_free}")
    print(f"当前SWAP持仓: \n")
    pprint(positions)

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
