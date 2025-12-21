import duckdb
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm
import sys

# ------------------------------------------------------------
# 说明：
# 这份代码实现的是一个本地金融数据仓库（Finstore）。
# ✅ 只负责：存储 / 读取 / 合并 / 批量计算技术指标 / 流式落盘
# ❌ 不负责：调用任何交易所 API（REST/WebSocket连接、鉴权、请求等都不在这里）
#
# 换句话说：交易所数据的“采集层”应该在别的模块里实现，
# 采集层把 message(dict) 传给 Stream.save_trade_data()，这里负责存盘。
# ------------------------------------------------------------

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Finstore:
    """
    Finstore：本地金融数据存储与读取工具

    目录结构（默认）：
    database/finstore/
      market_name=<market>/
        timeframe=<timeframe>/
          <symbol>/
            ohlcv_data.parquet
            technical_indicators.parquet
            raw_data.parquet

    参数说明：
    - market_name: 市场名称（比如 binance / okx / stocks 等，完全由你定义）
    - timeframe : 周期（1m/5m/1h/1d...）
    - base_directory: 根目录
    - enable_append: 是否对已有 parquet 追加（增量写入）
    - limit_data_lookback: 指标计算时只取最近 N 条（加速）
    - pair: 可选，用于某些市场符号组织方式（如 folder/pair）
    """

    def __init__(
        self,
        market_name: str,
        timeframe: str,
        base_directory: str = 'database/finstore',
        enable_append: bool = True,
        limit_data_lookback: int = -1,
        pair: str = ''
    ):
        self.base_directory = base_directory
        self.market_name = market_name
        self.timeframe = timeframe
        self.enable_append = enable_append
        self.limit_data_lookback = limit_data_lookback
        self.pair = pair

        # 分成三个“子模块”，职责拆分清晰：
        self.read = self.Read(self)
        self.write = self.Write(self)
        self.stream = self.Stream(self)

        # 调试：列一下目录里有哪些 symbol 文件夹
        self.list_items_in_dir()

    def list_items_in_dir(self):
        """列出 market + timeframe 目录下的所有子项（调试用）"""
        dir_path = os.path.join(
            self.base_directory,
            f"market_name={self.market_name}",
            f"timeframe={self.timeframe}"
        )
        os.makedirs(dir_path, exist_ok=True)
        try:
            items = os.listdir(dir_path)
            print(f"Len items in '{dir_path}': {len(items)}")
        except FileNotFoundError:
            print(f"Directory '{dir_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # ============================================================
    # Read：读取模块（只读本地 parquet，不会请求交易所）
    # ============================================================
    class Read:
        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.pair = finstore_instance.pair

        def symbol(self, symbol: str):
            """
            读取某个 symbol 的 OHLCV parquet（ohlcv_data.parquet）

            使用 DuckDB 读取 parquet 的好处：
            - 可直接 SQL 读 parquet（无需先导入数据库）
            - 读取时可多线程（PRAGMA threads=4）
            - 对分析型查询很友好
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol,
                'ohlcv_data.parquet'
            )
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Parquet file not found for symbol '{symbol}' at '{file_path}'")

            conn = duckdb.connect()
            conn.execute("PRAGMA threads=4")

            df = conn.execute(f"SELECT * FROM read_parquet('{file_path}')").fetchdf()
            conn.close()

            return symbol, df

        def merged_df(self, symbol: str):
            """
            读取并合并：
            - ohlcv_data.parquet
            - technical_indicators.parquet

            technical_indicators.parquet 是“长表”：
              timestamp | indicator_name | indicator_value
            这里会 pivot 成“宽表”，让每个指标变成一列，然后按 timestamp merge 到 OHLCV。
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol,
                'ohlcv_data.parquet'
            )
            technical_indicators_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol,
                'technical_indicators.parquet'
            )

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Parquet file not found for symbol '{symbol}' at '{file_path}'")
            if not os.path.isfile(technical_indicators_path):
                raise FileNotFoundError(
                    f"Technical indicators file not found for symbol '{symbol}' at '{technical_indicators_path}'"
                )

            conn = duckdb.connect()
            conn.execute("PRAGMA threads=4")

            df = conn.execute(f"SELECT * FROM read_parquet('{file_path}')").fetchdf()
            technical_indicators_df = conn.execute(
                f"SELECT * FROM read_parquet('{technical_indicators_path}')"
            ).fetchdf()

            # 去重：同一 timestamp + indicator_name 保留一条
            technical_indicators_df = technical_indicators_df.drop_duplicates(subset=['timestamp', 'indicator_name'])

            # pivot：长表 -> 宽表
            technical_indicators_df = technical_indicators_df.pivot(
                index='timestamp',
                columns='indicator_name',
                values='indicator_value'
            ).reset_index()

            # merge：把指标列合并到 OHLCV
            merged_df = df.merge(technical_indicators_df, on='timestamp', how='left')

            conn.close()

            return symbol, merged_df

        def symbol_list(self, symbol_list: list, merged_dataframe: bool = False):
            """
            并行读取多个 symbol
            返回 {symbol: df}

            注意：
            - 这里使用 ProcessPoolExecutor 多进程并行
            - merged_dataframe=True 时读取合并后的大表（OHLCV+指标）
            """
            results = {}
            with ProcessPoolExecutor() as executor:
                if merged_dataframe:
                    futures = {executor.submit(self.merged_df, symbol): symbol for symbol in symbol_list}
                else:
                    futures = {executor.submit(self.symbol, symbol): symbol for symbol in symbol_list}

                for future in futures:
                    symbol = futures[future]
                    try:
                        symbol, df = future.result()
                        results[symbol] = df
                    except Exception as e:
                        print(f"Error reading data for symbol {symbol}: {e}")

            return results

        def get_symbol_list(self):
            """
            获取当前 market+timeframe 下本地已有的 symbol 列表（通过目录枚举）
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}"
            )
            if not os.path.isdir(file_path):
                raise FileNotFoundError(f"Directory not found for market '{self.market_name}' at '{file_path}'")

            # pair 的用途：如果你目录层级需要拼接成 folder/pair，就在这里做
            if self.pair != '':
                symbol_list = [
                    str(folder) + '/' + str(self.pair)
                    for folder in os.listdir(file_path)
                    if os.path.isdir(os.path.join(file_path, folder))
                ]
            else:
                symbol_list = [
                    folder
                    for folder in os.listdir(file_path)
                    if os.path.isdir(os.path.join(file_path, folder))
                ]
            return symbol_list

    # ============================================================
    # Write：写入模块（只写本地 parquet，不会请求交易所）
    # ============================================================
    class Write:
        """
        写入模块：
        - 写 OHLCV 到 ohlcv_data.parquet
        - 写技术指标到 technical_indicators.parquet

        enable_append=True 时支持增量追加：
        - 会先读旧 parquet 再 concat 新数据再去重
        （数据量大时，这种“读全量再写回”会变慢，是可优化点）
        """

        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.enable_append = finstore_instance.enable_append
            self.limit_data_lookback = finstore_instance.limit_data_lookback

        def symbol(self, symbol: str, data: pd.DataFrame):
            """
            写入某个 symbol 的 OHLCV 数据到 parquet
            """
            dir_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol
            )
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, 'ohlcv_data.parquet')

            # 增量追加：读旧数据 + 拼接 + timestamp 去重
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                data = pd.concat([existing_df, data], ignore_index=True)
                data = data.drop_duplicates(subset=['timestamp'])

            data.to_parquet(file_path, index=False, compression='zstd')

        def symbol_list(self, data_ohlcv: dict):
            """
            并行写入多个 symbol 的 OHLCV
            data_ohlcv: {symbol: df}
            """
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.symbol, symbol, data): symbol
                    for symbol, data in data_ohlcv.items()
                }
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Writing symbols"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error writing data for symbol : {e}")

        def technical_data(self, symbol: str, indicators_df: pd.DataFrame):
            """
            写技术指标到 technical_indicators.parquet

            要求 indicators_df 结构类似：
              timestamp | indicator_name | indicator_value
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol,
                'technical_indicators.parquet'
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 补齐元信息
            indicators_df['symbol'] = symbol
            indicators_df['timeframe'] = self.timeframe

            # 标准化字段顺序
            formatted_df = indicators_df[['symbol', 'timeframe', 'timestamp', 'indicator_name', 'indicator_value']]
            formatted_df.loc[:, 'indicator_value'] = formatted_df['indicator_value'].astype(float)

            # 增量追加：读旧数据 + 拼接 + 去重
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                formatted_df = pd.concat([existing_df, formatted_df], ignore_index=True)
                formatted_df = formatted_df.drop_duplicates(
                    subset=['timestamp', 'indicator_name', 'symbol', 'timeframe']
                )

            formatted_df.to_parquet(file_path, index=False, compression='zstd')

        def process_indicator(self, symbol, df, calculation_func, calculation_kwargs):
            """
            单个 symbol 的指标计算与写入
            - limit_data_lookback > 0 时只使用最近 N 条 K线数据算指标
            """
            try:
                if self.limit_data_lookback > 0:
                    df = df.iloc[-self.limit_data_lookback:]

                indicators_df = calculation_func(df, **calculation_kwargs)
                self.technical_data(symbol=symbol, indicators_df=indicators_df)
            except Exception as e:
                print(f"Error calculating {calculation_func.__name__} for {symbol}: {e}")
                return

        def indicator(self, ohlcv_data: dict, calculation_func, **calculation_kwargs):
            """
            批量计算并写入技术指标：
            - ohlcv_data: {symbol: df}
            - calculation_func: 你的指标函数（应返回 timestamp/indicator_name/indicator_value 格式的 df）
            """
            use_multiprocessing = True

            if use_multiprocessing:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(self.process_indicator, symbol, df, calculation_func, calculation_kwargs)
                        for symbol, df in ohlcv_data.items()
                    ]
                    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing symbols"):
                        pass
            else:
                for symbol, df in tqdm(ohlcv_data.items(), desc="Processing symbols"):
                    self.process_indicator(symbol, df, calculation_func, calculation_kwargs)

            print(
                f"{calculation_func.__name__} calculation and insertion completed for market: "
                f"{self.market_name} and timeframe: {self.timeframe}"
            )

    # ============================================================
    # Stream：流式落盘模块（接收外部消息 message，自己不去连交易所）
    # ============================================================
    class Stream:
        """
        Stream 模块用于实时落盘，但它不是交易所客户端：

        ✅ 你把交易所 websocket/REST 获取到的 message(dict) 传进来
        ✅ 这里根据 preset/parse_func 解析为统一格式并写 parquet
        ❌ 这里不维护 websocket 连接、不请求交易所、不处理鉴权

        所以它与交易所 API 是解耦的：采集层和存储层分离。
        """

        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.enable_append = finstore_instance.enable_append

        # 预设解析器：把“交易所消息结构”映射到你的统一 schema
        PRESET_CONFIGS = {
            # Binance kline 消息 -> OHLCV 行
            'binance_kline': lambda message: {
                'timestamp': message['k']['t'],
                'open': float(message['k']['o']),
                'high': float(message['k']['h']),
                'low': float(message['k']['l']),
                'close': float(message['k']['c']),
                'volume': float(message['k']['v']),
                'buy_volume': float(message['k']['V']),
                # dedup 用于去重：kline 开始时间
                'dedup': message['k']['t'],
            },

            # Binance aggTrade 消息 -> 成交行
            'agg_trade': lambda message: {
                'event_type': message['e'],
                'event_time': message['E'],
                'symbol': message['s'],
                'aggregate_trade_id': message['a'],
                'price': float(message['p']),
                'quantity': float(message['q']),
                'first_trade_id': message['f'],
                'last_trade_id': message['l'],
                'trade_time': message['T'],
                'is_buyer_maker': message['m'],
                # dedup 用于去重：聚合成交 id
                'dedup': message['a'],
            },
        }

        def save_trade_data(
            self,
            symbol: str,
            message: dict,
            parse_func: callable = None,
            preset: str = None,
            save_raw_data: bool = True
        ):
            """
            保存“外部传入的一条消息”到 parquet

            重点：
            - message 的来源不在这里：通常来自你写的交易所采集器（websocket回调等）
            - 这里的职责：解析 -> 去重 -> 追加写 parquet

            参数：
            - parse_func: 自定义解析函数（message -> dict row）
            - preset: 预设解析器名（如 'binance_kline'）
            - save_raw_data: 是否同时保存原始 message 到 raw_data.parquet
            """
            dir_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol
            )
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, 'ohlcv_data.parquet')

            # 选择解析器：preset 优先
            if preset and preset in self.PRESET_CONFIGS:
                parse_func = self.PRESET_CONFIGS[preset]

            # 解析 message 为 1 行 DataFrame
            if parse_func:
                df = pd.DataFrame([parse_func(message)])
            else:
                # 没提供解析器就原样保存（字段不一定统一）
                df = pd.DataFrame([message])

            # 增量追加 + 去重
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

                # 去重逻辑：优先 timestamp，否则用 dedup
                if 'timestamp' in df.columns:
                    df = df.drop_duplicates(subset=['timestamp'], keep='last')
                elif 'dedup' in df.columns:
                    df = df.drop_duplicates(subset=['dedup'], keep='last')

            df.to_parquet(file_path, index=False, compression='zstd')

            # 可选：保存原始消息（用于审计/回放/排查）
            if save_raw_data:
                df_raw = pd.DataFrame([message])
                raw_path = os.path.join(dir_path, 'raw_data.parquet')

                if os.path.isfile(raw_path) and self.enable_append:
                    existing_df = pd.read_parquet(raw_path)
                    df_raw = pd.concat([existing_df, df_raw], ignore_index=True)

                df_raw.to_parquet(raw_path, index=False, compression='zstd')

        def fetch_trade_data(self, symbol: str) -> pd.DataFrame:
            """
            读取某个 symbol 的 ohlcv_data.parquet 并返回 DataFrame
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol,
                'ohlcv_data.parquet'
            )

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Trade data file not found for symbol '{symbol}' at '{file_path}'")

            return pd.read_parquet(file_path)
