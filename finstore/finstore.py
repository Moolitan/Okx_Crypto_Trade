import duckdb
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm
import sys

# ------------------------------------------------------------
# Notes:
# This code implements a local financial data warehouse (Finstore).
#
# ✅ Responsibilities:
# - Store / read / merge local market data
# - Batch compute technical indicators
# - Stream ingestion to local storage (append + dedup)
#
# ❌ Not responsible for:
# - Any exchange API calls (REST/WebSocket connections, auth, requests, etc.)
#
# In other words:
# The "data collection layer" (exchange clients) should live elsewhere.
# That layer passes message(dict) into Stream.save_trade_data(),
# and this module handles persistence only.
# ------------------------------------------------------------

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Finstore:
    """
    Finstore: a local storage and retrieval utility for financial data.

    Default directory layout:
    database/finstore/
      market_name=<market>/
        timeframe=<timeframe>/
          <symbol>/
            ohlcv_data.parquet
            technical_indicators.parquet
            raw_data.parquet

    Parameters:
    - market_name: market identifier (e.g., binance / okx / stocks; user-defined)
    - timeframe: bar interval (1m/5m/1h/1d...)
    - base_directory: root directory path
    - enable_append: whether to append to existing parquet (incremental write)
    - limit_data_lookback: compute indicators using only last N bars (speed-up)
    - pair: optional folder suffix for markets that organize symbols as folder/pair
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

        self.read = self.Read(self)
        self.write = self.Write(self)
        self.stream = self.Stream(self)

        self.list_items_in_dir()

    def list_items_in_dir(self):
        """List all sub-items under market+timeframe directory (debug helper)."""
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
    # Read: read-only module (local parquet only, no exchange requests)
    # ============================================================
    class Read:
        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.pair = finstore_instance.pair

        def symbol(self, symbol: str):
            """
            Load OHLCV parquet for a single symbol (ohlcv_data.parquet).

            Why DuckDB for parquet reading:
            - Query parquet via SQL without importing into a DB
            - Multi-threaded scanning (PRAGMA threads=4)
            - Friendly for analytical queries
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
            Load and merge:
            - ohlcv_data.parquet
            - technical_indicators.parquet

            technical_indicators.parquet is a "long" table:
            timestamp | indicator_name | indicator_value

            This method pivots it into a "wide" table where each indicator is a column,
            then merges by timestamp.
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

            # Deduplicate: keep one record per timestamp + indicator_name
            technical_indicators_df = technical_indicators_df.drop_duplicates(subset=['timestamp', 'indicator_name'])

            # pivot: long format -> wide format
            technical_indicators_df = technical_indicators_df.pivot(
                index='timestamp',
                columns='indicator_name',
                values='indicator_value'
            ).reset_index()

            # merge: merge indicator columns into OHLCV
            merged_df = df.merge(technical_indicators_df, on='timestamp', how='left')

            conn.close()

            return symbol, merged_df

        def symbol_list(self, symbol_list: list, merged_dataframe: bool = False):
            """
            Read multiple symbols in parallel.
            Returns {symbol: df}

            Note:
            - Uses ProcessPoolExecutor for multi-process parallelism.
            - When merged_dataframe=True, reads the merged big table (OHLCV+indicators).
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
            Get the list of locally available symbols under current market+timeframe (via directory enumeration).
            """
            file_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}"
            )
            if not os.path.isdir(file_path):
                raise FileNotFoundError(f"Directory not found for market '{self.market_name}' at '{file_path}'")

            # Usage of 'pair': If your directory structure requires concatenation like folder/pair, handle it here.
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
    # Write: Writing module (Local parquet only, does not request exchange)
    # ============================================================
    class Write:
        """
        Write module:
        - Write OHLCV to ohlcv_data.parquet
        - Write technical indicators to technical_indicators.parquet

        Supports incremental append when enable_append=True:
        - Reads old parquet, concats new data, then deduplicates.
        (With large data, this 'read full then write back' approach slows down and is an optimization point).
        """

        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.enable_append = finstore_instance.enable_append
            self.limit_data_lookback = finstore_instance.limit_data_lookback

        def symbol(self, symbol: str, data: pd.DataFrame):
            """
            Write OHLCV data for a specific symbol to parquet
            """
            dir_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol
            )
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, 'ohlcv_data.parquet')

            # Incremental append: read old data + concat + deduplicate by timestamp
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                data = pd.concat([existing_df, data], ignore_index=True)
                data = data.drop_duplicates(subset=['timestamp'])

            data.to_parquet(file_path, index=False, compression='zstd')

        def symbol_list(self, data_ohlcv: dict):
            """
            Write OHLCV for multiple symbols in parallel.
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
            Write technical indicators to technical_indicators.parquet.

            Requires indicators_df structure like:
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

            # Fill metadata
            indicators_df['symbol'] = symbol
            indicators_df['timeframe'] = self.timeframe

            # Standardize column order
            formatted_df = indicators_df[['symbol', 'timeframe', 'timestamp', 'indicator_name', 'indicator_value']]
            formatted_df.loc[:, 'indicator_value'] = formatted_df['indicator_value'].astype(float)

            # Incremental append: read old data + concat + deduplicate
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                formatted_df = pd.concat([existing_df, formatted_df], ignore_index=True)
                formatted_df = formatted_df.drop_duplicates(
                    subset=['timestamp', 'indicator_name', 'symbol', 'timeframe']
                )

            formatted_df.to_parquet(file_path, index=False, compression='zstd')

        def process_indicator(self, symbol, df, calculation_func, calculation_kwargs):
            """
            Indicator calculation and writing for a single symbol.
            - When limit_data_lookback > 0, only use the last N bars of K-line data to calculate indicators.
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
            Batch calculate and write technical indicators:
            - ohlcv_data: {symbol: df}
            - calculation_func: Your indicator function (should return df in timestamp/indicator_name/indicator_value format)
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
    # Stream: Streaming ingestion module (Receives external messages, does NOT connect to exchange)
    # ============================================================
    class Stream:
        """
        Stream module is for real-time persistence, but it is NOT an exchange client:

        ✅ You pass in the message(dict) obtained from exchange websocket/REST.
        ✅ Parses into unified format based on preset/parse_func and writes to parquet.
        ❌ Does NOT maintain websocket connections, request exchanges, or handle auth.

        Thus it is decoupled from Exchange APIs: separation of collection layer and storage layer.
        """

        def __init__(self, finstore_instance):
            self.market_name = finstore_instance.market_name
            self.timeframe = finstore_instance.timeframe
            self.base_directory = finstore_instance.base_directory
            self.enable_append = finstore_instance.enable_append

        # Preset parsers: map 'exchange message structure' to your unified schema
        PRESET_CONFIGS = {
            # Binance kline message -> OHLCV row
            'binance_kline': lambda message: {
                'timestamp': message['k']['t'],
                'open': float(message['k']['o']),
                'high': float(message['k']['h']),
                'low': float(message['k']['l']),
                'close': float(message['k']['c']),
                'volume': float(message['k']['v']),
                'buy_volume': float(message['k']['V']),
                # dedup used for deduplication: kline start time
                'dedup': message['k']['t'],
            },

            # Binance aggTrade message -> Trade row
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
                # dedup used for deduplication: aggregate trade id
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
            Save 'one externally passed message' to parquet.

            Key points:
            - The source of the message is not here: usually comes from your exchange collector (websocket callbacks, etc.)
            - Responsibility here: Parse -> Deduplicate -> Append write to parquet

            Parameters:
            - parse_func: Custom parse function (message -> dict row)
            - preset: Preset parser name (e.g., 'binance_kline')
            - save_raw_data: Whether to save raw message to raw_data.parquet
            """
            dir_path = os.path.join(
                self.base_directory,
                f"market_name={self.market_name}",
                f"timeframe={self.timeframe}",
                symbol
            )
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, 'ohlcv_data.parquet')

            # Select parser: preset takes precedence
            if preset and preset in self.PRESET_CONFIGS:
                parse_func = self.PRESET_CONFIGS[preset]

            # Parse message into a 1-row DataFrame
            if parse_func:
                df = pd.DataFrame([parse_func(message)])
            else:
                # If no parser provided, save as is (fields may not be unified)
                df = pd.DataFrame([message])

            # Incremental append + deduplicate
            if os.path.isfile(file_path) and self.enable_append:
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

                # Deduplication logic: prioritize timestamp, otherwise use dedup
                if 'timestamp' in df.columns:
                    df = df.drop_duplicates(subset=['timestamp'], keep='last')
                elif 'dedup' in df.columns:
                    df = df.drop_duplicates(subset=['dedup'], keep='last')

            df.to_parquet(file_path, index=False, compression='zstd')

            # Optional: Save raw message (for audit/replay/troubleshooting)
            if save_raw_data:
                df_raw = pd.DataFrame([message])
                raw_path = os.path.join(dir_path, 'raw_data.parquet')

                if os.path.isfile(raw_path) and self.enable_append:
                    existing_df = pd.read_parquet(raw_path)
                    df_raw = pd.concat([existing_df, df_raw], ignore_index=True)

                df_raw.to_parquet(raw_path, index=False, compression='zstd')

        def fetch_trade_data(self, symbol: str) -> pd.DataFrame:
            """
            Read ohlcv_data.parquet for a specific symbol and return DataFrame
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