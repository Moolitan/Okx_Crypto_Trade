# OKX 加密货币交易框架

一个基于 **OKX 官方 SDK** 构建的模块化 Python 交易框架，用于开发、测试和运行自动化加密货币交易策略。

本项目专注于**清晰的架构和可扩展性**，将交易所接入、账户管理和交易逻辑分离为独立组件。

> ⚠️ 本项目仅供教育和研究目的。加密货币交易涉及重大风险。

---

## 项目目的

本项目旨在为 OKX 上的算法交易提供**灵活且可扩展的基础框架**，允许您：

- 开发并在多种交易策略之间切换
- 在不同策略间复用相同的交易所和账户逻辑
- 以结构化的方式管理仓位、杠杆和风险
- 逐步从简单策略演进到更复杂的系统

---

## 架构概述

项目采用三层架构设计：

- **交易所层 (`executor/`)**：处理与 OKX 的所有交互，如市场数据获取和订单下达。
- **账户层 (`accounts/`)**：管理账户相关操作，包括余额查询、杠杆设置和持仓跟踪。
- **策略层 (`strategy/`)**：包含交易策略，实现各自的决策逻辑，同时依赖交易所和账户层。
- **数据层 (`data/`)**：处理市场数据的收集、存储和指标计算。

这种分离使得可以轻松扩展或替换单个组件，而不影响系统的其他部分。

---

## 目录结构说明

### 📁 `accounts/` - 账户管理模块

负责账户相关的操作，包括余额查询、杠杆设置、持仓管理等。

- `base.py`: 账户抽象基类，定义账户操作的接口规范
- `okx_account.py`: OKX 账户的具体实现，继承自基类，实现实际的账户操作

### 📁 `data/` - 数据层模块

处理市场数据的收集、存储、指标计算和持久化。

#### `data/okx/` - OKX 数据相关

- `core.py`: OKX API 基础混入类，提供统一的 API 调用、重试和限流机制
- `data.py`: 市场数据总线（MarketDataBus），负责收集、缓存和持久化市场数据
- `store.py`: 数据持久化存储层，使用 DuckDB 存储 OHLCV、资金费率、成交记录等
- `indicators.py`: 量化指标计算引擎，提供多种技术指标（MA、RSI、MACD、KDJ 等）
- `path.py`: 数据库路径配置

#### `data/binance/` - Binance 数据相关

预留用于 Binance 交易所的数据接口（当前为空）

### 📁 `database/` - 数据库目录

存储本地数据库文件和日志。

- `db/`: DuckDB 数据库文件存储位置
- `logs/`: 日志文件存储位置

### 📁 `executor/` - 执行器模块

实现交易所接口和交易策略执行逻辑。

- `base.py`: 交易所抽象基类，定义交易所操作的接口规范
- `okx_sdk.py`: OKX SDK 交易所实现，继承自基类，实现行情获取和订单下达
- `momentum.py`: 动量策略实现（MomentumV1），示例策略代码
- `Agent/`: 智能体相关代码（预留）

### 📁 `finstore/` - 金融数据存储模块

本地金融数据仓库，用于存储、读取、合并本地市场数据，批量计算技术指标。

- `finstore.py`: 核心存储类，提供读取、写入和流式数据存储功能

### 📁 `logger/` - 日志模块

提供统一的日志记录功能。

- `custom_logger.py`: 自定义日志工具类，提供统一的日志输出接口

### 📁 `strategy/` - 策略模块

包含各种交易策略实现。

### 📁 `config/` - 配置模块

项目配置文件（当前为空，预留用于配置管理）

### 📁 `scripts/` - 脚本目录

包含测试脚本和运行脚本。

- `test/`: 测试脚本
- `runner/`: 运行脚本（预留）

---

## 核心特性

- **模块化、面向对象设计**
- **集成 OKX 官方 Python SDK**
- **基于环境变量的凭证管理（`.env`）**
- **策略无关的核心架构**（基础设施中不硬编码策略逻辑）
- **数据持久化**（使用 DuckDB 存储历史数据）
- **丰富的技术指标库**（MA、RSI、MACD、KDJ、布林带、ATR 等）
- **为未来扩展而设计**（风险管理、回测、多策略支持）

---

## 环境要求

- Python 3.9+
- OKX 账户及 API 访问权限

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 配置说明

### 环境变量配置

在项目根目录创建 `.env` 文件，配置以下 OKX API 凭证：

```env
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
```

> **注意**：请妥善保管 API 凭证，不要将其提交到版本控制系统。

---

## 快速开始

### 1. 克隆项目

```bash
git clone <repository_url>
cd Okx_Crypto_Trade
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件并填入 OKX API 凭证。

### 4. 运行测试

```bash
python scripts/test/test_okx_indicators_btc.py
```

---

## 使用示例

### 使用市场数据总线

```python
from data.okx.data import MarketDataBus

# 初始化市场数据总线
bus = MarketDataBus(
    base_symbol="BTC-USDT-SWAP",
    bar="1m",
    flag="0"  # "0" 实盘, "1" 模拟盘
)

# 获取 K 线数据
df = bus.get_klines_df()
print(df.tail())
```

### 使用账户管理

```python
from accounts.okx_account import OkxAccount

# 初始化账户
account = OkxAccount(flag="0", leverage=10)

# 获取可用余额
balance = account.get_usdt_free()
print(f"可用余额: {balance} USDT")

# 获取持仓信息
positions = account.get_all_positions(simple=True)
print(positions)
```

### 使用技术指标

```python
from data.okx.indicators import QuantitativeIndicator

# 初始化指标计算器
indicator = QuantitativeIndicator()

# 计算所有指标
df_with_indicators = indicator.compute_all(df)

# 查看结果
print(df_with_indicators[['close', 'MA_7', 'RSI', 'MACD_dif']].tail())
```

---

## 数据存储

项目使用 DuckDB 作为主要数据存储引擎：

- **OHLCV 数据**：存储在 `database/db/okx.duckdb` 中
- **技术指标**：可通过 `finstore` 模块存储为 Parquet 格式
- **成交记录和信号**：存储在 DuckDB 的 `fills` 和 `signals` 表中

---

## 注意事项

1. **风险管理**：本项目提供的策略示例仅供学习参考，实际使用前请充分测试并设置合理的风险控制参数。
2. **API 限制**：注意 OKX API 的调用频率限制，避免触发限流。
3. **数据持久化**：建议定期备份数据库文件，防止数据丢失。
4. **实盘交易**：使用实盘 API 前，建议先在模拟环境中充分测试。

---

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

---

## 许可证

本项目仅供教育和研究目的使用。使用本框架进行交易的所有风险由使用者自行承担。

---

## 联系方式

如有问题或建议，请通过 Issue 联系。
