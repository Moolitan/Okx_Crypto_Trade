# OKX Crypto Trading Framework

A **modular Python trading framework** built on the **OKX official SDK**, designed for developing, testing, and running automated cryptocurrency trading strategies.

This project focuses on **clean architecture and extensibility**, separating exchange access, account management, and trading logic into independent components.

> ⚠️ This project is for educational and research purposes only. Trading cryptocurrencies involves significant risk.

---

## Purpose

The goal of this project is to provide a **flexible and extensible foundation** for algorithmic trading on OKX, allowing you to:

- Develop and switch between multiple trading strategies
- Reuse the same exchange and account logic across strategies
- Manage positions, leverage, and risk in a structured way
- Gradually evolve from simple strategies to more advanced systems

---

## Architecture Overview

The project is organized into three main layers:

- **Exchange Layer (`exchanges/`)**Handles all interactions with OKX, such as market data retrieval and order placement.
- **Account Layer (`accounts/`)**Manages account-related operations including balance queries, leverage settings, and position tracking.
- **Strategy Layer (`strategies/`)**
  Contains trading strategies that implement their own decision logic while relying on the exchange and account layers.

This separation makes it easy to extend or replace individual components without affecting the rest of the system.

---

## Key Characteristics

- Modular, object-oriented design
- OKX official Python SDK integration
- Environment-based credential management (`.env`)
- Strategy-agnostic core (no strategy logic hard-coded into infrastructure)
- Designed for future expansion (risk management, backtesting, multiple strategies)

---

## Requirements

- Python 3.9+
- OKX account with API access

Install dependencies:

```bash
pip install -r requirements.txt
```
