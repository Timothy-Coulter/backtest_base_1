"""Quantitative Backtesting Framework.

This package provides a comprehensive framework for quantitative portfolio backtesting
with advanced risk management, strategy implementation, and performance analysis.
"""

# Core exports
from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.portfolio import DualPoolPortfolio

__version__ = "0.1.0"
__all__ = [
    "BacktestEngine",
    "BacktesterConfig",
    "DualPoolPortfolio",
]
