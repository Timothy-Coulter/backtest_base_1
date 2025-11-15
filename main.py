"""QuantBench Main Entry Point.

This module provides the main entry point for the QuantBench application.
"""

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktestRunConfig, get_config


def main() -> None:
    """Bootstrap the backtest engine with an immutable runtime configuration snapshot."""
    run_config = BacktestRunConfig(get_config()).build()
    BacktestEngine(config=run_config)
    print("Backtest engine initialised with a validated configuration snapshot.")


if __name__ == "__main__":
    main()
