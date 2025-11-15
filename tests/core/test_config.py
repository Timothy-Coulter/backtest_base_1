"""Tests for configuration helpers and validation."""

import pytest

from backtester.core.config import (
    BacktesterConfig,
    BacktestRunConfig,
    DataRetrievalConfig,
    validate_run_config,
)


def test_backtest_run_config_applies_data_overrides() -> None:
    """Builder should merge overrides without mutating the base config."""
    base_config = BacktesterConfig()
    builder = BacktestRunConfig(base_config)

    run_config = builder.with_data_overrides(tickers=["MSFT"], start_date="2022-01-01").build()

    assert run_config.data is not None
    assert run_config.data.tickers == ["MSFT"]
    assert base_config.data is not None
    assert base_config.data.tickers != run_config.data.tickers


def test_validate_run_config_requires_tickers() -> None:
    """Missing tickers should raise during validation."""
    config = BacktesterConfig(data=DataRetrievalConfig(tickers=[]))

    with pytest.raises(ValueError, match="tickers"):
        validate_run_config(config)


def test_validate_run_config_checks_date_order() -> None:
    """Finish dates earlier than start dates must be rejected."""
    config = BacktesterConfig(
        data=DataRetrievalConfig(start_date="2023-01-02", finish_date="2023-01-01")
    )

    with pytest.raises(ValueError, match="finish_date"):
        validate_run_config(config)


def test_validate_run_config_prevents_negative_leverage() -> None:
    """Portfolio leverage inputs must be positive."""
    config = BacktesterConfig()
    assert config.portfolio is not None
    config.portfolio.leverage_base = -1.0

    with pytest.raises(ValueError, match="leverage_base"):
        validate_run_config(config)
