"""Runtime helpers for CLI-driven configuration overrides."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backtester.core.config import BacktesterConfig, BacktestRunConfig, get_config


@dataclass(frozen=True, slots=True)
class CLIOverrides:
    """Container for parsed CLI/environment overrides."""

    data: dict[str, Any]
    strategy: dict[str, Any]
    portfolio: dict[str, Any]
    primary_ticker: str | None = None

    def has_overrides(self) -> bool:
        """Return True when any component overrides are present."""
        return bool(self.data or self.strategy or self.portfolio)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the shared argparse parser for CLI entrypoints."""
    parser = argparse.ArgumentParser(description="QuantBench backtest runner.")
    parser.add_argument("--ticker", action="append", help="Ticker to include (repeatable).")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Space-delimited list of tickers that overrides the configured universe.",
    )
    parser.add_argument("--start-date", help="Explicit ISO start date (YYYY-MM-DD).")
    parser.add_argument("--finish-date", help="Explicit ISO finish date (YYYY-MM-DD).")
    parser.add_argument(
        "--date-preset",
        choices=["year", "ytd", "max", "month", "week"],
        help="Relative preset used instead of explicit start dates.",
    )
    parser.add_argument("--freq", help="Data frequency override (e.g. daily, 1h).")
    parser.add_argument("--strategy-name", help="Strategy name override.")
    parser.add_argument(
        "--strategy-ma-short",
        type=int,
        help="Short moving-average window passed through BacktestRunConfig.",
    )
    parser.add_argument(
        "--strategy-ma-long",
        type=int,
        help="Long moving-average window passed through BacktestRunConfig.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialise the engine without invoking BacktestEngine.run_backtest().",
    )
    return parser


def parse_runtime_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments using the shared parser."""
    parser = build_arg_parser()
    return parser.parse_args(list(argv) if argv is not None else None)


def collect_overrides(
    args: argparse.Namespace,
    env: Mapping[str, str] | None = None,
) -> CLIOverrides:
    """Merge CLI and environment values into component override dictionaries."""
    env = env or os.environ
    tickers = _merge_tickers(
        _split_env_values(env.get("BACKTEST_TICKERS")),
        args.ticker or [],
        args.tickers or [],
    )
    primary_ticker = tickers[0] if tickers else None

    start_date = _select_value(env.get("BACKTEST_START_DATE"), args.start_date)
    preset = _select_value(env.get("BACKTEST_DATE_PRESET"), args.date_preset)
    finish_date = _select_value(env.get("BACKTEST_FINISH_DATE"), args.finish_date)
    freq = _select_value(env.get("BACKTEST_FREQ"), args.freq)

    _validate_date_inputs(start_date, preset)

    data_overrides: dict[str, Any] = {}
    if tickers:
        data_overrides['tickers'] = tickers
    if preset:
        data_overrides['start_date'] = preset
    elif start_date:
        data_overrides['start_date'] = start_date
    if finish_date:
        data_overrides['finish_date'] = finish_date
    if freq:
        data_overrides['freq'] = freq

    strategy_overrides: dict[str, Any] = {}
    strategy_name = _select_value(env.get("BACKTEST_STRATEGY_NAME"), args.strategy_name)
    if strategy_name:
        strategy_overrides['strategy_name'] = strategy_name
    ma_short = _select_int(env.get("BACKTEST_STRATEGY_MA_SHORT"), args.strategy_ma_short)
    if ma_short is not None:
        strategy_overrides['ma_short'] = ma_short
    ma_long = _select_int(env.get("BACKTEST_STRATEGY_MA_LONG"), args.strategy_ma_long)
    if ma_long is not None:
        strategy_overrides['ma_long'] = ma_long

    return CLIOverrides(
        data=data_overrides,
        strategy=strategy_overrides,
        portfolio={},
        primary_ticker=primary_ticker,
    )


def build_run_config_from_cli(
    overrides: CLIOverrides,
    *,
    base_config: BacktesterConfig | None = None,
) -> BacktesterConfig:
    """Apply parsed CLI overrides to the provided base configuration."""
    builder = BacktestRunConfig(base_config or get_config())
    if overrides.data:
        builder.with_data_overrides(**overrides.data)
    if overrides.strategy:
        builder.with_strategy_overrides(**overrides.strategy)
    if overrides.portfolio:
        builder.with_portfolio_overrides(**overrides.portfolio)
    return builder.build()


# ---------------------------------------------------------------------------#
# Internal helpers
# ---------------------------------------------------------------------------#


def _split_env_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _merge_tickers(*sources: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        for ticker in source:
            normalized = ticker.strip()
            if not normalized:
                continue
            key = normalized.upper()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _select_value(env_value: str | None, cli_value: str | None) -> str | None:
    return cli_value if cli_value is not None else env_value


def _select_int(env_value: str | None, cli_value: int | None) -> int | None:
    if cli_value is not None:
        return cli_value
    if env_value is None:
        return None
    try:
        return int(env_value)
    except ValueError:
        raise ValueError(f"Invalid integer override: {env_value}") from None


def _validate_date_inputs(start_date: str | None, preset: str | None) -> None:
    if start_date and preset:
        raise ValueError(
            "start_date/start presets are mutually exclusive; remove one of the inputs."
        )
