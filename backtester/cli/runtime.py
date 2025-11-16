"""Runtime helpers for CLI-driven configuration overrides."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from backtester.core.config import (
    BacktesterConfig,
    DataRetrievalConfig,
    PortfolioConfig,
    StrategyConfig,
    get_config,
)
from backtester.core.config_processor import ConfigInput, ConfigProcessor


@dataclass(frozen=True, slots=True)
class CLIOverrides:
    """Container for parsed CLI/environment overrides."""

    data: DataRetrievalConfig | None = None
    strategy: StrategyConfig | None = None
    portfolio: PortfolioConfig | None = None
    primary_ticker: str | None = None
    _component_overrides: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Freeze raw override mappings to keep CLIOverrides immutable."""
        frozen_payload = {
            name: dict(payload) for name, payload in self._component_overrides.items()
        }
        object.__setattr__(self, "_component_overrides", MappingProxyType(frozen_payload))

    def has_overrides(self) -> bool:
        """Return True when any component overrides are present."""
        return bool(self._component_overrides)

    def component_overrides(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the raw component override payloads."""
        return self._component_overrides

    def get_override(self, component: str) -> Mapping[str, Any] | None:
        """Return the raw overrides for an individual component."""
        return self._component_overrides.get(component)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the shared argparse parser for CLI entrypoints."""
    parser = argparse.ArgumentParser(description="QuantBench backtest runner.")
    parser.add_argument(
        "-c",
        "--config",
        help=(
            "Path to a BacktesterConfig YAML (see component_configs/core/) that is merged "
            "before applying CLI/environment overrides."
        ),
    )
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
    *,
    processor: ConfigProcessor | None = None,
    config_source: ConfigInput | None = None,
) -> CLIOverrides:
    """Merge CLI and environment values into component override dictionaries."""
    env = env or os.environ
    processor = processor or ConfigProcessor(base=get_config())
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
    data_overrides = _build_data_overrides(tickers, start_date, preset, finish_date, freq)

    strategy_name = _select_value(env.get("BACKTEST_STRATEGY_NAME"), args.strategy_name)
    ma_short = _select_int(env.get("BACKTEST_STRATEGY_MA_SHORT"), args.strategy_ma_short)
    ma_long = _select_int(env.get("BACKTEST_STRATEGY_MA_LONG"), args.strategy_ma_long)
    strategy_overrides = _build_strategy_overrides(strategy_name, ma_short, ma_long)

    component_payloads: dict[str, dict[str, Any]] = {}
    data_model: DataRetrievalConfig | None = None
    if data_overrides:
        component_payloads['data'] = data_overrides
        resolved = processor.apply_component(
            "data",
            overrides=data_overrides,
            base=config_source,
        )
        data_model = cast(DataRetrievalConfig | None, resolved)

    strategy_model: StrategyConfig | None = None
    if strategy_overrides:
        component_payloads['strategy'] = strategy_overrides
        resolved = processor.apply_component(
            "strategy",
            overrides=strategy_overrides,
            base=config_source,
        )
        strategy_model = cast(StrategyConfig | None, resolved)

    return CLIOverrides(
        data=data_model,
        strategy=strategy_model,
        portfolio=None,
        primary_ticker=primary_ticker,
        _component_overrides=component_payloads,
    )


def build_run_config_from_cli(
    overrides: CLIOverrides,
    *,
    config_source: ConfigInput | None = None,
    base_config: BacktesterConfig | None = None,
    processor: ConfigProcessor | None = None,
) -> BacktesterConfig:
    """Apply parsed CLI overrides to the provided base configuration."""
    processor = processor or ConfigProcessor(base=base_config or get_config())
    component_overrides = overrides.component_overrides()
    resolved = processor.apply(
        source=config_source,
        component_overrides=component_overrides if component_overrides else None,
    )
    assert isinstance(resolved, BacktesterConfig)
    return resolved


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


def _build_data_overrides(
    tickers: Sequence[str],
    start_date: str | None,
    preset: str | None,
    finish_date: str | None,
    freq: str | None,
) -> dict[str, Any]:
    """Construct the data override payload from CLI/env sources."""
    _validate_date_inputs(start_date, preset)
    overrides: dict[str, Any] = {}
    if tickers:
        overrides['tickers'] = list(tickers)
    if preset:
        overrides['start_date'] = preset
    elif start_date:
        overrides['start_date'] = start_date
    if finish_date:
        overrides['finish_date'] = finish_date
    if freq:
        overrides['freq'] = freq
    return overrides


def _build_strategy_overrides(
    strategy_name: str | None,
    ma_short: int | None,
    ma_long: int | None,
) -> dict[str, Any]:
    """Construct the strategy override payload from CLI/env sources."""
    overrides: dict[str, Any] = {}
    if strategy_name:
        overrides['strategy_name'] = strategy_name
    if ma_short is not None:
        overrides['ma_short'] = ma_short
    if ma_long is not None:
        overrides['ma_long'] = ma_long
    return overrides
