"""QuantBench Main Entry Point.

This module provides the main entry point for the QuantBench application.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from backtester.cli import build_run_config_from_cli, collect_overrides, parse_runtime_args
from backtester.core.backtest_engine import BacktestEngine


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap the backtest engine with runtime configuration overrides."""
    args = parse_runtime_args(argv)
    if os.environ.get("BACKTEST_DRY_RUN", "").lower() in {"1", "true", "yes"}:
        args.dry_run = True
    try:
        overrides = collect_overrides(args)
    except ValueError as exc:
        print(f"[quantbench] {exc}", file=sys.stderr)
        return 2

    run_config = build_run_config_from_cli(overrides)
    engine = BacktestEngine(config=run_config)

    if args.dry_run:
        print("Backtest engine initialised with a validated configuration snapshot.")
        return 0

    try:
        engine.run_backtest(
            ticker=overrides.primary_ticker,
            start_date=overrides.data.get('start_date'),
            end_date=overrides.data.get('finish_date'),
            interval=overrides.data.get('freq'),
            strategy_params=overrides.strategy or None,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"[quantbench] Backtest execution failed: {exc}", file=sys.stderr)
        return 1

    print("Backtest engine initialised with a validated configuration snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
