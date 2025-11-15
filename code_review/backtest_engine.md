# Backtest Engine

## `load_data` ignores runtime overrides
- **Where:** `backtester/core/backtest_engine.py:135-151`
- **Issue:** The method logs the caller-supplied `ticker`, `start_date`, `end_date`, and `interval`, but immediately returns whatever `DataRetrieval.get_data()` last loaded from the config. The `DataRetrieval` instance is never reconfigured per call, so every backtest run reuses the same dataset even if different symbols or horizons are requested.
- **Why it matters:** Strategies cannot be parameterised per run, optimisers cannot sweep symbols/periods, and callers get a false sense of control because the log line reports the requested range even though the data does not change.
- **Fix:** Thread the method arguments into `DataRetrieval` (e.g. call `self.data_handler.update_config(...)` before `get_data()`, or instantiate a fresh retriever per run) so the retrieved frame actually matches the requested inputs, and add regression tests that assert the handler receives the overrides.

## Base/alpha pool tracking calls attributes as callables
- **Where:** `backtester/core/backtest_engine.py:583-591`
- **Issue:** The loop appending `base_values` wraps `getattr(..., 'base_pool', fallback)` and then calls the result: `getattr(...)().capital`. When `self.current_portfolio` is a dual‑pool portfolio, `base_pool` is a `PoolState`, so the extra `()` raises `TypeError: 'PoolState' object is not callable`. The bug is silently masked only because the fallback object is a type, but real dual‑pool backtests will crash the first time `_run_simulation` runs.
- **Fix:** Remove the spurious call and read `.capital` directly from the attribute, falling back to a simple struct when the attribute is missing.

## Positions never get revalued or stopped out
- **Where:** `backtester/core/backtest_engine.py:638-642` vs `backtester/portfolio/general_portfolio.py:292-333`
- **Issue:** `GeneralPortfolio.process_tick` expects a `market_data` map so it can update each `Position`, run stop-loss/take-profit checks, and recompute total value. `_process_signals_and_update_portfolio` only passes scalar `current_price/day_high/day_low`, leaving `market_data=None`. As a result no position updates ever happen, unrealised P&L stays frozen, and exit rules are never triggered.
- **Fix:** Pass a `{symbol: historical_data}` frame (or at least the most recent bar) into `process_tick`, and add regression tests that open a position and rely on the stop-loss to close it.

## Portfolio-level risk checks use share counts as market value
- **Where:** `backtester/core/backtest_engine.py:770-804`
- **Issue:** `_check_risk_management` builds a `positions_dict` by iterating the broker’s `positions` dictionary and emitting `{'market_value': position}` where `position` is simply the raw quantity. The downstream `RiskControlManager` therefore sees tiny “market values” (e.g. `1.0` instead of `$450`) and never triggers any of the configured limits.
- **Fix:** Translate broker positions into the structure `RiskControlManager` expects (symbol, side, market value, cost basis, etc.), ideally by reconciling with the portfolio’s position objects so both components share the same notion of exposure. Add tests that deliberately breach a limit and assert `_check_risk_management` cancels orders and records a risk signal.
