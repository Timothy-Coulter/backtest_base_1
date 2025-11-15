# Unit Test Coverage Plan

1. **Core engine**
   - Expand `tests/core/test_backtest_engine.py` (or add `tests/core/test_backtest_engine_runtime.py`) with targeted cases for `load_data`, `_process_signals_and_update_portfolio`, `_check_risk_management`, and `_calculate_performance_metrics`, using fakes/mocks to cover success/failure/fallback branches.
2. **Event bus & handlers**
   - Create `tests/core/test_event_bus.py` covering subscribe/unsubscribe/publish, priority ordering, reentrancy, and error handling.
   - Add per-handler tests (`tests/core/test_event_handlers.py`) for `MarketDataHandler`, `SignalHandler`, etc., feeding representative `Event` instances.
3. **Portfolio implementations**
   - Extend `tests/portfolio/test_general_portfolio.py` and `tests/portfolio/test_dual_pool.py` with scenarios that exercise `process_tick`, financing costs, stop-loss / take-profit triggers, and trade logging.
4. **Risk components**
   - Strengthen `tests/risk_management/*` (or add new files) covering `RiskControlManager`, `RiskLimits`, `PositionSizer`, and config validation, including limit breaches and invalid config data.
5. **Utilities**
   - Add edge-case coverage in `tests/utils/test_cache_utils.py`, `test_string_utils.py`, etc., verifying behaviour under filesystem errors, locale variations, and invalid inputs.
