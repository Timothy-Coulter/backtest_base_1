# Backtest Engine Workflow Plan

1. **Document the canonical loop**
   - Add a dedicated section to `README.md` (or `docs/engine_workflow.md`) describing the sequence: data fetch ➝ market data event ➝ strategy signal ➝ risk evaluation ➝ order placement ➝ broker fill ➝ portfolio update ➝ performance logging.
   - Include a diagram (PlantUML / Mermaid) checked into `docs/` so contributors have a visual reference.
2. **Refactor `_run_simulation` around the flow**
   - Rework `backtester/core/backtest_engine.py` so `_run_simulation` uses the event bus exclusively, emitting events and letting handlers react in order; eliminate direct invocations that bypass the bus.
   - Capture state transitions (trades, fills, risk alerts) in structured logs or data classes, writing them to `self.trade_history` / `self.performance_metrics`.
3. **Add lifecycle hooks**
   - Introduce hook interfaces (e.g., `before_run`, `before_tick`, `after_tick`, `after_run`) in `BaseSignalStrategy`, `GeneralPortfolio`, and `SimulatedBroker`, and have `BacktestEngine` invoke them at the right times.
   - Ensure hooks can be optionally implemented (use `hasattr` or abstract base classes) so existing strategies compile.
4. **Verify with scenario tests**
   - Add table-driven tests under `tests/core/test_backtest_engine.py` or a new `tests/core/test_engine_workflow.py` that assert the ordered sequence of events for single-asset and multi-asset runs.
   - Extend `tests/integration/test_event_driven_engine.py` to cover at least one multi-strategy workflow, validating that every lifecycle stage fires exactly once per tick.
