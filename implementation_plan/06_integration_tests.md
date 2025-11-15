# Integration Test Plan

1. **Event-driven backtest smoke test**
   - Extend `tests/integration/test_event_driven_engine.py` (or add `test_full_pipeline.py`) to feed a small OHLCV frame through the entire engine, asserting that `MarketDataEvent`, `SignalEvent`, `OrderEvent`, `PortfolioUpdateEvent`, and `RiskAlertEvent` all appear on the bus.
2. **Real data retrieval path**
   - Add an integration test (e.g., `tests/integration/test_data_retrieval_real.py`) that fetches cached SPY data via `DataRetrieval.get_data()`, validating cache-first logic and environment variable handling (optionally using VCR or pre-baked fixtures to avoid live calls).
3. **Risk breach scenario**
   - Create a scenario test (`tests/integration/test_risk_breach.py`) that intentionally violates drawdown/leverage limits and asserts the engine cancels orders, emits risk alerts, and halts if configured.
4. **Multi-strategy orchestration**
   - Write an integration test spinning up multiple strategies under `backtester/strategy/orchestration/*`, verifying dependency ordering, conflict resolution, and aggregated performance.
5. **Kelly portfolio workflow**
   - Add an end-to-end test for the Kelly strategy ensuring `portfolio_strategy` target weights propagate to `GeneralPortfolio`/`DualPoolPortfolio` and that base/alpha metrics are reported; reuse realistic fixtures to keep runtime manageable.
