# Separation of Concerns Plan

1. **Define explicit interfaces**
   - Introduce `Protocol`/ABC definitions in a new module (e.g., `backtester/core/interfaces.py`) for `Strategy`, `Portfolio`, `Broker`, and `RiskManager`, capturing the minimal cross-component surface area.
   - Refactor `backtester/core/backtest_engine.py`, `backtester/strategy/*`, `backtester/portfolio/*`, and `backtester/execution/broker.py` to accept these interfaces via constructor or dependency injection.
2. **Route risk decisions through the risk manager**
   - Remove ad-hoc risk checks from strategies and portfolios, replacing them with helper methods on `RiskControlManager` (`can_open`, `record_order`, `record_fill`); update callers in `backtester/strategy/*` and `backtester/portfolio/*`.
   - Ensure `backtester/risk_management/risk_control_manager.py` owns all risk computation and emits appropriate events/logs.
3. **Keep strategies free of execution details**
   - Update `backtester/strategy/signal/base_signal_strategy.py` to publish only `SignalEvent`s and stop mutating portfolio/broker state directly.
   - Move order creation logic into `backtester/execution/broker.py` (or a dedicated `OrderService`) and ensure portfolios react only to fills.
4. **Enforce boundaries with tests**
   - Add unit tests under `tests/core`/`tests/strategy`/`tests/portfolio` that inject mocks for adjacent interfaces and assert no forbidden method calls occur (e.g., strategy tries to call broker methods directly).
