# Event Bus Integration Roadmap

1. **Define canonical event payloads**
   - Update `backtester/core/events.py` so `create_market_data_event` injects `symbol`, `symbols`, bar metadata, and provenance fields into `metadata`.
   - Add developer documentation (either inline or in `README.md`) describing the guaranteed keys every event publishes.
2. **Fix subscription plumbing**
   - Refactor `backtester/core/event_bus.py` to store `(subscription_id, filter, handler)` tuples and to compare IDs exactly when unsubscribing; add type hints and logging for diagnostics.
   - Align `backtester/strategy/signal/base_signal_strategy.py` (and any other subscriber) with the new metadata contract, supporting wildcard/all-symbol subscriptions where needed.
3. **Ensure every component communicates via the bus**
   - Emit events from the core components (`SignalEvent` inside `base_signal_strategy`, `OrderEvent` inside `backtester/execution/broker.py`, `PortfolioUpdateEvent` in `backtester/portfolio/*`, `RiskAlertEvent` in `backtester/risk_management/risk_control_manager.py`).
   - Rework `BacktestEngine._run_simulation` to publish/consume events instead of calling components directly, wiring handlers through `backtester/core/event_handlers.py`.
4. **Introduce orchestration smoke tests**
   - Add unit tests under `tests/core/test_event_bus.py` that cover subscribe → publish → unsubscribe flows and event fan-out ordering.
   - Extend `tests/integration/test_event_driven_engine.py` (or add a new integration file) to validate the end-to-end publish/subscribe path without relying on fallback synchronous calls.
