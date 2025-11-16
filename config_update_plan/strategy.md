# backtester/strategy

## Observations
- Strategy packages (signal/orchestration/portfolio) hand-roll config merging: e.g., signal strategies accept a specialized config but BacktestEngine rebuilds them from `StrategyConfig` rather than letting each strategy own its defaults.
- Orchestrators rely on a default `OrchestrationConfig` baked into the engine; there is no YAML-defined pipeline of strategies.

## Plan
1. **Per-strategy defaults**: add `default_config()` to each strategy class (signal & portfolio). Provide conversion helpers so a high-level `StrategyConfig` from BacktesterConfig can be transformed into the concrete strategy config via ConfigProcessor, not `BacktestEngine` methods.
2. **Strategy registry**: expose a `StrategyFactory` that accepts `strategy_name` + config source (model/dict/YAML) and returns the appropriate instance. The factory should look up YAML fragments stored under `component_configs/strategy/<type>/<name>.yaml`.
3. **Orchestration pipeline**: allow `OrchestrationConfig` to be loaded from YAML (list of strategies, dependencies, conflict resolution). ConfigProcessor should validate references (strategy IDs exist, dependencies valid) before orchestration begins.
4. **Config-only initialization**: ensure orchestrators accept `OrchestrationConfig` as their sole required argument; event bus, logger, etc. become optional kwargs. Remove fallback defaults built inside `BacktestEngine` so YAML definitions drive orchestration structure.
