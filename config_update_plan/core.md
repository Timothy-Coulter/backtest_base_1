# backtester/core

## Issues to fix
- `BacktestEngine` manually clones configs and builds strategy/portfolio configs with helpers (`_build_momentum_config`, `_build_portfolio_strategy_config`) instead of delegating to dedicated component factories.
- Components (`DataRetrieval`, `GeneralPortfolio`, `RiskControlManager`, `SimulatedBroker`, `PerformanceAnalyzer`) accept a mix of config views, primitive kwargs, or optional config objects which makes YAML injection brittle.
- There is no single entry point for loading `.yaml` / dict / model inputs into `BacktesterConfig`; `get_config()` hides global state.

## Required changes
1. **ConfigProcessor adoption**
   - Replace calls to `BacktestRunConfig(...).build()` with `ConfigProcessor(base_config=self._global_defaults).apply(...)` so every run starts from a normalized config snapshot.
   - Provide `BacktestEngine.__init__(config_source: BacktesterConfig | str | Path | Mapping | None)` to allow YAML paths.
2. **Component initialization contracts**
   - Refactor `self.data_handler = DataRetrieval(self.config.data)` etc. so each component receives its own Config model (never views) and exposes `@classmethod default_config()` returning the Config it expects. The engine should simply pass `config.data`, `config.execution`, `config.risk`, etc. without building ad‑hoc views.
   - Move `_build_momentum_config` and `_build_portfolio_strategy_config` into `backtester/strategy` factories that accept `StrategyConfig` / `PortfolioStrategyConfig`. Engine should call `ConfigProcessor` to merge overrides into those configs, then instantiate the strategy.
3. **Config diffing/logging**
   - Update `config_diff` utilities to accept either models or raw YAML so diffs can be produced between default config and any YAML-sourced config.
4. **Event handler wiring**
   - When engine instantiates event bus / orchestrator / handlers, pass explicit config objects (e.g., `OrchestrationConfig`) rather than implicit defaults; expose defaults via `@classmethod default_config()`.
5. **Validation lifecycle**
   - Centralize `validate_run_config` inside ConfigProcessor. Deprecate manual `assert self.config.data is not None` checks by surfacing validation errors before engine initialization.
