# backtester/cli

## Current pain points
- CLI overrides only produce nested dicts that are later merged into `BacktestRunConfig`; there is no standard interface for loading `.yaml` bundles or per-component configs.
- The CLI layer knows about data/strategy/portfolio keys but not the Config models themselves, so defaults/validation are recreated in `BacktestEngine` helpers.

## Streamlined config plan
1. **Introduce ConfigProcessor usage**: accept either an in-memory `BacktesterConfig`, a path to `.yaml`, or CLI-provided overrides. Provide a top-level `--config` flag plus `BACKTEST_CONFIG_PATH` env var that hands off to `ConfigProcessor` so CLI never manipulates raw dicts.
2. **Typed overrides**: update `CLIOverrides` to carry `DataRetrievalConfig`, `StrategyConfig`, etc. Instead of dicts, call `ConfigProcessor.apply(component="strategy", overrides=strategy_kwargs)` so downstream always receives a model instance.
3. **Default config file discovery**: allow `runtime.build_run_config_from_cli` to accept `config_source: BacktesterConfig | str | Path | Mapping` and run it through ConfigProcessor, ensuring YAML > CLI > env precedence.
4. **Validation hooks**: after merging CLI inputs, invoke `BacktesterConfig.model_validate` (via ConfigProcessor) before returning so `BacktestEngine` can assume a fully validated snapshot.
5. **Docs / examples**: update CLI help strings to mention `.yaml` usage and reference new config folders under `component_configs/` for quick smoke setups.
