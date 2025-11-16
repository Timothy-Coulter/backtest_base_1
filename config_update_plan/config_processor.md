# ConfigProcessor plan

## Goals
- Provide a single entry point for loading, merging, and validating component configs regardless of source (model instance, dict, kwargs, YAML path).
- Emit fully typed `BacktesterConfig` snapshots plus component-specific models without mutating caller inputs.

## Proposed API
```python
processor = ConfigProcessor(base=BacktesterConfig.default())
resolved = processor.apply(
    source=config_or_path_or_mapping,
    component_overrides={
        "data": {"tickers": ["AAPL", "MSFT"]},
        "strategy": StrategyConfig(...),
        "portfolio": "component_configs/portfolio/general.yaml",
    },
)
```
- `apply` returns a new `BacktesterConfig` (or specific component when `component="data"` is provided).
- Additional helpers: `load_yaml(path)`, `merge_models(base_model, overrides)`, `validate(config)`.

## Implementation steps
1. **Data loading layer**: support sources of type `BacktesterConfig | BaseModel | Mapping | PathLike | str | None`. YAML files should be read with `yaml.safe_load`, then validated against the target model schema.
2. **Default management**: accept an optional `base` config in `__init__`. If omitted, call `BacktesterConfig()` (which already populates defaults). Keep an immutable copy for diffing & resets.
3. **Component overrides**: allow `.apply_component("strategy", overrides)` that merges overrides into `StrategyConfig`. Under the hood, use Pydantic's `model_copy(update=...)` while ensuring nested configs are re-instantiated rather than mutated.
4. **Validation**: run `BacktesterConfig.model_validate` (or `validate_run_config`) on every resulting config. Capture errors and raise a custom `ConfigValidationError` with context (component name, YAML path).
5. **Diff + logging integration**: expose `diff_with_defaults(resolved)` returning `ConfigDelta` list from `core.config_diff`. BacktestEngine can log this after ConfigProcessor runs.
6. **Thread safety**: make ConfigProcessor stateless per call (no shared mutable overrides) so CLI/integration tests can reuse a singleton without risk.
7. **Testing**: add unit tests covering input sources (pydantic model, dict, YAML), override precedence (kwargs > YAML > base), and validation failure messages.
