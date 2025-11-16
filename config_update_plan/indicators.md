# backtester/indicators

## Observations
- All indicator classes derive from `BaseIndicator` and already accept `IndicatorConfig`, but defaults are enforced in constructors rather than centralized.
- Factory registration uses strings; there is no YAML schema describing indicator stacks used by strategies.

## Plan
1. **Default config methods**: add `@classmethod default_config(cls) -> IndicatorConfig` to each indicator so ConfigProcessor can construct them without manual period/threshold adjustments inside `__init__`.
2. **Config normalization**: move indicator-specific overrides (e.g., forcing `indicator_type`) into ConfigProcessor validators so constructors become side-effect free.
3. **YAML-driven indicator suites**: define canonical YAML fragments under `component_configs/indicators/<indicator>.yaml` describing typical parameters. Strategies can reference them by name when building their configs.
4. **Validation**: update `IndicatorConfig` to expose `model_config['json_schema_extra']` or docstrings that map to YAML keys, enabling linting before indicators are instantiated.
