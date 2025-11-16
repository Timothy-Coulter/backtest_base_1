# backtester/optmisation

## Issues
- Parameter spaces and optimization configs are defined programmatically; there is no YAML-driven way to describe studies.
- `OptimizationRunner` mixes `OptimizationConfig` dataclass values with ad hoc kwargs, and the objective builds BacktestEngine overrides manually.

## Plan
1. **YAML for studies**: define schema for `OptimizationConfig` + `ParameterSpace` under `component_configs/optmisation/<study>.yaml`. ConfigProcessor should load these to produce both the study metadata and per-parameter definitions.
2. **Config-driven objective**: allow `OptimizationObjective` to accept a `BacktesterConfig` (or YAML path) plus a list of strategy/portfolio overrides. Use ConfigProcessor to produce the config snapshot for each trial instead of manually editing dicts.
3. **Result serialization**: include the resolved config (or pointer to YAML + override diff) inside `OptimizationResult.study_summary` so runs are reproducible.
4. **Validation**: extend `ParameterDefinition` to run `ConfigProcessor.validate_schema("optmisation.parameter")` ensuring YAML-sourced ranges are consistent (low < high, etc.).
