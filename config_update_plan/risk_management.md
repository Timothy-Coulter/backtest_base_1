# backtester/risk_management

## Observations
- Component configs already live under `component_configs`, but `RiskControlManager` mutates them and also accepts `RiskConfigView`, creating duplication.
- Stop loss / take profit / position sizing default configs are instantiated inside `ComprehensiveRiskConfig.__init__`, preventing ConfigProcessor from knowing whether a user supplied overrides.

## Plan
1. **Explicit defaults**: expose `StopLossConfig.default()`, etc. and let ConfigProcessor inject them only when absent. Remove implicit instantiation inside `ComprehensiveRiskConfig.__init__` so we can tell when YAML overrides a component.
2. **Single source of truth**: drop `RiskConfigView` in favor of handing the actual `ComprehensiveRiskConfig` to `RiskControlManager`. Provide `@classmethod from_source(cls, source)` helper that defers to ConfigProcessor for YAML/dict inputs.
3. **Validation improvements**: encode cross-field validations (e.g., `max_daily_loss < max_drawdown`) inside config validators instead of runtime warnings.
4. **YAML samples & smoke tests**: maintain curated configs in `component_configs/risk_management/...` and ensure smoke tests load them via ConfigProcessor to instantiate each sub-component directly.
