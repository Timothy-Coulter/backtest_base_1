# backtester/portfolio

## Observations
- `GeneralPortfolio`, `DualPoolPortfolio`, and `BasePortfolio` accept long lists of primitive kwargs and optionally a `PortfolioConfigView`; they do not expose default configs themselves.
- Risk manager wiring (`risk_manager` parameter) is optional and set post-init, making it hard to reproduce from YAML.

## Plan
1. **Constructor contract**: require each portfolio implementation to accept a single `PortfolioConfig` (or subclass) plus optional `*, overrides: dict`. Provide `@classmethod default_config()` returning `PortfolioConfig` or richer models for dual‑pool variants. Deprecate primitive kwargs.
2. **Config-driven state**: move derived attributes (max positions, leverage, etc.) into the config via ConfigProcessor so constructors simply store the config and maybe compute read-only views.
3. **Risk integration**: define `PortfolioRuntimeConfig` that bundles `PortfolioConfig` + references to `RiskControlManager` + `ExecutionConfig`. ConfigProcessor should produce it and feed both the portfolio and broker.
4. **YAML parity**: document sample configs in `component_configs/portfolio/` (general, dual_pool, etc.) to ensure smoke tests can instantiate each portfolio from files only.
