# Next Steps After Configuration Management

## 1. Wire CLI and scripting overrides into `BacktestRunConfig`
- Extend `main.py` (and any automation scripts under `scripts/`) to parse CLI / env arguments for tickers, date ranges, and strategy knobs, then feed them into `BacktestRunConfig.with_*_overrides`.
- Add guardrails so mutually exclusive arguments (e.g., `start_date` vs. presets like `year`) surface clear errors before the engine starts.
- Document the new flags in `README.md` and add an integration test (e.g., `tests/integration/test_cli_entrypoint.py`) that exercises `uv run python main.py --ticker AAPL`.

## 2. Provide safer config snapshots to downstream components
- Update constructors for `GeneralPortfolio`, `SimulatedBroker`, `RiskControlManager`, and strategy orchestrators so they accept dataclass- or Pydantic-derived read-only views rather than the mutable `BacktesterConfig`.
- Introduce helper functions in `backtester/core/config.py` that return shallow dataclasses (`DataConfigView`, `PortfolioConfigView`, etc.) and replace direct `config.*` attribute access inside the engine with these frozen views.
- Add regression tests in `tests/core/test_backtest_engine.py` ensuring attempts to mutate a view raise `AttributeError`, and cover helper creation in `tests/core/test_config.py`.

## 3. Enhance data caching and invalidation
- Move the in-memory cache introduced in `backtester/data/data_retrieval.py` into `backtester/utils/cache_utils.py` so other modules (e.g., indicators) can reuse it.
- Support TTL and max-size eviction policies; expose tuning knobs on `DataRetrievalConfig` (`cache_ttl_seconds`, `cache_max_entries`) with validation.
- Expand `tests/data/test_data_retrieval.py` to verify eviction and TTL behaviour, and add stress tests that simulate concurrent `get_data` calls to ensure thread safety.

## 4. Build config-diff tooling for reproducibility
- Create `backtester/core/config_diff.py` with helpers that compare two `BacktesterConfig` snapshots and emit human-readable diffs (identify which component/fields changed).
- Surface the diff in `BacktestEngine.run_backtest` logs at INFO level when overrides are applied, so run logs capture the exact configuration.
- Add unit tests in `tests/core/test_config_diff.py` covering nested diff cases, and integration coverage that asserts the engine logs diffs when `ticker` overrides are passed.

## 5. Expand documentation and onboarding artifacts
- Grow `docs/configuration.md` with concrete walkthroughs (CLI overrides, programmatic builder usage) and link it from `README.md`.
- Add a runnable notebook or markdown tutorial (`docs/run_config_tutorial.md`) that shows building multiple `BacktestRunConfig` snapshots for parameter sweeps.
- Ensure `docs/` pages are enforced via CI by adding a markdown linter or link checker step (e.g., `scripts/check_docs.py`) and documenting the workflow in `CONTRIBUTING.md`.
