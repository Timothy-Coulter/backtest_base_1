# Configuration Management Plan

1. **Centralise runtime overrides**
   - Extend `backtester/core/config.py` with a `BacktestRunConfig` (or builder helpers on `BacktesterConfig`) that accepts per-run overrides without mutating the module-level singleton.
   - Provide helper methods such as `with_data_overrides`, `with_strategy_overrides`, etc., and ensure `BacktestEngine` (and CLI entry points in `main.py`) consume the merged snapshot.
2. **Ensure DataRetrieval honours overrides**
   - Add an API on `backtester/data/data_retrieval.py` (e.g., `get_data(config_override: DataRetrievalConfig | None)`) to allow temporary parameter swaps, and update `BacktestEngine.load_data` accordingly.
   - Implement a keyed cache (possibly inside `DataRetrieval` or a new utility module) keyed by `(data_source, ticker_list, start, end, interval)` so repeated runs reuse frames safely; add coverage in `tests/data/test_data_retrieval.py`.
3. **Validate configurations early**
   - Use Pydantic validators or a dedicated `validate_run_config` helper in `core/config.py` to catch invalid combinations (end < start, missing tickers, negative leverage) before components initialise, and add regression tests in `tests/core/test_config.py`.
4. **Document configuration ownership**
   - Update the README or a new `docs/configuration.md` to explain which component consumes which part of the config (data vs. strategy vs. risk) and expose immutable views to prevent cross-component mutation.
