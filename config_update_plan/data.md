# backtester/data

## Observations
- `DataRetrieval` constructor already accepts `DataRetrievalConfig` but mutates it (API keys) and caches globally without ConfigProcessor awareness.
- Tests/config overrides rely on dict updates rather than serialized configs, making YAML ingestion error prone.

## Plan
1. **Immutable config handling**: update `DataRetrieval` to accept either a `DataRetrievalConfig`, YAML path, or kwargs and immediately resolve them through `ConfigProcessor`. Store only frozen copies to avoid mutating caller configs when injecting API keys.
2. **Cache key normalization**: move `_build_cache_key` into a helper that receives the config dict produced by ConfigProcessor so YAML-sourced tickers map identically to programmatic ones.
3. **Validation hooks**: ensure `DataRetrievalConfig` exposes `@classmethod default()` and `model_validate` to guarantee tickers, date ranges, and API keys are sanitized before any call to MarketDataRequest.
4. **CLI + YAML parity**: document that the `--freq`, `--start-date`, etc. CLI switches map to `DataRetrievalConfig` fields; ConfigProcessor should be the only entryway for these overrides so YAML and CLI produce the same config.
