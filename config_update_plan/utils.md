# backtester/utils

## Observations
- Utility modules are stateless but some (e.g., `CacheUtils`, `TimeUtils`) accept kwargs that should be configurable globally (cache TTLs, default locale, DB creds).
- There is no config bridging between `.env` files and config models.

## Plan
1. **Utility config models**: introduce lightweight configs (e.g., `CacheConfig`, `DBConfig`, `FormattingConfig`) with `default()` methods so ConfigProcessor can load YAML snippets and inject them where needed (DataRetrieval cache, db_manager, formatters).
2. **Env bridging**: update `db_manager.get_db_connection` to accept a `DBConfig` produced by ConfigProcessor. Provide YAML templates under `component_configs/utils/db.yaml` capturing default host/user/password to keep secrets out of code.
3. **Global settings**: allow `ConfigProcessor` to register module-level settings (e.g., `FormatUtils.default_locale`). Provide a `settings.yaml` sample consumed at startup.
