# Package API

## `backtester.core` exports undefined symbols
- **Where:** `backtester/core/__init__.py:57-114`
- **Issue:** `__all__` lists `RiskConfig`, `BacktestConfig`, and `ConfigValidator`, but none of these names are imported or defined in the module. Attempting `from backtester.core import RiskConfig` raises `ImportError`. This breaks downstream code that relies on the package’s documented surface area.
- **Fix:** Either import the correct classes (e.g., introduce dedicated config validator objects) or drop the unused names from `__all__`. Add a thin smoke test that `dir(backtester.core)` exposes the advertised symbols.
