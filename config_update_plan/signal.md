# backtester/signal

## Observations
- Module currently only defines `SignalType`/`SignalGenerator`; no config yet but strategies rely on shape of signal dicts.

## Plan
1. **Schema definition**: add a `SignalSchema` (pydantic model) describing required keys so ConfigProcessor can validate signal definitions in YAML (e.g., when specifying canned signals for tests).
2. **YAML templates**: provide basic signal templates under `component_configs/signal/` (buy/hold/sell) that smoke tests can load to ensure formatting/validation works.
