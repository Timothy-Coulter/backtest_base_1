# backtester/execution

## Issues
- `SimulatedBroker` accepts a mix of `SimulatedBrokerConfig`, `ExecutionConfigView`, and primitive kwargs, leading to multiple code paths and partial config application.
- `OrderManager` / `Order` classes rely on loose kwargs for commission/latency, no typed config.

## Plan
1. **Single config entrypoint**: expose `SimulatedBroker.default_config() -> SimulatedBrokerConfig` and require callers to pass either that model or a YAML path handled via ConfigProcessor. Remove direct kwargs overrides, except optional runtime kwargs forwarded to ConfigProcessor.
2. **ExecutionConfigView removal**: replace `build_execution_config_view` usage with direct access to the config model; views can become lightweight dataclasses produced by ConfigProcessor (if immutability is desired) instead of ad hoc clones.
3. **Order settings**: extend `SimulatedBrokerConfig` to include slippage distribution, latency, and order throttling; share it with `OrderManager` so commission/minimums remain consistent regardless of how orders are created.
4. **YAML schema**: document YAML structure for execution config (under `component_configs/execution/`) so smoke tests can spin up brokers purely from files.
