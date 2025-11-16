# backtester/model

## Current state
- `BaseModel` expects a config object but can also ingest arbitrary `config.data_config` dicts, leading to mutation of caller state.
- `ModelFactory` registers adapters but there is no single place to parse `.yaml` definitions of models.

## Plan
1. **ConfigProcessor integration**: allow `ModelFactory.create_from_config_dict` to accept YAML paths/dicts, passing them through ConfigProcessor which resolves to `ModelConfig` (or subclass) before any adapter is touched.
2. **Default config exposure**: each concrete model (e.g., `SklearnModel`) should implement `@classmethod default_config()` returning the appropriate `ModelConfig` subclass, used by smoke tests and sample YAMLs.
3. **Adapter standardization**: require adapters to expose `@classmethod default_training_config()` so YAML files can override training hyperparams without editing code.
4. **Validation**: ensure `BaseModel._validate_configuration` leverages ConfigProcessor errors to surface invalid YAML early; remove ad hoc checks in derived models.
