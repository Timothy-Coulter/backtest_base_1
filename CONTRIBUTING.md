# Contributing Guidelines

Thanks for helping improve QuantBench! A few expectations keep the codebase tidy and reproducible.

## 1. Configuration & Reproducibility

- Use `BacktestRunConfig` for every override (CLI, notebooks, or scripts). Never mutate the global
  config (`backtester.core.config.get_config()`) in place.
- When adding new command-line flags or scripting helpers, wire them through the builder so the
  config diff tooling continues to track changes automatically.

## 2. Testing & Tooling

Before opening a PR run:

```bash
uv run ruff format . && uv run black . && uv run isort .
uv run ruff check --fix .
uv run mypy .
uv run python scripts/check_docs.py
uv run pytest
```

The `scripts/check_docs.py` helper ensures Markdown in `docs/` and the root README stay healthy
(missing headers, dead links, trailing whitespace, etc.). CI will run the same checks.
To run the entire suite locally in one shot, execute `uv run python scripts/run_ci_checks.py`
(`scripts\run_ci_checks.py` on Windows), which chains all of the above commands.

## 3. Documentation

- New features touching configuration, CLI options, or runtime behaviour must be documented in
  `docs/configuration.md` (or the relevant page) and linked from `README.md`.
- Use `docs/run_config_tutorial.md` as the canonical place to describe complex BacktestRunConfig
  workflows (parameter sweeps, automation).

## 4. Coding Style & Comments

- Prefer small, focused functions with docstrings.
- Configuration passed to downstream components should be read-only via the helper views in
  `backtester.core.config`.
- Inline comments are welcome when behaviour is not obvious, but avoid restating code literally.
