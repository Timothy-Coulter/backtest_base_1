"""Utilities for comparing configuration snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backtester.core.config import BacktesterConfig


@dataclass(frozen=True, slots=True)
class ConfigDelta:
    """Represents a single field difference between two configurations."""

    path: str
    before: Any
    after: Any


def diff_configs(base: BacktesterConfig, updated: BacktesterConfig) -> list[ConfigDelta]:
    """Return a flat list of differences between two BacktesterConfig snapshots."""
    baseline = base.model_dump(mode="python")
    candidate = updated.model_dump(mode="python")
    deltas: list[ConfigDelta] = []
    _collect_diffs(path="", left=baseline, right=candidate, deltas=deltas)
    return deltas


def format_config_diff(deltas: Sequence[ConfigDelta]) -> str:
    """Render a configuration diff in a log-friendly format."""
    return "\n".join(f"- {delta.path}: {delta.before!r} -> {delta.after!r}" for delta in deltas)


def _collect_diffs(
    *,
    path: str,
    left: Any,
    right: Any,
    deltas: list[ConfigDelta],
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        all_keys = set(left.keys()) | set(right.keys())
        for key in sorted(all_keys):
            next_path = f"{path}.{key}" if path else key
            if key not in left:
                deltas.append(ConfigDelta(next_path, None, right[key]))
                continue
            if key not in right:
                deltas.append(ConfigDelta(next_path, left[key], None))
                continue
            _collect_diffs(path=next_path, left=left[key], right=right[key], deltas=deltas)
        return

    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            deltas.append(ConfigDelta(path or "[root]", left, right))
        return

    if left != right:
        deltas.append(ConfigDelta(path or "[root]", left, right))
