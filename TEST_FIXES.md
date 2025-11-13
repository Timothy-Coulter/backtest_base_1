# Test Failure Recovery Plan

## Context
- Command `uv run pytest` fails during test collection because the module `findatapy` is not installed.
- The failure occurs while importing `backtester.data.data_retrieval`, which unconditionally imports several classes from `findatapy`. That module is pulled in via `backtester.__init__` when tests import components such as `backtester.core.event_bus`.

## Plan
1. **Ensure `findatapy` Dependency Is Available** ✅  
   - Install `findatapy` via `uv pip install findatapy` (or verify it is already present) so imports succeed during test discovery.
2. **Re-run the Test Suite** ✅  
   - Execute `uv run pytest` to capture the current failure surface now that the dependency is in place.
3. **Iterate if Needed** ✅  
   - Based on the updated failure output, extend this plan with targeted fixes and implement them.  
   - Outcome: Adjusted signal strategy info reporting to satisfy mixed-case expectations and introduced case-insensitive `StrategyTypeLabel`. Final test run: `1457 passed, 6 skipped`.
