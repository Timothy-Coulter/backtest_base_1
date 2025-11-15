# Event System

## Strategy subscriptions never match market data events
- **Where:** `backtester/strategy/signal/base_signal_strategy.py:96-108` and `backtester/core/events.py:470-495`
- **Issue:** Signal strategies subscribe with `EventFilter(metadata_filters={'symbols': self.config.symbols})`, but `create_market_data_event` never sets a `symbols` metadata key (it only stores the single-event payload). The filter therefore rejects every published `MARKET_DATA` event and the orchestrator falls back to calling strategies synchronously, defeating the purpose of the event bus.
- **Fix:** Either include a `symbols` key (or similar) when constructing `MarketDataEvent.metadata`, or change the filter to match on the existing keys (`symbol`, `data_type`, etc.). Add an integration test that proves a subscribed strategy receives the event without relying on the fallback path.

## `unsubscribe` cannot remove subscriptions
- **Where:** `backtester/core/event_bus.py:134-186`
- **Issue:** Subscription IDs include the monotonically increasing `_next_event_id`, but `unsubscribe` compares the caller-supplied ID against strings built with the *current list index* (`sub_{id(handler)}_{i}`) and even uses `in` instead of `==`. As soon as another handler is added or removed, the indices diverge and `unsubscribe` never finds a match, so handlers leak forever.
- **Fix:** Store the subscription IDs alongside the handlers (e.g. keep a list of `(subscription_id, filter, handler)` tuples) and perform exact equality checks. Add a unit test that subscribes, unsubscribes, and asserts the handler no longer fires.
