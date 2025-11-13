"""Event Bus System for decoupled component communication.

This module provides the central event bus that enables event-driven architecture
through publish/subscribe pattern implementation.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from backtester.core.logger import get_backtester_logger

# Type variables for generic event handling
T = TypeVar('T')
EventT = TypeVar('EventT', bound='Event')


class EventPriority(Enum):
    """Event priority levels for processing order."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Event:
    """Base event class for all system events."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ):
        """Initialize event."""
        self.event_type = event_type
        self.timestamp = timestamp
        self.source = source
        self.priority = priority
        self.metadata = metadata or {}
        self.event_id = event_id or f"event_{id(object())}"

        # Validate event data
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        if not self.source:
            raise ValueError("Event source cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'source': self.source,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'event_id': self.event_id,
        }


@dataclass
class EventFilter:
    """Filter criteria for event subscription."""

    event_types: set[str] | None = None
    sources: set[str] | None = None
    priority_min: EventPriority | None = None
    priority_max: EventPriority | None = None
    metadata_filters: dict[str, Any] | None = None

    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check source
        if self.sources and event.source not in self.sources:
            return False

        # Check priority range
        if self.priority_min and event.priority.value < self.priority_min.value:
            return False
        if self.priority_max and event.priority.value > self.priority_max.value:
            return False

        # Check metadata filters
        if self.metadata_filters:
            for key, value in self.metadata_filters.items():
                if key not in event.metadata or event.metadata[key] != value:
                    return False

        return True


class EventBus:
    """Central event bus for decoupled component communication."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the event bus.

        Args:
            logger: Logger instance for event bus operations
        """
        self.logger = logger or get_backtester_logger(__name__)

        # Subscription management
        self._subscriptions: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._filtered_subscriptions: list[tuple[EventFilter, Callable[[Event], None]]] = []

        # Event processing
        self._event_queue: list[Event] = []
        self._processing: bool = False
        self._event_handlers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)

        # Metrics
        self._published_events: int = 0
        self._processed_events: int = 0
        self._dropped_events: int = 0

        # Event ID tracking
        self._next_event_id: int = 0

        self.logger.info("Event bus initialized")

    def subscribe(
        self, handler: Callable[[Event], None], event_filter: EventFilter | None = None
    ) -> str:
        """Subscribe to events with optional filtering.

        Args:
            handler: Event handler function
            event_filter: Optional filter criteria

        Returns:
            Subscription ID for later unsubscribing
        """
        if not callable(handler):
            raise ValueError("Handler must be callable")

        subscription_id = f"sub_{id(handler)}_{self._next_event_id}"
        self._next_event_id += 1

        if event_filter:
            self._filtered_subscriptions.append((event_filter, handler))
            self.logger.debug(f"Added filtered subscription {subscription_id}")
        else:
            # Default to all events if no filter specified
            default_filter = EventFilter()
            self._filtered_subscriptions.append((default_filter, handler))
            self.logger.debug(f"Added default subscription {subscription_id}")

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was found and removed, False otherwise
        """
        removed = False

        # Remove from filtered subscriptions
        for i, (_, handler) in enumerate(self._filtered_subscriptions):
            if subscription_id in f"sub_{id(handler)}_{i}":
                del self._filtered_subscriptions[i]
                removed = True
                break

        if removed:
            self.logger.debug(f"Removed subscription {subscription_id}")
        else:
            self.logger.warning(f"Subscription {subscription_id} not found")

        return removed

    def publish(self, event: Event, immediate: bool = False) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
            immediate: If True, process event immediately, otherwise queue it
        """
        if not isinstance(event, Event):
            raise ValueError("Event must be an instance of Event class")

        # Update event metrics
        self._published_events += 1

        if immediate:
            self._process_event(event)
        else:
            self._event_queue.append(event)
            self.logger.debug(f"Queued event {event.event_id} of type {event.event_type}")

        # Process queued events if not already processing
        if not self._processing and self._event_queue:
            self._process_queued_events()

    async def publish_async(self, event: Event) -> None:
        """Publish an event asynchronously.

        Args:
            event: Event to publish
        """
        if not isinstance(event, Event):
            raise ValueError("Event must be an instance of Event class")

        self._published_events += 1
        self._event_queue.append(event)

        # Process in background if not already processing
        if not self._processing:
            asyncio.create_task(self._process_queued_events_async())

    def register_handler(self, event_type: str, handler: Callable[[Event], None]) -> str:
        """Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Handler function

        Returns:
            Handler ID for later deregistration
        """
        if not callable(handler):
            raise ValueError("Handler must be callable")

        handler_id = f"handler_{id(handler)}_{self._next_event_id}"
        self._next_event_id += 1

        self._event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler {handler_id} for event type {event_type}")

        return handler_id

    def deregister_handler(self, event_type: str, handler_id: str) -> bool:
        """Deregister a handler for a specific event type.

        Args:
            event_type: Type of event
            handler_id: Handler ID to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if event_type not in self._event_handlers:
            return False

        for i, handler in enumerate(self._event_handlers[event_type]):
            if handler_id == f"handler_{id(handler)}_{i}":
                del self._event_handlers[event_type][i]
                self.logger.debug(f"Deregistered handler {handler_id} for event type {event_type}")
                return True

        return False

    def _process_event(self, event: Event) -> None:
        """Process a single event through all subscribers."""
        try:
            # Process filtered subscriptions
            for event_filter, handler in self._filtered_subscriptions:
                if event_filter.matches(event):
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(
                            f"Error in filtered handler for event {event.event_id}: {e}"
                        )

            # Process type-specific handlers
            if event.event_type in self._event_handlers:
                for handler in self._event_handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(
                            f"Error in type-specific handler for event {event.event_id}: {e}"
                        )

            self._processed_events += 1
            self.logger.debug(f"Processed event {event.event_id} of type {event.event_type}")

        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {e}")
            self._dropped_events += 1

    def _process_queued_events(self) -> None:
        """Process all queued events synchronously."""
        self._processing = True

        while self._event_queue:
            event = self._event_queue.pop(0)
            self._process_event(event)

        self._processing = False

    async def _process_queued_events_async(self) -> None:
        """Process all queued events asynchronously."""
        self._processing = True

        while self._event_queue:
            event = self._event_queue.pop(0)
            self._process_event(event)
            # Small delay to prevent blocking
            await asyncio.sleep(0.001)

        self._processing = False

    def clear_queue(self) -> int:
        """Clear all queued events.

        Returns:
            Number of events cleared from queue
        """
        cleared_count = len(self._event_queue)
        self._event_queue.clear()
        self.logger.info(f"Cleared {cleared_count} events from queue")
        return cleared_count

    def get_metrics(self) -> dict[str, Any]:
        """Get event bus metrics.

        Returns:
            Dictionary of event bus metrics
        """
        return {
            'published_events': self._published_events,
            'processed_events': self._processed_events,
            'dropped_events': self._dropped_events,
            'queued_events': len(self._event_queue),
            'active_subscriptions': len(self._filtered_subscriptions),
            'registered_handlers': sum(len(handlers) for handlers in self._event_handlers.values()),
            'processing': self._processing,
        }

    def reset_metrics(self) -> None:
        """Reset all event bus metrics."""
        self._published_events = 0
        self._processed_events = 0
        self._dropped_events = 0
        self.logger.info("Event bus metrics reset")

    def get_status(self) -> dict[str, Any]:
        """Get current event bus status.

        Returns:
            Dictionary of event bus status information
        """
        return {
            'initialized': True,
            'processing': self._processing,
            'queue_size': len(self._event_queue),
            'subscriptions_count': len(self._filtered_subscriptions),
            'handlers_count': sum(len(handlers) for handlers in self._event_handlers.values()),
            'metrics': self.get_metrics(),
        }
