"""
Problem 9: Event Processing System

Build an event-driven system for processing and aggregating events in real-time.

Shopify Context:
- Process order events (created, fulfilled, cancelled)
- User activity events (login, view, purchase)
- Inventory events (stock updates, low stock alerts)
- Real-time analytics and monitoring
- Event-driven architecture (pub/sub pattern)

Interview Focus:
- Event-driven design
- Queue management
- Concurrent processing
- Error handling
- Idempotency
"""

from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue
import time
import hashlib
import json


class EventType(Enum):
    """Types of events."""
    ORDER_CREATED = "order.created"
    ORDER_FULFILLED = "order.fulfilled"
    ORDER_CANCELLED = "order.cancelled"
    USER_LOGIN = "user.login"
    PRODUCT_VIEWED = "product.viewed"
    PRODUCT_PURCHASED = "product.purchased"
    INVENTORY_UPDATE = "inventory.update"
    LOW_STOCK_ALERT = "inventory.low_stock"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """
    Represents an event in the system.

    Interview Tip: "Using dataclass for clean data modeling. In production,
    I'd use Pydantic for validation or Proto buffers for serialization."
    """
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        """Validate event after creation."""
        if not self.event_id:
            raise ValueError("event_id cannot be empty")
        if not self.data:
            raise ValueError("data cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority.value,
            'retry_count': self.retry_count
        }

    def __lt__(self, other: 'Event') -> bool:
        """Compare events for priority queue (higher priority first)."""
        # Higher priority value = higher priority
        # If same priority, earlier timestamp wins
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


class EventHandler:
    """
    Base class for event handlers.

    Interview Tip: "Using abstract base pattern makes it easy to add new
    handlers without modifying existing code (Open/Closed Principle)."
    """

    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def can_handle(self, event: Event) -> bool:
        """
        Check if this handler can process the event.

        Override in subclasses to filter events.
        """
        return True

    def handle(self, event: Event) -> None:
        """
        Process an event.

        Args:
            event: Event to process

        Raises:
            Exception: If processing fails
        """
        raise NotImplementedError("Subclasses must implement handle()")

    def on_success(self, event: Event) -> None:
        """Called after successful processing."""
        with self.lock:
            self.processed_count += 1

    def on_error(self, event: Event, error: Exception) -> None:
        """Called when processing fails."""
        with self.lock:
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self.lock:
            return {
                'name': self.name,
                'processed': self.processed_count,
                'errors': self.error_count
            }


class OrderEventHandler(EventHandler):
    """Handler for order-related events."""

    def __init__(self):
        super().__init__("OrderEventHandler")
        self.order_stats = {
            'created': 0,
            'fulfilled': 0,
            'cancelled': 0
        }
        self.stats_lock = threading.Lock()

    def can_handle(self, event: Event) -> bool:
        """Only handle order events."""
        return event.event_type in [
            EventType.ORDER_CREATED,
            EventType.ORDER_FULFILLED,
            EventType.ORDER_CANCELLED
        ]

    def handle(self, event: Event) -> None:
        """Process order event."""
        order_id = event.data.get('order_id')
        if not order_id:
            raise ValueError("Missing order_id in event data")

        # Simulate processing
        time.sleep(0.01)  # 10ms processing time

        # Update stats
        with self.stats_lock:
            if event.event_type == EventType.ORDER_CREATED:
                self.order_stats['created'] += 1
            elif event.event_type == EventType.ORDER_FULFILLED:
                self.order_stats['fulfilled'] += 1
            elif event.event_type == EventType.ORDER_CANCELLED:
                self.order_stats['cancelled'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics including order-specific stats."""
        base_stats = super().get_stats()
        with self.stats_lock:
            base_stats['order_stats'] = self.order_stats.copy()
        return base_stats


class InventoryEventHandler(EventHandler):
    """Handler for inventory events."""

    def __init__(self, low_stock_threshold: int = 10):
        super().__init__("InventoryEventHandler")
        self.low_stock_threshold = low_stock_threshold
        self.inventory: Dict[str, int] = {}
        self.low_stock_alerts: List[str] = []
        self.inventory_lock = threading.Lock()

    def can_handle(self, event: Event) -> bool:
        """Only handle inventory events."""
        return event.event_type in [
            EventType.INVENTORY_UPDATE,
            EventType.LOW_STOCK_ALERT
        ]

    def handle(self, event: Event) -> None:
        """Process inventory event."""
        product_id = event.data.get('product_id')
        if not product_id:
            raise ValueError("Missing product_id in event data")

        with self.inventory_lock:
            if event.event_type == EventType.INVENTORY_UPDATE:
                quantity = event.data.get('quantity', 0)
                self.inventory[product_id] = quantity

                # Check if we need to create low stock alert
                if quantity <= self.low_stock_threshold:
                    self.low_stock_alerts.append(product_id)

            elif event.event_type == EventType.LOW_STOCK_ALERT:
                self.low_stock_alerts.append(product_id)

    def get_low_stock_products(self) -> List[str]:
        """Get list of products with low stock."""
        with self.inventory_lock:
            return self.low_stock_alerts.copy()


class EventProcessor:
    """
    Main event processor with priority queue and worker threads.

    Interview Tip: "Using producer-consumer pattern with priority queue.
    Worker threads process events concurrently. This is similar to how
    Celery or RabbitMQ works."
    """

    def __init__(self, num_workers: int = 4, max_queue_size: int = 1000):
        """
        Args:
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size (0 = unlimited)
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

        # Priority queue for events
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)

        # Handlers
        self.handlers: List[EventHandler] = []
        self.handlers_lock = threading.RLock()

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Idempotency tracking (prevent duplicate processing)
        self.processed_event_ids: Set[str] = set()
        self.processed_lock = threading.Lock()

        # Dead letter queue for failed events
        self.dead_letter_queue: List[Event] = []
        self.dlq_lock = threading.Lock()

        # Statistics
        self.total_events = 0
        self.processed_events = 0
        self.failed_events = 0
        self.duplicate_events = 0
        self.stats_lock = threading.Lock()

    def register_handler(self, handler: EventHandler) -> None:
        """
        Register an event handler.

        Interview Tip: "Handlers are registered at startup. In production,
        I'd use dependency injection or a plugin system."
        """
        with self.handlers_lock:
            self.handlers.append(handler)

    def start(self) -> None:
        """
        Start worker threads.

        Interview Tip: "Workers are daemon threads so they'll shut down
        gracefully when main program exits."
        """
        if self.running:
            raise RuntimeError("Processor already running")

        self.running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop processor and wait for workers to finish.

        Args:
            timeout: Maximum time to wait for workers
        """
        if not self.running:
            return

        self.running = False

        # Wait for queue to be empty
        self.event_queue.join()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)

        self.workers.clear()

    def publish(self, event: Event) -> bool:
        """
        Publish an event to the queue.

        Args:
            event: Event to publish

        Returns:
            True if published successfully, False if queue is full

        Interview Tip: "Non-blocking publish with try/except. In production,
        I'd consider backpressure mechanisms or circuit breakers."
        """
        if not self.running:
            raise RuntimeError("Processor not running. Call start() first.")

        with self.stats_lock:
            self.total_events += 1

        try:
            # Non-blocking put with immediate timeout
            self.event_queue.put(event, block=False)
            return True
        except queue.Full:
            return False

    def _is_duplicate(self, event: Event) -> bool:
        """
        Check if event was already processed (idempotency check).

        Interview Tip: "Idempotency is crucial in distributed systems.
        Events might be delivered multiple times due to retries."
        """
        with self.processed_lock:
            if event.event_id in self.processed_event_ids:
                return True
            self.processed_event_ids.add(event.event_id)
            return False

    def _worker_loop(self) -> None:
        """
        Main worker loop - pulls events from queue and processes them.

        Interview Tip: "Each worker runs independently. They coordinate
        through the shared queue and thread-safe handlers."
        """
        while self.running:
            try:
                # Get event from queue (with timeout to allow checking self.running)
                event = self.event_queue.get(timeout=0.5)

                try:
                    # Check for duplicates
                    if self._is_duplicate(event):
                        with self.stats_lock:
                            self.duplicate_events += 1
                        continue

                    # Process event
                    self._process_event(event)

                finally:
                    # Mark task as done
                    self.event_queue.task_done()

            except queue.Empty:
                # No events in queue, continue loop
                continue
            except Exception as e:
                # Unexpected error in worker loop
                print(f"Worker error: {e}")

    def _process_event(self, event: Event) -> None:
        """
        Process a single event with all registered handlers.

        Interview Tip: "Each event can be processed by multiple handlers.
        This is the pub/sub pattern - one publisher, many subscribers."
        """
        processed_by_any = False

        with self.handlers_lock:
            handlers = [h for h in self.handlers if h.can_handle(event)]

        for handler in handlers:
            try:
                # Process event
                handler.handle(event)
                handler.on_success(event)
                processed_by_any = True

            except Exception as e:
                # Handler failed
                handler.on_error(event, e)

                # Retry logic
                if event.retry_count < event.max_retries:
                    event.retry_count += 1
                    # Re-queue with exponential backoff
                    time.sleep(0.1 * (2 ** event.retry_count))
                    try:
                        self.event_queue.put(event, block=False)
                    except queue.Full:
                        # Queue full, send to DLQ
                        self._send_to_dlq(event, e)
                else:
                    # Max retries exceeded, send to DLQ
                    self._send_to_dlq(event, e)

        if processed_by_any:
            with self.stats_lock:
                self.processed_events += 1

    def _send_to_dlq(self, event: Event, error: Exception) -> None:
        """
        Send failed event to dead letter queue.

        Interview Tip: "Dead letter queue captures events that can't be
        processed. You can inspect them later to debug issues."
        """
        with self.dlq_lock:
            self.dead_letter_queue.append(event)

        with self.stats_lock:
            self.failed_events += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get processor and handler statistics."""
        with self.stats_lock:
            processor_stats = {
                'total_events': self.total_events,
                'processed_events': self.processed_events,
                'failed_events': self.failed_events,
                'duplicate_events': self.duplicate_events,
                'queue_size': self.event_queue.qsize(),
                'dlq_size': len(self.dead_letter_queue)
            }

        with self.handlers_lock:
            handler_stats = [h.get_stats() for h in self.handlers]

        return {
            'processor': processor_stats,
            'handlers': handler_stats
        }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("EVENT PROCESSING SYSTEM DEMO")
    print("="*60 + "\n")

    # Create processor
    print("1. Creating event processor...")
    processor = EventProcessor(num_workers=4, max_queue_size=100)

    # Register handlers
    order_handler = OrderEventHandler()
    inventory_handler = InventoryEventHandler(low_stock_threshold=10)

    processor.register_handler(order_handler)
    processor.register_handler(inventory_handler)

    print("   ✓ Processor created with 4 workers")
    print("   ✓ Registered OrderEventHandler")
    print("   ✓ Registered InventoryEventHandler\n")

    # Start processor
    processor.start()
    print("2. Starting processor...\n")

    # Publish events
    print("3. Publishing events...")

    events = [
        # Order events
        Event(
            event_id="order-1",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            data={'order_id': 'ORD-001', 'total': 99.99},
            priority=EventPriority.NORMAL
        ),
        Event(
            event_id="order-2",
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            data={'order_id': 'ORD-002', 'total': 149.99},
            priority=EventPriority.HIGH
        ),
        Event(
            event_id="order-3",
            event_type=EventType.ORDER_FULFILLED,
            timestamp=datetime.now(),
            data={'order_id': 'ORD-001'},
            priority=EventPriority.NORMAL
        ),

        # Inventory events
        Event(
            event_id="inv-1",
            event_type=EventType.INVENTORY_UPDATE,
            timestamp=datetime.now(),
            data={'product_id': 'PROD-001', 'quantity': 50},
            priority=EventPriority.NORMAL
        ),
        Event(
            event_id="inv-2",
            event_type=EventType.INVENTORY_UPDATE,
            timestamp=datetime.now(),
            data={'product_id': 'PROD-002', 'quantity': 5},
            priority=EventPriority.CRITICAL
        ),

        # Duplicate event (should be deduplicated)
        Event(
            event_id="order-1",  # Same ID as first event
            event_type=EventType.ORDER_CREATED,
            timestamp=datetime.now(),
            data={'order_id': 'ORD-001', 'total': 99.99},
            priority=EventPriority.NORMAL
        ),
    ]

    for event in events:
        success = processor.publish(event)
        status = "✓" if success else "✗"
        print(f"   {status} Published {event.event_type.value} (ID: {event.event_id})")

    print()

    # Wait for processing
    print("4. Processing events...")
    time.sleep(1)  # Give workers time to process

    # Show statistics
    print("\n5. Statistics:")
    stats = processor.get_stats()

    print("\n   Processor:")
    print(f"     Total events: {stats['processor']['total_events']}")
    print(f"     Processed: {stats['processor']['processed_events']}")
    print(f"     Failed: {stats['processor']['failed_events']}")
    print(f"     Duplicates: {stats['processor']['duplicate_events']}")
    print(f"     Queue size: {stats['processor']['queue_size']}")

    print("\n   Handlers:")
    for handler_stat in stats['handlers']:
        print(f"\n     {handler_stat['name']}:")
        print(f"       Processed: {handler_stat['processed']}")
        print(f"       Errors: {handler_stat['errors']}")
        if 'order_stats' in handler_stat:
            print(f"       Orders: {handler_stat['order_stats']}")

    # Show low stock products
    low_stock = inventory_handler.get_low_stock_products()
    if low_stock:
        print(f"\n   Low stock alerts: {low_stock}")

    # Stop processor
    print("\n6. Stopping processor...")
    processor.stop()
    print("   ✓ Processor stopped\n")

    print("="*60)
    print("INTERVIEW DISCUSSION POINTS")
    print("="*60)
    print("""
    1. EVENT-DRIVEN ARCHITECTURE:
       - Pub/Sub pattern: One publisher, many subscribers
       - Loose coupling: Handlers don't know about each other
       - Easy to add new handlers without modifying existing code
       - Trade-off: Eventually consistent (not immediate)

    2. CONCURRENCY MODEL:
       - Producer-consumer pattern with thread pool
       - Priority queue ensures high-priority events processed first
       - Lock-free queue operations (queue.Queue is thread-safe)
       - Each handler must be thread-safe

    3. RELIABILITY:
       - Idempotency: Duplicate events are detected and skipped
       - Retry logic: Failed events are retried with exponential backoff
       - Dead Letter Queue: Unprocessable events are captured
       - Trade-off: Throughput vs reliability

    4. SCALABILITY:
       - Horizontal: Add more worker threads
       - Vertical: Use distributed queue (RabbitMQ, Kafka, SQS)
       - Partitioning: Shard by event type or key
       - Current: Single machine, limited by GIL

    5. PRODUCTION CONSIDERATIONS:
       - Persistent queue (Redis, Kafka) instead of in-memory
       - Distributed tracing (OpenTelemetry)
       - Metrics: throughput, latency, error rate
       - Circuit breakers for handlers
       - Schema validation (Pydantic, JSON Schema)
       - Event sourcing for audit trail

    6. SHOPIFY USE CASES:
       - Order processing pipeline (create → fulfill → ship → deliver)
       - Inventory synchronization across warehouses
       - Real-time analytics (sales dashboard)
       - Notification system (email, SMS, webhooks)
       - Fraud detection (analyze patterns across events)

    7. ALTERNATIVE APPROACHES:
       - Celery: Distributed task queue (Python)
       - Kafka: High-throughput event streaming
       - RabbitMQ: Traditional message broker
       - AWS SQS/SNS: Managed queue service
       - Redis Streams: Lightweight event streaming

    8. TESTING:
       - Unit test handlers in isolation
       - Integration test with mock queue
       - Load test: Can it handle 10k events/sec?
       - Chaos test: What happens if handler crashes?
       - Idempotency test: Send duplicate events
    """)
