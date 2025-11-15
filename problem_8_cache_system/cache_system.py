"""
Problem 8: Cache System with Multiple Eviction Policies

Build a cache system that supports:
- Multiple eviction policies (LRU, LFU, FIFO)
- TTL (Time To Live)
- Thread-safe operations
- Statistics tracking

Real-world ML use cases:
- Caching ML predictions
- Caching feature computations
- Caching API responses

Interview Focus:
- Data structures (choosing right one for the job)
- Algorithm trade-offs
- Concurrency
- System design
"""

from typing import Any, Optional, Dict
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time
from enum import Enum


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class LRUCache:
    """
    Least Recently Used (LRU) Cache.

    Data Structure: OrderedDict (maintains insertion order)
    Time Complexity: O(1) for get and put
    Space Complexity: O(capacity)

    How it works:
    - Most recently used items at the end
    - On access, move item to end
    - On eviction, remove from front

    Interview tip: "LRU is most common in practice - Redis uses it"
    """

    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items
            default_ttl: Default time to live in seconds
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()  # Reentrant lock for thread safety

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            self.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (overrides default)
        """
        with self.lock:
            # If key exists, update it
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = datetime.now()
                self.cache.move_to_end(key)
                return

            # Evict if at capacity
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
                self.evictions += 1

            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl or self.default_ttl
            )
            self.cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2%}",
                "evictions": self.evictions,
                "current_size": len(self.cache),
                "capacity": self.capacity
            }


class LFUCache:
    """
    Least Frequently Used (LFU) Cache.

    Data Structure: Dict + frequency tracking
    Time Complexity: O(1) for get and put
    Space Complexity: O(capacity)

    How it works:
    - Track access frequency for each item
    - On eviction, remove item with lowest frequency
    - If tie, remove least recently used among them

    Interview tip: "LFU is good when access patterns are stable"

    Trade-off: LRU vs LFU
    - LRU: Better for temporal locality (recent items accessed again)
    - LFU: Better for frequency patterns (popular items stay)
    - LRU is simpler and usually sufficient
    """

    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None

            # Increment frequency
            entry.access_count += 1
            entry.last_accessed = datetime.now()

            self.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = datetime.now()
                return

            # Evict if at capacity
            if len(self.cache) >= self.capacity:
                # Find least frequently used
                # Interview tip: "If tie, we use LRU as tiebreaker"
                lfu_key = min(
                    self.cache.keys(),
                    key=lambda k: (
                        self.cache[k].access_count,
                        self.cache[k].last_accessed
                    )
                )
                del self.cache[lfu_key]
                self.evictions += 1

            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl or self.default_ttl
            )
            self.cache[key] = entry

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2%}",
                "evictions": self.evictions,
                "current_size": len(self.cache),
                "capacity": self.capacity,
                "policy": "LFU"
            }


class Cache:
    """
    Unified cache interface supporting multiple eviction policies.

    Interview tip: "Using factory pattern to support different policies"
    """

    def __init__(self, capacity: int = 100,
                 policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl: Optional[int] = None):
        """
        Initialize cache.

        Args:
            capacity: Maximum number of items
            policy: Eviction policy to use
            default_ttl: Default time to live in seconds
        """
        self.policy = policy

        if policy == EvictionPolicy.LRU:
            self._cache = LRUCache(capacity, default_ttl)
        elif policy == EvictionPolicy.LFU:
            self._cache = LFUCache(capacity, default_ttl)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        self._cache.put(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self._cache.delete(key)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._cache.get_stats()
        stats["policy"] = self.policy.value
        return stats


def cached(cache: Cache, ttl: Optional[int] = None):
    """
    Decorator for caching function results.

    Interview tip: "This is how you'd use it in practice"

    Example:
        @cached(my_cache, ttl=300)
        def expensive_computation(x, y):
            return x ** y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("CACHE SYSTEM DEMO")
    print("="*60 + "\n")

    # Demo 1: LRU Cache
    print("1. LRU CACHE")
    print("-" * 40)

    lru_cache = Cache(capacity=3, policy=EvictionPolicy.LRU)

    # Add items
    lru_cache.put("user:1", {"name": "Alice", "age": 30})
    lru_cache.put("user:2", {"name": "Bob", "age": 25})
    lru_cache.put("user:3", {"name": "Charlie", "age": 35})

    print("Added 3 users (capacity is 3)")
    print(f"Stats: {lru_cache.get_stats()}\n")

    # Access user:1 (makes it recently used)
    user1 = lru_cache.get("user:1")
    print(f"Accessed user:1: {user1}")

    # Add user:4 (should evict user:2, the least recently used)
    lru_cache.put("user:4", {"name": "David", "age": 40})
    print("Added user:4 (should evict user:2)\n")

    # Try to get user:2 (should be evicted)
    user2 = lru_cache.get("user:2")
    print(f"Try to get user:2: {user2} (evicted!)")
    print(f"Stats: {lru_cache.get_stats()}\n")

    # Demo 2: TTL (Time To Live)
    print("\n2. TTL (TIME TO LIVE)")
    print("-" * 40)

    ttl_cache = Cache(capacity=10, policy=EvictionPolicy.LRU, default_ttl=2)

    ttl_cache.put("temp_data", "This will expire in 2 seconds")
    print("Added temp_data with 2s TTL")

    # Get immediately
    data = ttl_cache.get("temp_data")
    print(f"Immediately: {data}")

    # Wait and try again
    print("Waiting 3 seconds...")
    time.sleep(3)

    data = ttl_cache.get("temp_data")
    print(f"After 3 seconds: {data} (expired!)\n")

    # Demo 3: LFU vs LRU Comparison
    print("\n3. LFU vs LRU COMPARISON")
    print("-" * 40)

    print("\nAccess pattern: A, B, C, D, A, A, A, B, B, E")
    print("(A is accessed 4 times, B 3 times, C and D once)\n")

    for policy in [EvictionPolicy.LRU, EvictionPolicy.LFU]:
        cache = Cache(capacity=4, policy=policy)

        # Simulate access pattern
        accesses = ["A", "B", "C", "D", "A", "A", "A", "B", "B", "E"]

        for key in accesses:
            existing = cache.get(key)
            if existing is None:
                cache.put(key, f"value_{key}")

        print(f"{policy.value.upper()}:")

        # Check what's in cache
        for key in ["A", "B", "C", "D", "E"]:
            value = cache.get(key)
            status = "IN CACHE" if value else "EVICTED"
            print(f"  {key}: {status}")

        print(f"Stats: {cache.get_stats()}\n")

    # Demo 4: Caching decorator
    print("\n4. CACHING DECORATOR (ML Use Case)")
    print("-" * 40)

    ml_cache = Cache(capacity=100, policy=EvictionPolicy.LRU)

    @cached(ml_cache, ttl=60)
    def predict_price(bedrooms, bathrooms, sqft):
        """Simulate expensive ML prediction."""
        print(f"  Computing prediction for {bedrooms}bd, {bathrooms}ba, {sqft}sqft...")
        time.sleep(0.5)  # Simulate computation
        # Fake prediction
        return bedrooms * 100000 + bathrooms * 50000 + sqft * 200

    # First call - computes
    print("First call (will compute):")
    price1 = predict_price(3, 2, 1500)
    print(f"  Result: ${price1:,}\n")

    # Second call - cached
    print("Second call with same params (cached):")
    price2 = predict_price(3, 2, 1500)
    print(f"  Result: ${price2:,}\n")

    print(f"Cache stats: {ml_cache.get_stats()}\n")

    print("\n" + "="*60)
    print("DESIGN DECISIONS & TRADE-OFFS")
    print("="*60)
    print("""
    1. EVICTION POLICY CHOICE:

       LRU (Least Recently Used):
       - Best for: Temporal locality (recent items likely accessed again)
       - Use case: API responses, user sessions
       - Pros: Simple, works well in practice
       - Cons: Doesn't consider frequency

       LFU (Least Frequently Used):
       - Best for: Stable access patterns (popular items)
       - Use case: Popular product data, trending items
       - Pros: Keeps genuinely popular items
       - Cons: New items might get evicted quickly

       Recommendation for ML:
       - Use LRU for predictions (temporal locality)
       - Use LFU for model parameters (stable patterns)

    2. TTL (Time To Live):
       - Without TTL: Risk of stale data
       - With TTL: More cache misses, but fresh data
       - Adaptive TTL: Vary based on data change frequency

    3. THREAD SAFETY:
       - Used RLock (reentrant lock) for thread safety
       - Trade-off: Performance vs correctness
       - Alternative: Lock-free data structures (more complex)

    4. CAPACITY MANAGEMENT:
       - Fixed capacity: Predictable memory usage
       - Dynamic capacity: Harder to reason about
       - Memory vs Performance trade-off

    5. STATISTICS TRACKING:
       - Adds slight overhead
       - Essential for monitoring and tuning
       - Hit rate target: Usually aim for > 80%

    PRODUCTION CONSIDERATIONS:
    - Use Redis for distributed caching (multiple servers)
    - Monitor cache hit rates and adjust capacity
    - Consider tiered caching (L1: in-memory, L2: Redis, L3: DB)
    - Implement cache warming for predictable access patterns
    - Use consistent hashing for distributed cache
    """)

    print("\n" + "="*60)
    print("ML-SPECIFIC CACHING STRATEGIES")
    print("="*60)
    print("""
    1. PREDICTION CACHING:
       - Cache key: Hash of input features
       - TTL: 1-5 minutes (depends on model update frequency)
       - Policy: LRU (users often retry similar requests)

    2. FEATURE CACHING:
       - Cache computed features (expensive transformations)
       - TTL: Based on data freshness requirements
       - Policy: LRU

    3. MODEL CACHING:
       - Cache loaded models (avoid disk I/O)
       - TTL: Until model update
       - Policy: No eviction (critical for service)

    4. EMBEDDINGS CACHING:
       - Cache vector embeddings (expensive to compute)
       - TTL: Long (embeddings are stable)
       - Policy: LFU (popular items reused)

    Example: Shopify Product Recommendation

    # Cache product embeddings
    embedding_cache = Cache(capacity=10000, policy=EvictionPolicy.LFU)

    # Cache user recommendations
    rec_cache = Cache(capacity=1000, policy=EvictionPolicy.LRU, default_ttl=300)

    This ensures:
    - Popular product embeddings stay in cache (LFU)
    - Recent user recs are fast (LRU)
    - Recs refresh every 5 minutes (TTL)
    """)
