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
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import threading
import time
from enum import Enum

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


class CacheStats(BaseModel):
    hits: int = Field(default=0, description="The number of cache hits.")
    misses: int = Field(default=0, description="The number of cache misses.")
    evictions: int = Field(default=0, description="The number of cache evictions.")
    size: int = Field(default=0, description="The size of the cache.")
    capacity: int = Field(default=10, description="The capacity of the cache.")
    default_ttl: Optional[int] = Field(default=10, description="The default time to live in seconds for the cache.")
    policy: EvictionPolicy = Field(default=EvictionPolicy.LRU, description="The eviction policy of the cache.")

class CacheEntry(BaseModel):
    value: Any = Field(..., description="The value of the cache entry.")
    created_at: datetime = Field(default=datetime.now(), description="The timestamp when the cache entry was created.")
    last_accessed: datetime = Field(default=datetime.now(), description="The timestamp when the cache entry was last accessed.")
    access_count: int = Field(default=0, description="The number of times the cache entry has been accessed.")
    ttl_seconds: Optional[int] = Field(default=None, description="The time to live in seconds for the cache entry.")

    class Config:
        arbitrary_types_allowed = True

    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

class LRUCache:
    """Least Recently Used (LRU) Cache."""
    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        """Initialize the LRU cache."""
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()    
        self.lock = threading.RLock()
        self.stats = CacheStats(capacity=capacity, default_ttl=default_ttl, policy=EvictionPolicy.LRU, size=0)

    def get(self, key: str) -> Optional[Any]:
        """Get the value from the cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.stats.evictions += 1
                return None
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put the value into the cache."""
        with self.lock:
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = datetime.now()
                self.cache.move_to_end(key)
                return
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
                self.stats.evictions += 1
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl or self.default_ttl
            )
            self.cache[key] = entry
            return

    def delete(self, key: str) -> bool:
        """Delete the value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats(capacity=self.capacity, 
            default_ttl=self.default_ttl, policy=EvictionPolicy.LRU)

    def size(self) -> int:
        """Get the size of the cache."""
        with self.lock:
            return self.stats.size

    def get_stats(self) -> dict:
        """Get the statistics of the cache."""
        with self.lock:
            return self.stats.model_dump()


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
    
    """Least Frequently Used (LFU) Cache."""
    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        """Initialize the LFU cache."""
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats(capacity=capacity, default_ttl=default_ttl, policy=EvictionPolicy.LFU, size=0)

    def get(self, key: str) -> Optional[Any]:
        """Get the value from the cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.stats.evictions += 1
                return None
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put the value into the cache."""
        with self.lock:
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = datetime.now()
                return
        
            if len(self.cache) >= self.capacity:
                lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
                del self.cache[lfu_key]
                self.stats.evictions += 1
            entry = CacheEntry(value=value, created_at=datetime.now(), 
                        last_accessed=datetime.now(), 
                        access_count=0,
                        ttl_seconds=ttl or self.default_ttl)
            self.cache[key] = entry
            return

    def delete(self, key: str) -> bool:
        """Delete the value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats(capacity=self.capacity, 
            default_ttl=self.default_ttl, policy=EvictionPolicy.LFU)

    def size(self) -> int:
        """Get the size of the cache."""
        with self.lock:
            return self.stats.size

    def get_stats(self) -> dict:
        """Get the statistics of the cache."""
        with self.lock:
            return self.stats.model_dump()


class FIFOCache:
    """First In First Out (FIFO) Cache."""
    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        """Initialize the FIFO cache."""
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats(capacity=capacity, default_ttl=default_ttl, policy=EvictionPolicy.FIFO, size=0)

    def get(self, key: str) -> Optional[Any]:
        """Get the value from the cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.stats.evictions += 1
                return None
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put the value into the cache."""
        with self.lock:
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = datetime.now()
                return
            if len(self.cache) >= self.capacity:
                # find oldest key
                self.cache.popitem(last=False)
                self.stats.evictions += 1
            entry = CacheEntry(value=value, created_at=datetime.now(), 
            last_accessed=datetime.now(), 
            access_count=0,
            ttl_seconds=ttl or self.default_ttl)
            self.cache[key] = entry
            return

    def delete(self, key: str) -> bool:
        """Delete the value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get the size of the cache."""
        with self.lock:
            return self.stats.size

    def get_stats(self) -> dict:
        """Get the statistics of the cache."""
        with self.lock:
            return self.stats.model_dump()


class Cache:
    """Cache system with multiple eviction policies."""
    def __init__(self, capacity: int, policy: EvictionPolicy, default_ttl: Optional[int] = None):
        """Initialize the cache system."""
        self.policy = policy
        if policy == EvictionPolicy.LRU:
            self.cache = LRUCache(capacity, default_ttl)
        elif policy == EvictionPolicy.LFU:
            self.cache = LFUCache(capacity, default_ttl)
        elif policy == EvictionPolicy.FIFO:
            self.cache = FIFOCache(capacity, default_ttl)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

    def get(self, key: str) -> Optional[Any]:
        """Get the value from the cache."""
        return self.cache.get(key)

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put the value into the cache."""
        self.cache.put(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete the value from the cache."""
        return self.cache.delete(key)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def get_stats(self) -> dict:
        """Get the statistics of the cache."""
        return self.cache.get_stats()


def cached(cache: Cache, ttl: Optional[int] = None):
    """Decorator to cache the result of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result
        return wrapper
    return decorator


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
    print(f"Cache stats: {ml_cache.get_stats()}\n")

    # Second call - cached
    print("Second call with same params (cached):")
    price2 = predict_price(3, 2, 1500)
    print(f"  Result: ${price2:,}\n")
    print(f"Cache stats: {ml_cache.get_stats()}\n")