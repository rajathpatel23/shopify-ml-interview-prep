# Build a Least Recently Used (LRU) cache with the following:
from typing import Any, Optional, Union
import time
import threading

class Node:
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.timestamp = time.time()

class DoublyLinkedList:
    def __init__(self):
        """Initialize the doubly linked list."""
        self.head = Node("0", 0)
        self.tail = Node("0", 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def remove(self, node: Node) -> None:
        """Remove the node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
    
    def append(self, node: Node) -> None:
        """Append the node to the head of the list."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node


    def move_to_head(self, node: Node) -> None:
        """Move the node to the head of the list to make it the most recently used item."""
        if self.head.next == node:
            node.timestamp = time.time()
            return
        self.remove(node)
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        node.timestamp = time.time()
        
 







class LRUCache:
    def __init__(self, capacity: int, ttl: Optional[int] = None):
        """Initialize cache with max capacity."""
        self.capacity = capacity
        self.cache = DoublyLinkedList()
        self.cache_map = {}
        self.cache_miss = 0
        self.cache_hits = 0
        self.time_to_live = ttl
        self.size = 0
        self.lock = threading.RLock()


    def get(self, key: str) -> Union[int, None]:
        """Get value by key. Return None if not found or expired."""
        with self.lock:
            if key not in self.cache_map:
                self.cache_miss += 1
                return None
            node = self.cache_map[key]
            ttl_seconds = self.time_to_live
            if ttl_seconds is not None and time.time() - node.timestamp > ttl_seconds:
                self._evict(key)
                self.cache_miss += 1
                return None
            # move the node to the head of the list to make it the most recently used item
            self.cache.move_to_head(node)
            self.cache_hits += 1
            return node.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value. Evict oldest if at capacity."""
        with self.lock:
            if key in self.cache_map:
                node = self.cache_map[key]
                node.value = value
                # node time update in the method itself
                self.cache.move_to_head(node)
                return
            # evict LRU if at capacity (target real node at tail.prev)
            if self.size >= self.capacity:
                lru = self.cache.tail.prev
                if lru is not self.cache.head:
                    self._evict(lru.key)
            new_node = Node(key, value)
            self.cache.append(new_node)
            self.cache_map[key] = new_node
            self.size += 1

    def _evict(self, key: str) -> None:    
        """Evict the least recently used item from the cache."""
        if key not in self.cache_map:
            return
        node = self.cache_map[key]
        self.cache.remove(node)
        self.cache_map.pop(key)
        self.size -= 1

    
    def get_size(self) -> int:
        with self.lock:
            return self.size

    def get_stats(self) -> dict:
        with self.lock:
            total = self.cache_hits + self.cache_miss
            return {
                'hits': self.cache_hits,
                'misses': self.cache_miss,
                'hit_rate': self.cache_hits / total if total > 0 else 0,
                'size': self.size
            }

#   Constraints:
#   - All operations must be O(1) time complexity
#   - Must be thread-safe
#   - Support optional TTL (time-to-live) in seconds
#   - Track hit/miss statistics

#   ---
#   Test Cases to Implement

  # Test 1: Basic get/put
if __name__ == "__main__":
  cache = LRUCache(capacity=2)
  cache.put("a", 1)
  cache.put("b", 2)
  assert cache.get("a") == 1
  assert cache.get("c") is None

#   # Test 2: Eviction (LRU)
  cache.put("c", 3)  # Should evict "b" (least recently used)
  assert cache.get("b") is None
  assert cache.get("a") == 1
  assert cache.get("c") == 3

#   # Test 3: Update refreshes order
  cache.put("a", 10)  # Updates "a", makes it most recent
  cache.put("d", 4)   # Should evict "c" now
  assert cache.get("c") is None
  assert cache.get("a") == 10

#   # Test 4: TTL expiration
  cache_ttl = LRUCache(capacity=10, ttl=1)
  cache_ttl.put("x", 100)
  cache_ttl.put("y", 200)
  cache_ttl.put("z", 300)
  cache_ttl.put("w", 400)
  cache_ttl.put("a", 500)
  assert cache_ttl.get("a") == 500

  time.sleep(1.1)
  assert cache_ttl.get("x") is None  # Expired
  assert cache_ttl.get_stats() == {
    'hits': 1,
    'misses': 1,
    'hit_rate': 0.5,
    'size': 4
  }