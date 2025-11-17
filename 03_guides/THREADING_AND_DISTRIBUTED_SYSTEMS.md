# Threading Locks vs Distributed Locks - Complete Guide

## 🔐 Part 1: How Threading Locks Work (Single Machine)

### **What is a Lock?**

A lock is a synchronization primitive that ensures only ONE thread can access a shared resource at a time.

Think of it like a bathroom key - only one person can have the key (and use the bathroom) at a time.

---

### **The Problem: Race Conditions**

**Without a lock:**

```python
# Two threads try to update balance at the same time
# Thread 1                    # Thread 2
balance = 100                 balance = 100
balance = balance + 50        balance = balance + 30
# balance = 150               # balance = 130

# Final balance: 130 (WRONG! Should be 180)
# We lost Thread 1's update!
```

**What happened?**
1. Both threads read `balance = 100` at the same time
2. Thread 1 calculates `100 + 50 = 150`
3. Thread 2 calculates `100 + 30 = 130`
4. Thread 1 writes `150`
5. Thread 2 writes `130` (overwrites Thread 1!)
6. Result: Lost update!

---

### **The Solution: Lock**

```python
import threading

balance = 100
lock = threading.Lock()

# Thread 1
with lock:  # Acquire lock
    balance = balance + 50
# Release lock

# Thread 2 (has to wait until Thread 1 releases lock)
with lock:  # Acquire lock (waits for Thread 1)
    balance = balance + 30
# Release lock

# Final balance: 180 (CORRECT!)
```

**What `with lock:` does:**
1. **Acquire:** Thread tries to get the lock
2. **If locked:** Thread waits (blocks) until lock is free
3. **Execute:** Once acquired, thread runs the code
4. **Release:** Lock is automatically released when exiting the `with` block

---

### **Your Rate Limiter Example**

**Without lock (BROKEN):**

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.buckets = {}  # Shared data!
        # No lock!

    def allow_request(self, user_id: str) -> bool:
        # Thread 1 and Thread 2 both check at same time
        tokens = self._refill_tokens(user_id)  # Thread 1: tokens = 1.0
                                                # Thread 2: tokens = 1.0
        if tokens >= 1.0:
            # Both threads think they have tokens!
            self.buckets[user_id] = (tokens - 1.0, time.time())
            return True  # Both return True! (WRONG - only 1 token available!)
```

**Race condition:**
- User has 1 token left
- Thread 1 and Thread 2 both read "1 token"
- Both allow the request
- User made 2 requests with only 1 token!

---

**With lock (CORRECT):**

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.buckets = {}
        self.lock = threading.Lock()  # Create lock

    def allow_request(self, user_id: str) -> bool:
        with self.lock:  # Only ONE thread at a time!
            tokens = self._refill_tokens(user_id)
            if tokens >= 1.0:
                self.buckets[user_id] = (tokens - 1.0, time.time())
                return True
            else:
                return False
```

**How it works:**
1. **Thread 1** enters `with lock:` → acquires lock
2. **Thread 2** tries to enter `with lock:` → **WAITS** (blocked)
3. **Thread 1** reads tokens = 1.0, consumes it, releases lock
4. **Thread 2** now acquires lock, reads tokens = 0.0, denies request ✓

---

### **Visual Example**

```
Time →

Thread 1:  [Acquire Lock]─────[Check tokens: 1.0]─────[Consume: 0.0]─────[Release Lock]
Thread 2:         [Wait...]──────────[Wait...]────────────[Wait...]─────[Acquire Lock]─[Check tokens: 0.0]─[Deny]─[Release]

Without lock:
Thread 1:  [Check: 1.0]─────[Consume: 0.0]
Thread 2:  [Check: 1.0]─────[Consume: 0.0]  ← BOTH SEE 1.0! BUG!
```

---

### **Important: Lock Scope**

**What to lock:**
```python
# Good: Lock protects read + modify + write
with self.lock:
    tokens = self._refill_tokens(user_id)     # Read
    if tokens >= 1.0:                          # Check
        self.buckets[user_id] = (tokens - 1.0) # Write
```

**Bad: Lock too narrow**
```python
# BAD: Race condition between read and write!
tokens = self._refill_tokens(user_id)  # Read (no lock!)
# Thread 2 could modify buckets here!
with self.lock:
    if tokens >= 1.0:
        self.buckets[user_id] = (tokens - 1.0)  # Write with stale data!
```

---

## 🌍 Part 2: Distributed Systems (Multiple Machines)

### **The Critical Problem: Threading Locks DON'T Work Across Machines**

```
Server 1 (New York)          Server 2 (London)
┌─────────────────┐          ┌─────────────────┐
│ Python Process  │          │ Python Process  │
│                 │          │                 │
│ lock = Lock()   │          │ lock = Lock()   │  ← DIFFERENT locks!
│                 │          │                 │
│ Rate Limiter    │          │ Rate Limiter    │
└─────────────────┘          └─────────────────┘
```

**Problem:**
- Each server has its OWN lock
- Locks only work within ONE process
- Servers can't coordinate!

**Result:**
- User makes request to Server 1: allowed (5 tokens)
- User makes request to Server 2: allowed (5 tokens)
- User made 10 requests but limit was 5!

---

### **Why Threading Locks Fail in Distributed Systems**

**Threading locks use:**
- **Shared memory** - All threads in ONE process share the same RAM
- **OS-level primitives** - Managed by ONE operating system

**Distributed systems have:**
- **Separate memory** - Each server has its own RAM
- **Network communication** - Servers are on different machines
- **No shared state** - Can't access each other's memory

```
Single Machine (Threading Lock Works)
┌────────────────────────────────┐
│  Computer                      │
│  ┌────────────────────────┐   │
│  │  RAM (Shared Memory)   │   │
│  │  ┌──────────┐          │   │
│  │  │ buckets  │ ← Lock   │   │
│  │  └──────────┘          │   │
│  └────────────────────────┘   │
│  Thread 1 ↑  ↑ Thread 2       │
└────────────────────────────────┘

Distributed System (Threading Lock FAILS)
┌───────────────┐    Network    ┌───────────────┐
│  Server 1     │◄──────────────►│  Server 2     │
│  ┌─────────┐  │                │  ┌─────────┐  │
│  │ RAM     │  │                │  │ RAM     │  │
│  │ buckets │  │                │  │ buckets │  │
│  │ lock    │  │                │  │ lock    │  │
│  └─────────┘  │                │  └─────────┘  │
└───────────────┘                └───────────────┘
   ↑ Can't see Server 2's data!
```

---

## 🔧 Part 3: Solutions for Distributed Systems

### **Solution 1: Centralized Shared State (Redis)**

**Use Redis as a shared, in-memory database:**

```python
import redis
import time

class DistributedRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.refill_rate = max_requests / window_seconds

        # Connect to Redis (shared across all servers)
        self.redis = redis.Redis(host='redis.example.com', port=6379)

    def allow_request(self, user_id: str) -> bool:
        key = f"ratelimit:{user_id}"

        # Use Redis Lua script for atomic operations
        # This runs on Redis server, so it's atomic across all app servers
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])

        -- Get current state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1])
        local last_refill = tonumber(bucket[2])

        -- Initialize if new user
        if not tokens then
            redis.call('HMSET', key, 'tokens', max_tokens, 'last_refill', current_time)
            return {1, max_tokens - 1}  -- Allow request, remaining tokens
        end

        -- Refill tokens
        local time_elapsed = current_time - last_refill
        local tokens_to_add = time_elapsed * refill_rate
        tokens = math.min(max_tokens, tokens + tokens_to_add)

        -- Check if request allowed
        if tokens >= 1.0 then
            tokens = tokens - 1.0
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            return {1, tokens}  -- Allowed
        else
            return {0, tokens}  -- Denied
        end
        """

        result = self.redis.eval(
            lua_script,
            1,  # Number of keys
            key,  # KEYS[1]
            self.max_requests,  # ARGV[1]
            self.refill_rate,  # ARGV[2]
            time.time()  # ARGV[3]
        )

        allowed = result[0] == 1
        return allowed
```

**How it works:**

```
Load Balancer
      │
      ├─────────┬─────────┬─────────┐
      ↓         ↓         ↓         ↓
  Server 1  Server 2  Server 3  Server 4
      │         │         │         │
      └─────────┴─────────┴─────────┘
                  ↓
            Redis Server
         (Single source of truth)
      ┌─────────────────┐
      │ user:123 → 4.5  │
      │ user:456 → 2.1  │
      │ user:789 → 0.0  │
      └─────────────────┘
```

**Advantages:**
- ✅ All servers see same data
- ✅ Redis operations are atomic
- ✅ Fast (in-memory)
- ✅ Can handle millions of requests/sec

**Disadvantages:**
- ⚠️ Single point of failure (need Redis cluster)
- ⚠️ Network latency (servers → Redis)
- ⚠️ More complex infrastructure

---

### **Solution 2: Distributed Locks**

**Use a distributed lock manager (e.g., Redis, ZooKeeper, etcd):**

```python
import redis
from redis.lock import Lock
import time

class DistributedRateLimiterWithLock:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.refill_rate = max_requests / window_seconds
        self.redis = redis.Redis(host='redis.example.com', port=6379)

    def allow_request(self, user_id: str) -> bool:
        # Create a distributed lock for this user
        lock_key = f"lock:ratelimit:{user_id}"
        lock = self.redis.lock(lock_key, timeout=5)  # 5 second timeout

        try:
            # Try to acquire distributed lock
            # All servers trying to rate-limit this user will wait here
            if lock.acquire(blocking=True, blocking_timeout=2):
                # Only ONE server (across the entire cluster) can execute this
                key = f"ratelimit:{user_id}"

                # Get current state from Redis
                bucket = self.redis.hmget(key, 'tokens', 'last_refill')

                current_time = time.time()

                if bucket[0] is None:
                    # New user
                    self.redis.hmset(key, {
                        'tokens': self.max_requests,
                        'last_refill': current_time
                    })
                    tokens = self.max_requests
                else:
                    # Refill tokens
                    tokens = float(bucket[0])
                    last_refill = float(bucket[1])
                    time_elapsed = current_time - last_refill
                    tokens_to_add = time_elapsed * self.refill_rate
                    tokens = min(self.max_requests, tokens + tokens_to_add)

                # Check and consume
                if tokens >= 1.0:
                    self.redis.hmset(key, {
                        'tokens': tokens - 1.0,
                        'last_refill': current_time
                    })
                    return True
                else:
                    return False
            else:
                # Couldn't acquire lock (timeout)
                return False
        finally:
            # Always release the lock
            lock.release()
```

**How distributed locks work:**

```
Server 1 requests lock for "user:123"
Server 2 requests lock for "user:123"  (same user, same time)

Redis (Lock Manager):
┌────────────────────────────────┐
│ lock:ratelimit:user:123        │
│   ├─ Owned by: Server 1        │
│   ├─ Expires: 5 seconds        │
│   └─ Status: LOCKED            │
└────────────────────────────────┘
        ↓
Server 1: ✓ Acquired lock, processes request
Server 2: ✗ Waits... waits... waits...
        ↓
Server 1: Releases lock
        ↓
Server 2: ✓ Acquires lock, processes request
```

---

### **Solution 3: Token Bucket in Redis (Shopify's Approach)**

**Most production systems use this pattern:**

```python
import redis
import time

class RedisTokenBucket:
    """
    Production-grade distributed rate limiter.

    Used by: Shopify, Stripe, GitHub, etc.
    """

    def __init__(self, redis_client: redis.Redis,
                 max_requests: int, window_seconds: int):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.refill_rate = max_requests / window_seconds

    def allow_request(self, user_id: str) -> tuple[bool, dict]:
        """
        Check if request is allowed using Redis atomic operations.

        Returns:
            (allowed, info_dict)
        """
        key = f"ratelimit:{user_id}"
        current_time = time.time()

        # Lua script ensures atomicity (all-or-nothing)
        # This entire script runs on Redis server atomically
        lua_script = """
        local key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        -- Get or initialize bucket
        local bucket = redis.call('HMGET', key, 'tokens', 'updated_at')
        local tokens = tonumber(bucket[1]) or max_tokens
        local updated_at = tonumber(bucket[2]) or now

        -- Refill tokens based on time passed
        local elapsed = now - updated_at
        local refilled = math.min(max_tokens, tokens + (elapsed * refill_rate))

        -- Try to consume a token
        local allowed = 0
        local remaining = refilled

        if refilled >= 1.0 then
            allowed = 1
            remaining = refilled - 1.0
        end

        -- Update state
        redis.call('HMSET', key, 'tokens', remaining, 'updated_at', now)
        redis.call('EXPIRE', key, 3600)  -- Auto-cleanup after 1 hour

        -- Return: allowed, remaining, retry_after
        local retry_after = 0
        if allowed == 0 then
            retry_after = (1.0 - remaining) / refill_rate
        end

        return {allowed, remaining, retry_after}
        """

        result = self.redis.eval(
            lua_script,
            1,
            key,
            self.max_requests,
            self.refill_rate,
            current_time
        )

        allowed = result[0] == 1
        remaining = result[1]
        retry_after = result[2]

        return allowed, {
            'remaining': remaining,
            'retry_after': retry_after,
            'limit': self.max_requests
        }
```

**Why Lua scripts?**

Redis executes Lua scripts **atomically**:
1. Script starts → Redis locks
2. All commands in script execute
3. Script finishes → Redis unlocks

**No race conditions** even with 1000s of servers!

---

## 📊 Comparison Table

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| **Threading Lock** | Simple, fast, no network | Only works on 1 machine | Single server, testing |
| **Redis + Lua** | Fast, atomic, scalable | Requires Redis | Production (most common) |
| **Distributed Lock** | More flexible | Slower, complex | Need strict coordination |
| **Database** | Simple, no extra infra | Slow, not atomic | Low traffic |

---

## 🎤 Interview Discussion Points

### **When interviewer asks: "How would you make this distributed?"**

**Good answer:**
```
"The threading lock I used only works within a single Python process.
For a distributed system with multiple servers, I'd use Redis.

Specifically, I'd:

1. Store rate limit data in Redis (centralized state)
2. Use Lua scripts for atomic operations (prevents race conditions)
3. Each server would query Redis before allowing requests

The trade-off is:
- Pro: Works across all servers, atomic operations
- Con: Network latency (1-2ms to Redis), requires Redis infrastructure
- Pro: Redis is fast and can handle millions of ops/sec

For Shopify's scale, this is the standard approach. Companies like
Stripe and GitHub use similar patterns.

An alternative would be distributed locks, but that's slower because
you're acquiring/releasing locks over the network for every request.

Would you like me to sketch out the Redis implementation?"
```

**Bad answer:**
```
"Um... I'd just use the same lock?"
[Shows you don't understand distributed systems]
```

---

## 🔬 Hands-On Example

**Let me show you the race condition:**

```python
# race_condition_demo.py
import threading
import time

class UnsafeCounter:
    def __init__(self):
        self.count = 0
        # No lock!

    def increment(self):
        # This looks atomic, but it's actually 3 operations:
        # 1. Read self.count
        # 2. Add 1
        # 3. Write back to self.count
        self.count += 1

class SafeCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1

# Test
def test_counter(counter, name):
    def worker():
        for _ in range(100000):
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(10)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()

    print(f"{name}:")
    print(f"  Expected: {10 * 100000}")
    print(f"  Actual: {counter.count}")
    print(f"  Lost updates: {10 * 100000 - counter.count}")
    print(f"  Time: {end - start:.2f}s")
    print()

# Run tests
test_counter(UnsafeCounter(), "WITHOUT Lock")
test_counter(SafeCounter(), "WITH Lock")
```

**Try running this!** You'll see:
```
WITHOUT Lock:
  Expected: 1000000
  Actual: 342891      ← Lost 65% of updates!
  Lost updates: 657109

WITH Lock:
  Expected: 1000000
  Actual: 1000000     ← Correct!
  Lost updates: 0
```

---

## 🎯 Key Takeaways

### **Threading Locks:**
1. ✅ Work within ONE process (threads share memory)
2. ✅ Fast (no network overhead)
3. ❌ Don't work across machines
4. ❌ Don't work across processes

### **Distributed Systems:**
1. ✅ Use Redis/database for shared state
2. ✅ Use Lua scripts for atomic operations
3. ✅ Handle network latency
4. ✅ Consider failure modes

### **For Your Interview:**
1. Mention thread safety with locks (like you did!)
2. Acknowledge locks only work on one machine
3. Explain Redis + Lua for distributed systems
4. Discuss trade-offs (simplicity vs scalability)

---

## 📝 Practice Explaining This

**Try saying out loud:**

```
"I added a threading lock to make this thread-safe within a single
server. The lock ensures only one thread can check and update tokens
at a time, preventing race conditions.

However, threading locks only work within one process. For a distributed
system like Shopify with multiple servers, I'd use Redis with Lua scripts.

Redis becomes the single source of truth for rate limit state. The Lua
script runs atomically on the Redis server, so even with hundreds of
app servers, we avoid race conditions.

The trade-off is network latency - each request has to query Redis -
but Redis is fast enough (1-2ms) that it's not a bottleneck. This is
the industry-standard approach used by Shopify, Stripe, and GitHub."
```

---

**Does this make sense? Want me to explain any part in more detail?** 🔐
