"""
Threading Lock Demo - See Race Conditions in Action!

Run this to understand WHY locks are needed.
"""

import threading
import time


class UnsafeRateLimiter:
    """Rate limiter WITHOUT lock - has race conditions!"""

    def __init__(self, max_requests: int):
        self.max_requests = max_requests
        self.buckets = {}
        # NO LOCK!

    def allow_request(self, user_id: str) -> bool:
        # Initialize if new user
        if user_id not in self.buckets:
            self.buckets[user_id] = self.max_requests

        # Check tokens
        tokens = self.buckets[user_id]

        # Simulate some processing time (makes race condition more obvious)
        time.sleep(0.0001)

        if tokens >= 1:
            # RACE CONDITION HERE!
            # Multiple threads can read same token value before any writes
            self.buckets[user_id] = tokens - 1
            return True
        else:
            return False


class SafeRateLimiter:
    """Rate limiter WITH lock - thread-safe!"""

    def __init__(self, max_requests: int):
        self.max_requests = max_requests
        self.buckets = {}
        self.lock = threading.Lock()  # ADD LOCK!

    def allow_request(self, user_id: str) -> bool:
        with self.lock:  # USE LOCK!
            # Initialize if new user
            if user_id not in self.buckets:
                self.buckets[user_id] = self.max_requests

            # Check tokens
            tokens = self.buckets[user_id]

            # Simulate processing
            time.sleep(0.0001)

            if tokens >= 1:
                self.buckets[user_id] = tokens - 1
                return True
            else:
                return False


def test_limiter(limiter_class, name: str):
    """Test a rate limiter with multiple threads."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    # Create limiter with 5 tokens
    limiter = limiter_class(max_requests=5)

    results = []
    user_id = "test_user"

    def make_request():
        """Each thread tries to make a request."""
        allowed = limiter.allow_request(user_id)
        results.append(allowed)
        if allowed:
            print(f"  Thread {threading.current_thread().name}: ✓ Allowed")
        else:
            print(f"  Thread {threading.current_thread().name}: ✗ Denied")

    # Create 10 threads trying to make requests simultaneously
    threads = [threading.Thread(target=make_request, name=f"T{i+1}")
               for i in range(10)]

    print(f"\nStarting 10 threads (limit is 5 requests)...\n")

    # Start all threads at once
    start = time.time()
    for t in threads:
        t.start()

    # Wait for all to complete
    for t in threads:
        t.join()

    end = time.time()

    # Count results
    allowed_count = sum(results)
    denied_count = len(results) - allowed_count

    print(f"\nResults:")
    print(f"  Allowed: {allowed_count} (expected: 5)")
    print(f"  Denied: {denied_count} (expected: 5)")
    print(f"  Time: {end - start:.3f}s")

    # Check correctness
    if allowed_count == 5:
        print(f"  ✓ CORRECT! Rate limiting worked.")
    else:
        print(f"  ✗ WRONG! Race condition occurred!")
        print(f"    {allowed_count - 5} extra requests were allowed!")

    # Show final token count
    print(f"  Final tokens: {limiter.buckets.get(user_id, 0)}")


if __name__ == "__main__":
    print("="*60)
    print("THREADING LOCK DEMO")
    print("="*60)
    print("\nThis demo shows WHY you need locks for thread safety.")
    print("We'll run 10 threads trying to make requests")
    print("with a limit of 5 requests.")
    print("\nWatch what happens WITH and WITHOUT locks!")

    # Test without lock (race condition!)
    test_limiter(UnsafeRateLimiter, "WITHOUT Lock (Unsafe)")

    # Test with lock (correct!)
    test_limiter(SafeRateLimiter, "WITH Lock (Safe)")

    print("\n" + "="*60)
    print("EXPLANATION")
    print("="*60)
    print("""
WITHOUT LOCK:
- Multiple threads read the same token count
- They all think tokens are available
- They all write back at the same time
- Result: More requests allowed than limit!

WITH LOCK:
- Only one thread can execute at a time
- Each thread waits for the previous one to finish
- Token count is always correct
- Result: Exactly 5 requests allowed!

FOR YOUR INTERVIEW:
When you add a lock (line 98 in your code), you MUST use it:

    with self.lock:  # ← Actually USE the lock!
        # ... your code here ...

This ensures only ONE thread can execute this code at a time,
preventing race conditions.
    """)
