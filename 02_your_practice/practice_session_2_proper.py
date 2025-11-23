"""
Practice Session 2: API Rate Limiter - PROPER PRACTICE

TIME LIMIT: 45 minutes
START TIME: _____________ (Write it down!)

RULES FOR THIS PRACTICE:
1. ✅ TALK OUT LOUD the entire time (narrate EVERYTHING)
2. ✅ TEST after EVERY function you write
3. ✅ Think about edge cases BEFORE coding
4. ✅ No bonus features until core works perfectly

PROBLEM:
Build a rate limiter for an API that:
- Limits requests per user (e.g., 10 requests per minute)
- Uses Token Bucket algorithm
- Returns clear error messages when limit exceeded
- Thread-safe (mention it, don't need to implement)

REQUIREMENTS:
1. `RateLimiter` class with `allow_request(user_id: str) -> bool` method
2. Configurable: max_requests and window_seconds
3. Automatically refill tokens over time
4. Clean up old data (don't leak memory)

EXAMPLE USAGE:
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    if limiter.allow_request("user_123"):
        # Process request
        pass
    else:
        # Return 429 Too Many Requests
        pass

BEFORE YOU START - ANSWER THESE:

1. What edge cases do I need to handle?
   - is this time bound
   - if the user does not exist, should we create a new user? what should we the response ?
   - what should we do if the user_id is empty or None?
   - what should we do if the config is invalid in terms of max_requests or window_seconds?

2. How will I test this?
   - test the basic functionality
   - test the token refill over time
   - test that different users have separate limits
   - test the edge cases

3. What's the core algorithm?
   - we will use the Token Bucket algorithm
   - we will store the tokens and the last refill time for each user
   - we will refill the tokens based on the time elapsed
   - we will check if the user has at least 1 token
   - if yes, we will consume one token and allow the request
   - if no, we will deny the request

4. What could go wrong?
   - the token bucket algorithm is not implemented correctly
   - the token bucket algorithm is not thread safe
   - the functionality is too slow 
   - not using the right data structures

NOW START CODING AND TALK OUT LOUD!
"""

from typing import Dict, Tuple, Optional
import time
import pytest
import threading



# how to make this thread safe?
class RateLimiter:
    """
    Rate limiter using Token Bucket algorithm.

    TALK ALOUD: "I'm using Token Bucket because it allows bursts
    while maintaining average rate. The trade-off vs sliding window
    is that token bucket uses less memory but is slightly less accurate..."
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.

        TALK ALOUD: "I'm storing max_requests as the bucket capacity,
        and window_seconds to calculate refill rate. I need to track
        tokens and last refill time per user..."
        """
        self.lock = threading.Lock()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.post_init()
        self.refill_rate = max_requests / window_seconds
        self.buckets: Dict[str, Tuple[float, float]] = {}
       


    def post_init(self):
        with self.lock:
            if self.max_requests <= 0:
                raise ValueError("max_requests must be positive")
            if self.window_seconds <= 0:
                raise ValueError("window_seconds must be positive")

    def allow_request(self, user_id: str) -> Tuple[bool, Optional[str]]:
        if not user_id:
            raise ValueError("user_id cannot be empty")
        with self.lock:
            tokens = self._refill_tokens(user_id)
            if tokens >= 1.0:
                current_time = time.time()
                self.buckets[user_id] = (tokens - 1.0, current_time)
                return True, None
            else:
                retry_after = (1.0 - tokens) * self.refill_rate
                return False, retry_after


    def _refill_tokens(self, user_id: str) -> float:
        with self.lock:
            current_time = time.time()
            if user_id not in self.buckets:
                self.buckets[user_id] = (self.max_requests, current_time)
                return self.max_requests
            tokens, last_refill_time = self.buckets[user_id]
            time_elapsed = current_time - last_refill_time
            tokens_to_add = time_elapsed * self.refill_rate
            new_tokens = min(self.max_requests, tokens + tokens_to_add)
            self.buckets[user_id] = (new_tokens, current_time)
            return new_tokens


# TEST YOUR CODE IMMEDIATELY AFTER EACH FUNCTION!
if __name__ == "__main__":
    print("="*60)
    print("RATE LIMITER TESTS")
    print("="*60)

    # Test 1: Basic functionality
    print("\nTEST 1: Basic rate limiting")
    print("-"*40)

    # TODO: Create limiter with 5 requests per 10 seconds
    limiter = RateLimiter(5, 10)
    max_requests = limiter.max_requests
    results = []
    for i in range(7):
        allowed, retry_after = limiter.allow_request("user1")
        results.append(allowed)
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")
    expected = [True, True, True, True, True, False, False]
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"✓ Rate limiting works correctly!")

    # Test 2: Token refill over time + partial token refill
    time.sleep(6)
    print("\nTEST 2: Token refill over time")
    print("-"*40)
    results = []
    for i in range(7):
        allowed = limiter.allow_request("user1")
        results.append(allowed)
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")
    expected = [True, True, True, False, False, False, False]
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"✓ Rate limiting works correctly!")

    # Test 3: Multiple users have separate limits
    time.sleep(10 + 1)
    print("\nTEST 3: Multiple users have separate limits")
    print("-"*40)
    for user, req in [("user1", 7), ("user2", 6), ("user3", 5)]:
        results = []
        for i in range(req):
            allowed = limiter.allow_request(user)
            results.append(allowed)
            print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")
        expected = [True] * max_requests + [False] * (req - max_requests)
        assert results == expected, f"Expected {expected}, got {results}"

    # Test 4: Use up all tokens
    start_time = time.time()
    print("\nTEST 4: Use up all tokens")
    print("-"*40)
    results = []
    for i in range(30):
        allowed = limiter.allow_request("user3")
        results.append(allowed)
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")
    expected = [False] * 30
    end_time = time.time()
    assert limiter.window_seconds > end_time - start_time, f"Window seconds is not correct"
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"✓ Rate limiting works correctly!")

    # test 5: test the edge cases
    # how to test if a function raises an exception?
    print("\nTEST 5: Test the edge cases")
    print("-"*40)
    results = []
    with pytest.raises(ValueError):
        limiter.allow_request("")
    with pytest.raises(ValueError):
        limiter.allow_request(None)












"""
AFTER YOU'RE DONE:

1. Did you talk out loud the ENTIRE time? yes I did talk out louad a little 
2. Did you test after each function? we did test each function
3. Did you handle edge cases? I have tried to handle test case
4. Did all tests pass? yes all the test cases passed
5. Time taken: 45 minutes

WHAT WENT WELL:
- I have handled the edge cases
- I have tested the code incrementally
- I have made the code thread safe

WHAT TO IMPROVE:
- I still need to make the code more efficient
- I still struggle with confidence on  the data structure I use here
- I did not know how make this thread safe

COMPARED TO PRACTICE 1:
- More communication? I think this okay than previous practice
- Better testing? yes better testing for sure
- Fewer bugs? yes maybe less bugs than previous practice
"""
