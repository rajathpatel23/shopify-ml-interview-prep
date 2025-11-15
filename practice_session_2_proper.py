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
   -
   -
   -

2. How will I test this?
   -
   -
   -

3. What's the core algorithm?
   -
   -

4. What could go wrong?
   -
   -

NOW START CODING AND TALK OUT LOUD!
"""

from typing import Dict, Tuple
import time


# YOUR CODE STARTS HERE
# Remember: TALK OUT LOUD, TEST EACH FUNCTION, THINK EDGE CASES


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
        # TODO: Implement initialization
        pass


    def allow_request(self, user_id: str) -> bool:
        """
        Check if request is allowed for user.

        Returns:
            True if allowed, False if rate limited

        TALK ALOUD: "I need to:
        1. Check current tokens for this user
        2. Refill tokens based on time elapsed
        3. If tokens >= 1, consume one and allow
        4. Otherwise deny"
        """
        # TODO: Implement
        pass


    def _refill_tokens(self, user_id: str) -> float:
        """
        Refill tokens based on time elapsed.

        TALK ALOUD: "I'm calculating tokens to add based on time
        since last refill. Refill rate is max_requests / window_seconds
        tokens per second..."
        """
        # TODO: Implement
        pass


# TEST YOUR CODE IMMEDIATELY AFTER EACH FUNCTION!
if __name__ == "__main__":
    print("="*60)
    print("RATE LIMITER TESTS")
    print("="*60)

    # Test 1: Basic functionality
    print("\nTEST 1: Basic rate limiting")
    print("-"*40)

    # TODO: Create limiter with 5 requests per 10 seconds
    # TODO: Make 7 requests from same user
    # TODO: Verify first 5 succeed, next 2 fail

    # Test 2: Token refill
    print("\nTEST 2: Token refill over time")
    print("-"*40)

    # TODO: Use up all tokens
    # TODO: Wait a bit
    # TODO: Verify tokens refilled

    # Test 3: Multiple users
    print("\nTEST 3: Different users have separate limits")
    print("-"*40)

    # TODO: Test that user1 hitting limit doesn't affect user2

    # Test 4: Edge cases
    print("\nTEST 4: Edge cases")
    print("-"*40)

    # TODO: Empty user_id?
    # TODO: Negative max_requests?
    # TODO: Zero window_seconds?


"""
AFTER YOU'RE DONE:

1. Did you talk out loud the ENTIRE time? ___
2. Did you test after each function? ___
3. Did you handle edge cases? ___
4. Did all tests pass? ___
5. Time taken: _____ minutes

WHAT WENT WELL:
-
-
-

WHAT TO IMPROVE:
-
-
-

COMPARED TO PRACTICE 1:
- More communication? ___
- Better testing? ___
- Fewer bugs? ___
"""
