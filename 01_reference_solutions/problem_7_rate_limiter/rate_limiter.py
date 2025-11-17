"""
Problem 7: Rate Limiter System

Build a rate limiter that:
- Limits requests per user/API key
- Supports multiple algorithms (Token Bucket, Sliding Window)
- Works for API endpoints
- Returns meaningful error messages

Real-world use cases at Shopify:
- Protecting ML inference APIs from abuse
- Ensuring fair resource allocation
- Preventing DOS attacks

Interview Focus:
- Algorithm design and trade-offs
- Concurrency considerations
- System design thinking
- Performance optimization
"""

from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import time
import threading


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class RateLimitAlgorithm(Enum):
    """Different rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    max_requests: int  # Maximum requests allowed
    window_seconds: int  # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW


class TokenBucketLimiter:
    """
    Token Bucket Algorithm.

    How it works:
    - Bucket has fixed capacity of tokens
    - Tokens refill at constant rate
    - Each request consumes one token
    - If no tokens available, request is rejected

    Trade-offs:
    - Pros: Allows bursts, smooth rate limiting
    - Cons: More complex to implement, requires state management
    - Memory: O(1) per user (just counter and timestamp)

    Interview tip: "Token bucket allows brief bursts while maintaining average rate"
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize token bucket limiter.

        Args:
            config: Rate limit configuration
        """
        self.max_tokens = config.max_requests
        self.refill_rate = config.max_requests / config.window_seconds  # tokens per second
        self.window_seconds = config.window_seconds

        # State per user: {user_id: (tokens, last_refill_time)}
        self.buckets: Dict[str, Tuple[float, float]] = {}
        self.lock = threading.Lock()  # Thread safety

    def _refill_bucket(self, user_id: str) -> float:
        """
        Refill tokens based on time elapsed.

        Returns:
            Current number of tokens
        """
        current_time = time.time()

        if user_id not in self.buckets:
            # New user, give them full bucket
            self.buckets[user_id] = (self.max_tokens, current_time)
            return self.max_tokens

        tokens, last_refill = self.buckets[user_id]

        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = time_elapsed * self.refill_rate

        # Update tokens (capped at max)
        new_tokens = min(self.max_tokens, tokens + tokens_to_add)
        self.buckets[user_id] = (new_tokens, current_time)

        return new_tokens

    def allow_request(self, user_id: str) -> bool:
        """
        Check if request should be allowed.

        Args:
            user_id: Identifier for the user/client

        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:  # Thread-safe
            tokens = self._refill_bucket(user_id)

            if tokens >= 1.0:
                # Consume one token
                new_tokens = tokens - 1.0
                self.buckets[user_id] = (new_tokens, time.time())
                return True
            else:
                return False

    def get_remaining(self, user_id: str) -> Dict[str, any]:
        """Get remaining tokens for a user."""
        with self.lock:
            tokens = self._refill_bucket(user_id)
            return {
                "remaining_tokens": int(tokens),
                "max_tokens": self.max_tokens,
                "refill_rate": f"{self.refill_rate:.2f} tokens/second"
            }


class SlidingWindowLimiter:
    """
    Sliding Window Algorithm.

    How it works:
    - Track timestamps of all requests in the window
    - On new request, remove old requests outside window
    - If count < limit, allow request

    Trade-offs:
    - Pros: Accurate, no bursts
    - Cons: Higher memory usage O(max_requests) per user
    - Memory: Stores all request timestamps

    Interview tip: "Sliding window is more accurate but uses more memory"
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize sliding window limiter.

        Args:
            config: Rate limit configuration
        """
        self.max_requests = config.max_requests
        self.window_seconds = config.window_seconds

        # State per user: {user_id: deque of request timestamps}
        self.windows: Dict[str, deque] = {}
        self.lock = threading.Lock()

    def _clean_old_requests(self, user_id: str) -> None:
        """Remove requests outside the sliding window."""
        if user_id not in self.windows:
            self.windows[user_id] = deque()
            return

        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Remove old requests
        while (self.windows[user_id] and
               self.windows[user_id][0] < window_start):
            self.windows[user_id].popleft()

    def allow_request(self, user_id: str) -> bool:
        """
        Check if request should be allowed.

        Args:
            user_id: Identifier for the user/client

        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            self._clean_old_requests(user_id)

            current_count = len(self.windows[user_id])

            if current_count < self.max_requests:
                # Allow request and record timestamp
                self.windows[user_id].append(time.time())
                return True
            else:
                return False

    def get_remaining(self, user_id: str) -> Dict[str, any]:
        """Get remaining requests for a user."""
        with self.lock:
            self._clean_old_requests(user_id)
            current_count = len(self.windows[user_id])

            # Calculate when window resets
            if self.windows[user_id]:
                oldest_request = self.windows[user_id][0]
                reset_time = oldest_request + self.window_seconds
                seconds_until_reset = max(0, reset_time - time.time())
            else:
                seconds_until_reset = 0

            return {
                "remaining_requests": self.max_requests - current_count,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "reset_in_seconds": int(seconds_until_reset)
            }


class FixedWindowLimiter:
    """
    Fixed Window Algorithm.

    How it works:
    - Divide time into fixed windows (e.g., every minute)
    - Count requests in current window
    - Reset counter at window boundary

    Trade-offs:
    - Pros: Simple, O(1) memory per user
    - Cons: Burst at window boundaries (can get 2x limit)
    - Memory: O(1) per user

    Interview tip: "Fixed window is simplest but has burst problem at boundaries"
    """

    def __init__(self, config: RateLimitConfig):
        self.max_requests = config.max_requests
        self.window_seconds = config.window_seconds

        # State: {user_id: (count, window_start)}
        self.counters: Dict[str, Tuple[int, float]] = {}
        self.lock = threading.Lock()

    def _get_current_window(self) -> float:
        """Get start time of current window."""
        current_time = time.time()
        return current_time - (current_time % self.window_seconds)

    def allow_request(self, user_id: str) -> bool:
        """Check if request should be allowed."""
        with self.lock:
            current_window = self._get_current_window()

            if user_id not in self.counters:
                self.counters[user_id] = (1, current_window)
                return True

            count, window_start = self.counters[user_id]

            # Reset if new window
            if window_start != current_window:
                self.counters[user_id] = (1, current_window)
                return True

            # Same window
            if count < self.max_requests:
                self.counters[user_id] = (count + 1, window_start)
                return True
            else:
                return False

    def get_remaining(self, user_id: str) -> Dict[str, any]:
        """Get remaining requests for a user."""
        with self.lock:
            current_window = self._get_current_window()

            if user_id not in self.counters:
                remaining = self.max_requests
            else:
                count, window_start = self.counters[user_id]
                if window_start == current_window:
                    remaining = self.max_requests - count
                else:
                    remaining = self.max_requests

            next_window = current_window + self.window_seconds
            reset_in = int(next_window - time.time())

            return {
                "remaining_requests": remaining,
                "max_requests": self.max_requests,
                "reset_in_seconds": reset_in
            }


class RateLimiter:
    """
    Unified rate limiter supporting multiple algorithms.

    This is the main class you'd use in production.

    Interview tip: "Using strategy pattern to support different algorithms"
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter with specified algorithm.

        Args:
            config: Rate limit configuration
        """
        self.config = config

        # Factory pattern to create appropriate limiter
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self.limiter = TokenBucketLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.limiter = SlidingWindowLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self.limiter = FixedWindowLimiter(config)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    def check_rate_limit(self, user_id: str) -> None:
        """
        Check rate limit for a user.

        Args:
            user_id: Identifier for the user/client

        Raises:
            RateLimitExceeded: If rate limit is exceeded

        Interview tip: "This would be used as a decorator or middleware in production"
        """
        if not self.limiter.allow_request(user_id):
            remaining_info = self.limiter.get_remaining(user_id)
            raise RateLimitExceeded(
                f"Rate limit exceeded for user {user_id}. "
                f"Limit: {self.config.max_requests} requests per {self.config.window_seconds}s. "
                f"Retry in {remaining_info.get('reset_in_seconds', 0)}s"
            )

    def get_status(self, user_id: str) -> Dict[str, any]:
        """Get current rate limit status for a user."""
        return self.limiter.get_remaining(user_id)


# Example usage for interview
if __name__ == "__main__":
    print("="*60)
    print("RATE LIMITER COMPARISON")
    print("="*60 + "\n")

    # Common config: 5 requests per 10 seconds
    base_config = {
        "max_requests": 5,
        "window_seconds": 10
    }

    algorithms = [
        RateLimitAlgorithm.TOKEN_BUCKET,
        RateLimitAlgorithm.SLIDING_WINDOW,
        RateLimitAlgorithm.FIXED_WINDOW
    ]

    for algo in algorithms:
        print(f"\n{algo.value.upper()}")
        print("-" * 40)

        config = RateLimitConfig(**base_config, algorithm=algo)
        limiter = RateLimiter(config)

        user_id = "user_123"

        # Make 7 requests (should allow 5, reject 2)
        for i in range(7):
            try:
                limiter.check_rate_limit(user_id)
                status = limiter.get_status(user_id)
                remaining = status.get('remaining_requests') or status.get('remaining_tokens', 'N/A')
                print(f"Request {i+1}: ALLOWED - {remaining} remaining")
            except RateLimitExceeded as e:
                print(f"Request {i+1}: REJECTED - {e}")

    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    print("""
    1. TOKEN BUCKET:
       - Memory: O(1) per user (just counter and timestamp)
       - Allows bursts: Yes (good for ML inference spikes)
       - Accuracy: Good
       - Complexity: Medium
       - Best for: Services with bursty traffic

    2. SLIDING WINDOW:
       - Memory: O(max_requests) per user (stores timestamps)
       - Allows bursts: No (strict)
       - Accuracy: Excellent
       - Complexity: Medium
       - Best for: Fair resource allocation

    3. FIXED WINDOW:
       - Memory: O(1) per user
       - Allows bursts: Yes (problem at boundaries - can get 2x limit)
       - Accuracy: Poor (boundary issue)
       - Complexity: Low
       - Best for: Simple use cases, non-critical limits

    PRODUCTION CONSIDERATIONS:
    - Use Redis for distributed rate limiting (multiple servers)
    - Consider rate limits per endpoint, not just per user
    - Add monitoring and alerts for rate limit hits
    - Provide clear error messages with retry-after headers
    - Consider dynamic rate limits based on user tier (free/premium)
    """)

    print("\n" + "="*60)
    print("SHOPIFY ML USE CASE")
    print("="*60)
    print("""
    Example: Product Recommendation API

    Rate Limit Strategy:
    - Free tier: 100 requests/hour (Token Bucket - allows bursts)
    - Paid tier: 1000 requests/hour
    - Internal: 10000 requests/hour

    Why Token Bucket?
    - E-commerce has bursty traffic (flash sales, peak hours)
    - Token bucket allows legitimate bursts while maintaining average rate
    - Better user experience than hard cutoff

    Implementation:
    - Use Redis for distributed rate limiting
    - Different keys for different tiers: "ratelimit:{tier}:{user_id}"
    - Return rate limit headers in API response
    - Monitor rate limit hits to adjust limits
    """)
