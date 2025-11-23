"""
Problem 6: API Integration Service

Build a service that integrates with external APIs with caching and rate limiting.

Shopify Context:
- Call external APIs (shipping providers, payment gateways, etc.)
- Handle rate limits from external services
- Cache responses to reduce API calls
- Retry logic for failures
- Monitor API health and latency

Interview Focus:
- API design patterns
- Error handling and retries
- Caching strategies
- Rate limiting
- Async/concurrency
"""

from typing import Dict, Any, Optional, Callable, List
import time
import threading
from collections import OrderedDict
from enum import Enum
import hashlib
import json


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class APIError(Exception):
    """Base exception for API errors."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass


class APITimeoutError(APIError):
    """Raised when API request times out."""
    pass


class APIResponse:
    """
    Represents an API response.

    Interview Tip: "I'm wrapping the response to provide consistent
    error handling and make it easier to mock in tests."
    """

    def __init__(self, status_code: int, data: Any,
                 headers: Optional[Dict[str, str]] = None,
                 latency_ms: float = 0):
        self.status_code = status_code
        self.data = data
        self.headers = headers or {}
        self.latency_ms = latency_ms
        self.cached = False

    @property
    def ok(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300

    def __repr__(self):
        cached_str = " (cached)" if self.cached else ""
        return f"<APIResponse {self.status_code}{cached_str} {self.latency_ms:.1f}ms>"


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Interview Tip: "Using token bucket because it allows bursts while
    maintaining average rate. This is important for API calls where
    you might need to make several calls quickly."
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("max_requests and window_seconds must be positive")

        self.max_tokens = max_requests
        self.refill_rate = max_requests / window_seconds
        self.buckets: Dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self.lock = threading.Lock()

    def _refill_tokens(self, key: str, current_time: float) -> float:
        """Refill tokens based on time elapsed."""
        if key not in self.buckets:
            return self.max_tokens

        tokens, last_refill = self.buckets[key]

        # Calculate tokens to add based on time elapsed
        elapsed = current_time - last_refill
        tokens_to_add = elapsed * self.refill_rate

        # Cap at max_tokens
        return min(self.max_tokens, tokens + tokens_to_add)

    def allow_request(self, key: str = "default") -> bool:
        """
        Check if request is allowed.

        Args:
            key: Identifier for the bucket (e.g., user_id, api_endpoint)

        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            current_time = time.time()
            tokens = self._refill_tokens(key, current_time)

            if tokens >= 1:
                # Allow request and consume token
                self.buckets[key] = (tokens - 1, current_time)
                return True
            else:
                # Rate limited
                self.buckets[key] = (tokens, current_time)
                return False

    def wait_if_needed(self, key: str = "default") -> None:
        """
        Wait if rate limited.

        Interview Tip: "This is useful for batch processing where you want
        to automatically throttle rather than fail."
        """
        while not self.allow_request(key):
            time.sleep(0.1)  # Wait 100ms before retry


class LRUCache:
    """
    LRU cache for API responses.

    Interview Tip: "Using LRU because recent API responses are most likely
    to be requested again. OrderedDict gives us O(1) operations."
    """

    def __init__(self, capacity: int, default_ttl: Optional[int] = None):
        """
        Args:
            capacity: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds (None = no expiry)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()  # key -> (value, expire_time)
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _make_key(self, method: str, endpoint: str, params: Optional[Dict] = None) -> str:
        """Create cache key from request parameters."""
        # Create deterministic key from request
        key_parts = [method, endpoint]
        if params:
            # Sort params for consistency
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)

        key_str = "|".join(key_parts)
        # Hash to keep keys short
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Get cached response.

        Returns:
            Cached value or None if not found/expired
        """
        key = self._make_key(method, endpoint, params)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, expire_time = self.cache[key]

            # Check if expired
            if expire_time is not None and time.time() > expire_time:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return value

    def put(self, method: str, endpoint: str, value: Any,
            params: Optional[Dict] = None, ttl: Optional[int] = None) -> None:
        """
        Cache a response.

        Args:
            method: HTTP method
            endpoint: API endpoint
            value: Response to cache
            params: Request parameters
            ttl: Time-to-live in seconds (overrides default)
        """
        key = self._make_key(method, endpoint, params)

        with self.lock:
            # Calculate expiry time
            ttl_to_use = ttl if ttl is not None else self.default_ttl
            expire_time = time.time() + ttl_to_use if ttl_to_use else None

            # Add to cache
            self.cache[key] = (value, expire_time)
            self.cache.move_to_end(key)

            # Evict if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove oldest

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class APIClient:
    """
    API client with caching, rate limiting, and retry logic.

    Interview Tip: "This combines multiple design patterns: caching for
    performance, rate limiting for API quotas, and retries for reliability.
    In production, I'd also add circuit breakers and observability."
    """

    def __init__(self,
                 base_url: str,
                 rate_limit: int = 100,
                 rate_window: int = 60,
                 cache_size: int = 1000,
                 cache_ttl: int = 300,
                 max_retries: int = 3,
                 timeout: float = 5.0):
        """
        Args:
            base_url: Base URL for the API
            rate_limit: Max requests per window
            rate_window: Time window in seconds
            cache_size: Cache capacity
            cache_ttl: Default cache TTL in seconds
            max_retries: Max retry attempts
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = TokenBucketRateLimiter(rate_limit, rate_window)
        self.cache = LRUCache(cache_size, cache_ttl)
        self.max_retries = max_retries
        self.timeout = timeout

        # Mock HTTP client (in real implementation, use requests/httpx)
        self._http_client: Optional[Callable] = None

        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def set_http_client(self, client: Callable) -> None:
        """
        Set custom HTTP client for dependency injection.

        Interview Tip: "This makes the code testable. In tests, I can inject
        a mock client instead of making real HTTP calls."
        """
        self._http_client = client

    def _make_http_request(self, method: str, url: str,
                          params: Optional[Dict] = None,
                          body: Optional[Dict] = None) -> APIResponse:
        """
        Make actual HTTP request (mock implementation).

        In real implementation, this would use requests or httpx.
        """
        if self._http_client:
            return self._http_client(method, url, params, body)

        # Mock implementation for demo
        start_time = time.time()

        # Simulate network latency
        time.sleep(0.05)  # 50ms

        latency_ms = (time.time() - start_time) * 1000

        # Mock successful response
        return APIResponse(
            status_code=200,
            data={'result': 'success', 'endpoint': url},
            latency_ms=latency_ms
        )

    def request(self,
                method: HTTPMethod,
                endpoint: str,
                params: Optional[Dict] = None,
                body: Optional[Dict] = None,
                cache: bool = True,
                cache_ttl: Optional[int] = None) -> APIResponse:
        """
        Make API request with caching, rate limiting, and retries.

        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., '/users/123')
            params: Query parameters
            body: Request body
            cache: Whether to use cache (only for GET)
            cache_ttl: Override default cache TTL

        Returns:
            APIResponse

        Raises:
            RateLimitError: If rate limit exceeded
            APITimeoutError: If request times out
            APIError: For other API errors

        Interview Tip: "The request flow is: check cache -> check rate limit ->
        retry loop -> cache result. Each step has a clear responsibility."
        """
        method_str = method.value
        url = f"{self.base_url}{endpoint}"

        # Check cache (only for GET requests)
        if method == HTTPMethod.GET and cache:
            cached_response = self.cache.get(method_str, endpoint, params)
            if cached_response is not None:
                cached_response.cached = True
                return cached_response

        # Check rate limit
        if not self.rate_limiter.allow_request(endpoint):
            raise RateLimitError(f"Rate limit exceeded for {endpoint}")

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with self.lock:
                    self.request_count += 1

                # Make request
                response = self._make_http_request(method_str, url, params, body)

                # Cache successful GET responses
                if method == HTTPMethod.GET and cache and response.ok:
                    self.cache.put(method_str, endpoint, response, params, cache_ttl)

                return response

            except Exception as e:
                last_error = e
                with self.lock:
                    self.error_count += 1

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt * 0.1  # 100ms, 200ms, 400ms
                    time.sleep(wait_time)

        # All retries failed
        raise APIError(f"Request failed after {self.max_retries} attempts: {last_error}")

    def get(self, endpoint: str, params: Optional[Dict] = None,
            cache: bool = True, cache_ttl: Optional[int] = None) -> APIResponse:
        """GET request."""
        return self.request(HTTPMethod.GET, endpoint, params=params,
                          cache=cache, cache_ttl=cache_ttl)

    def post(self, endpoint: str, body: Optional[Dict] = None,
             params: Optional[Dict] = None) -> APIResponse:
        """POST request."""
        return self.request(HTTPMethod.POST, endpoint, params=params, body=body, cache=False)

    def put(self, endpoint: str, body: Optional[Dict] = None,
            params: Optional[Dict] = None) -> APIResponse:
        """PUT request."""
        return self.request(HTTPMethod.PUT, endpoint, params=params, body=body, cache=False)

    def delete(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """DELETE request."""
        return self.request(HTTPMethod.DELETE, endpoint, params=params, cache=False)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Interview Tip: "Metrics are crucial in production. You need to monitor
        request rates, error rates, cache hit rates, and latency percentiles."
        """
        cache_stats = self.cache.get_stats()

        with self.lock:
            total_requests = self.request_count
            error_rate = self.error_count / total_requests if total_requests > 0 else 0

        return {
            'total_requests': total_requests,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'cache': cache_stats
        }

    def invalidate_cache(self, endpoint: Optional[str] = None) -> None:
        """
        Invalidate cache.

        Args:
            endpoint: Specific endpoint to invalidate (None = clear all)

        Interview Tip: "Cache invalidation is one of the hardest problems
        in computer science. For now, I'm doing simple TTL + manual invalidation.
        In production, I'd consider event-driven invalidation."
        """
        if endpoint is None:
            self.cache.clear()
        else:
            # Clear all cached responses for this endpoint
            # (simplified - in production, would need more sophisticated invalidation)
            self.cache.clear()


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("API INTEGRATION SERVICE DEMO")
    print("="*60 + "\n")

    # Create API client
    print("1. Creating API client...")
    client = APIClient(
        base_url="https://api.example.com",
        rate_limit=10,  # 10 requests
        rate_window=1,  # per second
        cache_size=100,
        cache_ttl=60  # 60 second cache
    )
    print("   ✓ Client created with rate limit: 10 req/s\n")

    # Make requests
    print("2. Making API requests...")

    # First request (cache miss)
    print("\n   Request 1: GET /users/123")
    response1 = client.get("/users/123")
    print(f"   {response1}")

    # Second request (cache hit)
    print("\n   Request 2: GET /users/123 (same endpoint)")
    response2 = client.get("/users/123")
    print(f"   {response2}")

    # Different endpoint
    print("\n   Request 3: GET /users/456")
    response3 = client.get("/users/456", params={'include': 'orders'})
    print(f"   {response3}")

    # POST request (not cached)
    print("\n   Request 4: POST /orders")
    response4 = client.post("/orders", body={'item': 'widget', 'quantity': 5})
    print(f"   {response4}")

    # Show statistics
    print("\n3. Statistics:")
    stats = client.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['cache']['hits']}")
    print(f"   Cache misses: {stats['cache']['misses']}")
    print(f"   Cache hit rate: {stats['cache']['hit_rate']:.1%}")
    print()

    # Test rate limiting
    print("4. Testing rate limiting...")
    print("   Making 15 rapid requests (limit is 10/second)...")

    allowed = 0
    rate_limited = 0

    for i in range(15):
        try:
            client.get(f"/test/{i}", cache=False)
            allowed += 1
        except RateLimitError:
            rate_limited += 1

    print(f"   Allowed: {allowed}")
    print(f"   Rate limited: {rate_limited}")
    print()

    print("="*60)
    print("INTERVIEW DISCUSSION POINTS")
    print("="*60)
    print("""
    1. CACHING STRATEGY:
       - Cache only GET requests (idempotent)
       - Use LRU eviction (most recent = most likely to be reused)
       - TTL to prevent stale data
       - Trade-off: Memory vs API calls

    2. RATE LIMITING:
       - Token bucket allows bursts while maintaining average rate
       - Per-endpoint limiting prevents one endpoint from starving others
       - Trade-off: Fail fast vs wait (depends on use case)

    3. RETRY LOGIC:
       - Exponential backoff prevents overwhelming failing service
       - Max retries to prevent infinite loops
       - Only retry transient errors (not 4xx client errors)
       - Trade-off: Latency vs reliability

    4. THREAD SAFETY:
       - All shared state protected by locks
       - RLock for cache (allows re-entrant calls)
       - Lock for statistics (atomic updates)

    5. PRODUCTION CONSIDERATIONS:
       - Circuit breaker to prevent cascade failures
       - Observability: metrics, tracing, logging
       - Connection pooling for HTTP clients
       - Async/await for better concurrency
       - Graceful degradation (serve stale cache if API down)
       - Per-customer rate limits (multi-tenant)

    6. TESTING:
       - Dependency injection for HTTP client (easy mocking)
       - Test cache hit/miss scenarios
       - Test rate limiting edge cases
       - Test retry logic with flaky responses

    7. SHOPIFY CONTEXT:
       - Call shipping APIs (Canada Post, UPS, FedEx)
       - Payment gateways (Stripe, PayPal)
       - Inventory management systems
       - Each has different rate limits and SLAs
       - Need to aggregate multiple API calls efficiently
    """)
