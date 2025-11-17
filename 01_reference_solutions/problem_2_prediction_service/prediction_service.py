"""
Problem 2: Model Prediction Service

Build a production-ready prediction service that:
- Loads a pre-trained model (or mocks it)
- Accepts input via function call/API
- Validates input data
- Returns predictions with confidence scores
- Handles errors gracefully
- Implements caching for efficiency
- Monitors performance

Interview Focus:
- System design principles
- Error handling and validation
- Performance optimization (caching)
- Code organization and SOLID principles
- Trade-offs discussion
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time
from datetime import datetime, timedelta
from enum import Enum
import pickle
import logging
from functools import wraps


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for different model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class PredictionError(Exception):
    """Custom exception for prediction errors."""
    pass


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def timer_decorator(func):
    """
    Decorator to measure function execution time.

    Interview tip: "This helps us monitor performance in production"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


class SimpleCache:
    """
    Simple in-memory cache with TTL (Time To Live).

    Interview tip: "In production, we'd use Redis or Memcached,
    but this demonstrates the caching pattern"

    Trade-offs:
    - Memory vs Speed: Caching uses memory but speeds up repeated predictions
    - TTL: Prevents stale predictions but might cache-miss more often
    - Size limit: Prevents unbounded memory growth
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time to live for cached items
            max_size: Maximum number of items to cache
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                # Expired, remove it
                del self.cache[key]

        self.misses += 1
        logger.debug(f"Cache miss for key: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        # Interview tip: "Implementing simple LRU by removing oldest if at capacity"
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple eviction strategy)
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (value, datetime.now())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }


class MockModel:
    """
    Mock ML model for demonstration purposes.

    Interview tip: "In a real scenario, this would be a scikit-learn,
    TensorFlow, or PyTorch model loaded from disk"
    """

    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.feature_names = ["feature_1", "feature_2", "feature_3"]
        self.n_features = len(self.feature_names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction - returns random values."""
        if self.model_type == ModelType.CLASSIFICATION:
            # Return class labels (0 or 1)
            return np.random.randint(0, 2, size=X.shape[0])
        else:
            # Return continuous values
            return np.random.randn(X.shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mock probability prediction for classification."""
        if self.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification")

        # Return random probabilities that sum to 1
        n_samples = X.shape[0]
        probs = np.random.rand(n_samples, 2)
        # Normalize to sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class PredictionService:
    """
    Production-ready prediction service.

    Design Principles:
    1. Single Responsibility: Each method has one clear purpose
    2. Open/Closed: Extensible (can add new validation rules) without modifying core logic
    3. Dependency Injection: Model is injected, making it testable
    4. Error Handling: Comprehensive error handling with custom exceptions
    5. Observability: Logging and metrics for monitoring
    """

    def __init__(self, model: MockModel, enable_cache: bool = True):
        """
        Initialize prediction service.

        Args:
            model: Pre-trained model instance
            enable_cache: Whether to enable caching
        """
        self.model = model
        self.cache = SimpleCache() if enable_cache else None
        self.prediction_count = 0
        self.error_count = 0

        logger.info(f"PredictionService initialized with {model.model_type.value} model")

    def _validate_input(self, features: Dict[str, float]) -> None:
        """
        Validate input features.

        Args:
            features: Dictionary of feature name -> value

        Raises:
            ValidationError: If input is invalid

        Interview tip: "Input validation is critical for production systems"
        """
        # Check if features is None or empty
        if not features:
            raise ValidationError("Features cannot be empty")

        # Check if all required features are present
        missing_features = set(self.model.feature_names) - set(features.keys())
        if missing_features:
            raise ValidationError(f"Missing required features: {missing_features}")

        # Check for extra features (warning, not error)
        extra_features = set(features.keys()) - set(self.model.feature_names)
        if extra_features:
            logger.warning(f"Extra features provided (will be ignored): {extra_features}")

        # Check if values are valid numbers
        for name, value in features.items():
            if name in self.model.feature_names:
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Feature '{name}' must be numeric, got {type(value)}")
                if np.isnan(value) or np.isinf(value):
                    raise ValidationError(f"Feature '{name}' has invalid value: {value}")

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array in correct order.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Numpy array of features in correct order
        """
        # Interview tip: "Ensuring features are in the correct order for the model"
        return np.array([[features[name] for name in self.model.feature_names]])

    def _create_cache_key(self, features: Dict[str, float]) -> str:
        """
        Create cache key from features.

        Interview tip: "Using sorted features to ensure consistent keys"
        """
        # Sort for consistent hashing
        sorted_items = sorted(features.items())
        return str(sorted_items)

    @timer_decorator
    def predict(self, features: Dict[str, float],
                return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make a prediction.

        Args:
            features: Dictionary of feature name -> value
            return_probabilities: Whether to return class probabilities (classification only)

        Returns:
            Dictionary containing prediction and metadata

        Raises:
            ValidationError: If input is invalid
            PredictionError: If prediction fails

        Interview tip: "Let me walk through the prediction flow with error handling"
        """
        try:
            # Step 1: Validate input
            self._validate_input(features)

            # Step 2: Check cache
            cache_key = None
            if self.cache:
                cache_key = self._create_cache_key(features)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Returning cached prediction")
                    return cached_result

            # Step 3: Convert to array
            X = self._features_to_array(features)

            # Step 4: Make prediction
            prediction = self.model.predict(X)[0]

            # Step 5: Get probabilities if requested (for classification)
            probabilities = None
            confidence = None
            if return_probabilities and self.model.model_type == ModelType.CLASSIFICATION:
                probs = self.model.predict_proba(X)[0]
                probabilities = probs.tolist()
                confidence = float(np.max(probs))  # Highest probability as confidence

            # Step 6: Build response
            result = {
                "prediction": float(prediction) if isinstance(prediction, np.ndarray) else prediction,
                "model_type": self.model.model_type.value,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"  # Model version for tracking
            }

            if probabilities is not None:
                result["probabilities"] = probabilities
                result["confidence"] = confidence

            # Step 7: Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, result)

            # Step 8: Update metrics
            self.prediction_count += 1

            logger.info(f"Prediction successful: {prediction}")
            return result

        except ValidationError as e:
            # Re-raise validation errors
            self.error_count += 1
            logger.error(f"Validation error: {str(e)}")
            raise

        except Exception as e:
            # Catch any other errors
            self.error_count += 1
            logger.error(f"Prediction error: {str(e)}")
            raise PredictionError(f"Failed to make prediction: {str(e)}")

    def batch_predict(self, batch_features: List[Dict[str, float]],
                     return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of inputs.

        Interview tip: "Batch processing is more efficient for multiple predictions"

        Trade-off:
        - If one prediction fails, do we fail the entire batch or continue?
        - Here we continue and mark individual failures
        """
        results = []

        for i, features in enumerate(batch_features):
            try:
                result = self.predict(features, return_probabilities)
                results.append(result)
            except (ValidationError, PredictionError) as e:
                # Interview tip: "Continuing batch even if one fails"
                logger.error(f"Failed to predict for sample {i}: {str(e)}")
                results.append({
                    "error": str(e),
                    "sample_index": i
                })

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics for monitoring.

        Interview tip: "Metrics are crucial for monitoring production services"
        """
        metrics = {
            "total_predictions": self.prediction_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.prediction_count, 1)
        }

        if self.cache:
            metrics["cache_stats"] = self.cache.get_stats()

        return metrics

    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint for load balancers.

        Interview tip: "Health checks help orchestration systems like Kubernetes"
        """
        try:
            # Try a simple prediction
            test_features = {name: 0.0 for name in self.model.feature_names}
            self.predict(test_features)

            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_type": self.model.model_type.value
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Example usage - Interview walkthrough
if __name__ == "__main__":
    print("="*60)
    print("MODEL PREDICTION SERVICE DEMO")
    print("="*60 + "\n")

    # Interview tip: "Let me demonstrate how this service works"

    # Initialize model and service
    model = MockModel(ModelType.CLASSIFICATION)
    service = PredictionService(model, enable_cache=True)

    print("1. Health Check")
    print("-" * 40)
    health = service.health_check()
    print(f"Health status: {health}\n")

    print("2. Single Prediction")
    print("-" * 40)
    features = {
        "feature_1": 1.5,
        "feature_2": 2.3,
        "feature_3": 0.8
    }

    try:
        result = service.predict(features, return_probabilities=True)
        print(f"Prediction result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    print("3. Cached Prediction (should be faster)")
    print("-" * 40)
    # Same features - should hit cache
    result = service.predict(features, return_probabilities=True)
    print(f"Cached result: {result}\n")

    print("4. Invalid Input Handling")
    print("-" * 40)
    invalid_features = {
        "feature_1": 1.5,
        # Missing feature_2 and feature_3
    }

    try:
        result = service.predict(invalid_features)
    except ValidationError as e:
        print(f"Validation error caught: {e}\n")

    print("5. Batch Prediction")
    print("-" * 40)
    batch = [
        {"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0},
        {"feature_1": 1.5, "feature_2": 2.5, "feature_3": 3.5},
        {"feature_1": 2.0},  # Invalid - missing features
    ]

    results = service.batch_predict(batch, return_probabilities=True)
    for i, res in enumerate(results):
        print(f"Sample {i}: {res}")
    print()

    print("6. Service Metrics")
    print("-" * 40)
    metrics = service.get_metrics()
    print(f"Metrics: {metrics}\n")

    print("\n" + "="*60)
    print("DESIGN TRADE-OFFS DISCUSSION")
    print("="*60)
    print("""
    1. Caching:
       - Pro: Faster responses for repeated predictions
       - Con: Uses memory, might return stale results
       - Trade-off: TTL balances freshness vs cache hits

    2. Error Handling:
       - Continue batch on individual failures vs fail entire batch
       - Chose: Continue (better UX, partial results still useful)

    3. Validation:
       - Strict validation adds latency but prevents bad predictions
       - Trade-off: Validate early to fail fast

    4. Logging:
       - Detailed logging helps debugging but impacts performance
       - Trade-off: Use log levels (DEBUG in dev, INFO in prod)

    5. Synchronous vs Asynchronous:
       - This is synchronous (simpler, easier to reason about)
       - For high load: Use async/await or message queue (Celery, RabbitMQ)
    """)
