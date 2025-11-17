# Threading & Concurrency in ML Systems - Complete Guide

## 🧠 How Threading/Locks Apply to ML Training & Inference

This is a **critical topic** for Senior ML Engineer interviews at companies like Shopify!

---

## 🎯 Part 1: ML Inference (Prediction Serving)

### **The Problem: Multiple Users Making Predictions**

**Scenario at Shopify:**
- 1000s of users requesting product recommendations simultaneously
- One model loaded in memory
- Multiple threads handling requests

**Two common patterns:**

---

### **Pattern 1: Shared Model (Thread-Safe)**

```python
import threading
import numpy as np
import torch  # or tensorflow
from typing import Dict, List

class ModelServer:
    """
    Serves predictions from a single model instance.

    Problem: Multiple threads calling predict() on same model
    Solution: Most ML frameworks are thread-safe for inference
    """

    def __init__(self, model_path: str):
        # Load model ONCE (expensive operation)
        self.model = torch.load(model_path)
        self.model.eval()  # Set to inference mode

        # NO LOCK NEEDED for PyTorch/TensorFlow inference!
        # They use thread-safe operations internally

        # BUT you might need locks for:
        self.request_count = 0
        self.lock = threading.Lock()  # For shared state

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make prediction (thread-safe in PyTorch/TF).

        Multiple threads can call this simultaneously!
        """
        # PyTorch inference is thread-safe
        with torch.no_grad():  # Disable gradient computation
            tensor_input = torch.from_numpy(features)
            prediction = self.model(tensor_input)

        # Update metrics (needs lock!)
        with self.lock:
            self.request_count += 1

        return prediction.numpy()


# Multiple threads can use this safely
server = ModelServer("model.pth")

def worker(user_features):
    # Many threads calling this at once - safe!
    prediction = server.predict(user_features)
    return prediction
```

**Key Point:**
- ✅ PyTorch/TensorFlow inference IS thread-safe (read-only operations)
- ❌ Shared state (counters, caches) is NOT thread-safe without locks
- ✅ Model weights are only read, never written during inference

---

### **Pattern 2: Model Pool (Better for High Load)**

```python
from queue import Queue
import threading

class ModelPool:
    """
    Pool of model instances for parallel inference.

    Why: Avoid GIL contention, better throughput
    Trade-off: Uses more memory (multiple model copies)
    """

    def __init__(self, model_path: str, pool_size: int = 4):
        self.pool = Queue(maxsize=pool_size)

        # Create multiple model instances
        for _ in range(pool_size):
            model = torch.load(model_path)
            model.eval()
            self.pool.put(model)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Get model from pool, predict, return model.
        """
        # Block until a model is available
        model = self.pool.get()  # Thread-safe queue

        try:
            with torch.no_grad():
                tensor_input = torch.from_numpy(features)
                prediction = model(tensor_input)
            return prediction.numpy()
        finally:
            # Always return model to pool
            self.pool.put(model)


# Usage with multiple threads
pool = ModelPool("model.pth", pool_size=4)

def worker(features):
    # Up to 4 threads can predict in parallel
    return pool.predict(features)
```

**When to use:**
- Heavy CPU-bound models
- High request volume
- GIL (Global Interpreter Lock) is bottleneck

---

### **Pattern 3: Async Inference (Modern Approach)**

```python
import asyncio
import torch
from typing import List

class AsyncModelServer:
    """
    Async inference server using batching.

    Best practice for production ML serving!
    """

    def __init__(self, model_path: str, batch_size: int = 32):
        self.model = torch.load(model_path)
        self.model.eval()
        self.batch_size = batch_size

        # Request batching
        self.pending_requests = []
        self.lock = asyncio.Lock()  # Async lock, not threading!

    async def predict_single(self, features: np.ndarray) -> np.ndarray:
        """
        Add request to batch, wait for result.

        This batches requests for efficiency!
        """
        # Create future to wait for result
        future = asyncio.Future()

        async with self.lock:
            self.pending_requests.append((features, future))

            # Trigger batch processing if full
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch()

        # Wait for this request's result
        return await future

    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        if not self.pending_requests:
            return

        # Get all pending requests
        batch_data = []
        futures = []

        for features, future in self.pending_requests:
            batch_data.append(features)
            futures.append(future)

        self.pending_requests = []

        # Batch inference (much faster than one-by-one!)
        batch_input = torch.stack([torch.from_numpy(f) for f in batch_data])

        with torch.no_grad():
            batch_output = self.model(batch_input)

        # Send results back to each request
        for future, output in zip(futures, batch_output):
            future.set_result(output.numpy())


# Usage
server = AsyncModelServer("model.pth", batch_size=32)

async def handle_request(features):
    # Multiple coroutines can call this
    # They'll be automatically batched!
    prediction = await server.predict_single(features)
    return prediction
```

**Advantages:**
- ✅ Automatic request batching (huge speedup!)
- ✅ Better GPU utilization
- ✅ Industry standard (TensorFlow Serving, TorchServe use this)

---

## 🏋️ Part 2: ML Training (More Complex!)

### **Single GPU Training - GIL Impact**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import threading

class Trainer:
    """
    Training with data loading in background threads.

    Python's GIL (Global Interpreter Lock):
    - Only ONE Python thread runs at a time
    - But I/O operations (reading files) release GIL
    - So data loading can be parallel while training happens
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self):
        """
        Standard training loop.

        DataLoader uses threads internally for loading!
        """
        # num_workers=4 means 4 background threads loading data
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,  # ← Background threads!
            pin_memory=True  # Faster GPU transfer
        )

        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(train_loader):
                # While we're computing gradients here,
                # background threads are loading next batch!

                output = self.model(data)
                loss = criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

**Key Insight:**
- Training itself doesn't need locks (single thread)
- Data loading happens in background (multiple threads)
- PyTorch manages thread safety internally

---

### **Distributed Training (Multiple GPUs/Machines)**

**This is where it gets interesting for distributed systems!**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    """
    Train model across multiple GPUs/machines.

    Similar to our rate limiter distributed problem!

    Instead of Redis, we use:
    - NCCL (NVIDIA Collective Communications Library)
    - Gloo (Facebook's collective library)
    - MPI (Message Passing Interface)
    """

    def __init__(self, rank: int, world_size: int):
        """
        Args:
            rank: This process's ID (0 to world_size-1)
            world_size: Total number of processes (GPUs)
        """
        self.rank = rank
        self.world_size = world_size

        # Initialize distributed backend
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU communication
            init_method='tcp://localhost:23456',
            rank=rank,
            world_size=world_size
        )

        # Create model and wrap with DDP
        self.model = MyModel().to(rank)
        self.model = DDP(self.model, device_ids=[rank])

    def train(self):
        """
        Each GPU runs this simultaneously!

        Synchronization happens automatically:
        1. Each GPU computes gradients on its batch
        2. Gradients are averaged across all GPUs (AllReduce)
        3. All GPUs update weights with averaged gradients
        """
        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.rank), target.to(self.rank)

                output = self.model(data)
                loss = criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()  # Compute gradients

                # DDP automatically synchronizes gradients here!
                # This is like our distributed lock but for gradients!
                self.optimizer.step()


# Launch multiple processes (one per GPU)
def main():
    world_size = 4  # 4 GPUs

    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

def train_worker(rank, world_size):
    trainer = DistributedTrainer(rank, world_size)
    trainer.train()
```

**Key Concepts (Similar to Distributed Rate Limiting!):**

| Distributed Training | Distributed Rate Limiting |
|---------------------|---------------------------|
| Multiple GPUs/machines | Multiple app servers |
| Synchronize gradients | Synchronize token counts |
| AllReduce operation | Redis Lua script |
| NCCL/Gloo library | Redis |
| Ring-AllReduce algorithm | Token bucket algorithm |

---

### **The Gradient Synchronization Problem**

**Without synchronization (broken):**
```
GPU 0: Computes gradients on batch 1
GPU 1: Computes gradients on batch 2
GPU 2: Computes gradients on batch 3

Each GPU updates model independently
→ Models diverge!
→ Training fails!
```

**With synchronization (correct):**
```
GPU 0: Computes gradients → [Send to all]
GPU 1: Computes gradients → [Send to all]
GPU 2: Computes gradients → [Send to all]
            ↓
    [Average all gradients]
            ↓
All GPUs receive averaged gradients
            ↓
All GPUs update with same gradients
→ Models stay in sync!
```

**This is implemented with AllReduce:**

```python
# Conceptually similar to:
class GradientSynchronizer:
    """
    Like our Redis rate limiter, but for gradients!
    """

    def synchronize_gradients(self, local_gradients):
        """
        Average gradients across all workers.

        Similar to how we synchronized rate limits across servers!
        """
        # Send local gradients to all other workers
        # Receive gradients from all other workers
        # Average them
        # Return averaged gradients

        # This is what PyTorch DDP does automatically!
        pass
```

---

## 🎯 Part 3: Real Production Scenarios at Shopify

### **Scenario 1: Product Recommendation Inference**

```python
class ProductRecommendationServer:
    """
    Serves product recommendations at scale.

    Traffic: 10,000 requests/second
    Model: Embedding-based recommender
    Infrastructure: 100 servers behind load balancer
    """

    def __init__(self, model_path: str):
        # Each server has its own model copy
        self.model = load_model(model_path)
        self.model.eval()

        # Shared state needs synchronization
        self.request_count = 0
        self.lock = threading.Lock()

        # Cache for embeddings (read-heavy, rarely updated)
        self.product_embeddings = {}
        self.embedding_lock = threading.RLock()

    def get_recommendations(self, user_id: str, top_k: int = 10):
        """
        Get product recommendations for user.

        Thread-safety considerations:
        1. Model inference - thread-safe (read-only)
        2. Embedding cache - needs read-write lock
        3. Request counting - needs lock
        """
        # Update counter (shared state - needs lock)
        with self.lock:
            self.request_count += 1

        # Get user embedding (thread-safe)
        user_embedding = self.model.get_user_embedding(user_id)

        # Get product embeddings (potentially update cache)
        product_embeddings = self._get_product_embeddings()

        # Compute similarities (thread-safe, read-only)
        similarities = compute_similarity(user_embedding, product_embeddings)

        # Return top K (thread-safe)
        top_products = get_top_k(similarities, k=top_k)

        return top_products

    def _get_product_embeddings(self):
        """
        Get product embeddings with caching.

        Multiple threads reading: OK
        One thread writing while others read: NOT OK (need lock!)
        """
        # Read lock (multiple readers allowed)
        with self.embedding_lock:
            if not self.product_embeddings:
                # Only first thread does this
                self.product_embeddings = self._compute_all_embeddings()

            return self.product_embeddings

    def update_product_embeddings(self, new_embeddings):
        """
        Update embeddings (called when new products added).

        Needs write lock to prevent reading partially updated data!
        """
        with self.embedding_lock:
            self.product_embeddings = new_embeddings
```

**Key Points:**
- Model inference: Thread-safe (read-only)
- Caching: Needs locks (read-write)
- Metrics: Needs locks (shared state)

---

### **Scenario 2: Model Update Without Downtime**

```python
class LiveModelUpdater:
    """
    Update model in production without stopping service.

    Challenge: How to swap models while serving requests?
    """

    def __init__(self, initial_model_path: str):
        self.current_model = load_model(initial_model_path)
        self.model_lock = threading.RLock()
        self.model_version = 1

    def predict(self, features):
        """
        Make prediction with current model.

        Uses read lock - multiple threads can predict simultaneously.
        """
        with self.model_lock:
            # Get reference to current model
            model = self.current_model
            version = self.model_version

        # Release lock before inference (allows model updates)
        prediction = model.predict(features)

        return {
            'prediction': prediction,
            'model_version': version
        }

    def update_model(self, new_model_path: str):
        """
        Hot-swap the model.

        Uses write lock - blocks all predictions during swap.
        """
        # Load new model (outside lock - takes time!)
        new_model = load_model(new_model_path)

        # Quick swap with write lock
        with self.model_lock:
            old_model = self.current_model
            self.current_model = new_model
            self.model_version += 1

        # Clean up old model (outside lock)
        del old_model

        print(f"Model updated to version {self.model_version}")
```

---

### **Scenario 3: Distributed Training for Product Embeddings**

```python
class DistributedEmbeddingTrainer:
    """
    Train product embeddings across multiple GPUs.

    Similar architecture to our distributed rate limiter!

    Instead of:
    - Redis → NCCL (for GPU communication)
    - Token counts → Gradients
    - Rate limiting → Gradient averaging
    """

    def __init__(self, rank: int, world_size: int):
        # Initialize distributed backend (like connecting to Redis)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        # Each GPU has same model (like each server has same rate limiter)
        self.model = EmbeddingModel().to(rank)
        self.model = DDP(self.model, device_ids=[rank])

    def train_step(self, batch):
        """
        One training step.

        Synchronization happens automatically (like Redis Lua script)!
        """
        # Each GPU computes loss on its batch
        embeddings = self.model(batch)
        loss = compute_loss(embeddings)

        # Compute gradients
        loss.backward()

        # DDP automatically:
        # 1. Collects gradients from all GPUs (AllReduce)
        # 2. Averages them
        # 3. Updates all GPUs with averaged gradients
        # This is all atomic and thread-safe!

        self.optimizer.step()

        # All GPUs now have identical model weights
        # (Like all servers seeing same rate limit in Redis)
```

---

## 🎤 Interview Discussion Points

### **Question: "How would you deploy an ML model that handles 10,000 requests/second?"**

**Strong Answer:**

```
"I'd use a multi-tier architecture:

1. Load Balancer
   - Distributes requests across N servers

2. Model Servers (horizontally scaled)
   - Each server has model loaded in memory
   - Use async inference with batching for efficiency
   - PyTorch/TF inference is thread-safe, so multiple threads can handle requests

3. Caching Layer (Redis)
   - Cache predictions for common inputs
   - Significantly reduces inference load
   - Use TTL to handle model updates

4. Monitoring
   - Track request latency (p50, p95, p99)
   - Track throughput
   - Track error rates
   - Use locks for metric counters (thread-safety)

For thread safety:
- Model inference: thread-safe (read-only operations)
- Shared caches/counters: need locks
- Redis for cross-server coordination

Trade-offs:
- More servers = higher throughput but higher cost
- Batching = lower latency per request but higher throughput
- Caching = faster but might serve stale predictions

I'd start with 10-20 servers (500 req/sec each), measure, and scale based on metrics."
```

---

### **Question: "How does distributed training work?"**

**Strong Answer:**

```
"Distributed training is similar to the distributed systems problem we discussed
with rate limiting, but instead of synchronizing token counts, we synchronize
gradients.

Here's how it works:

1. Data Parallelism (most common)
   - Same model replicated across N GPUs
   - Each GPU gets different batch of data
   - Each computes gradients on its batch
   - Gradients are averaged across GPUs (AllReduce operation)
   - All GPUs update with averaged gradients

2. The AllReduce operation is like our Redis Lua script:
   - It's atomic and distributed
   - Prevents race conditions
   - Ensures all GPUs stay synchronized

3. Communication happens via:
   - NCCL (for NVIDIA GPUs) - like Redis for gradients
   - Gloo (CPU) - like Redis
   - MPI (clusters)

4. Key difference from inference:
   - Inference: Read-only, thread-safe
   - Training: Read-write, needs synchronization

In PyTorch, DistributedDataParallel (DDP) handles all this automatically.
It's equivalent to how we'd use Redis for distributed rate limiting - it's
the infrastructure that keeps everything in sync.

Trade-offs:
- Communication overhead increases with more GPUs
- Linear scaling only up to a point (Amdahl's law)
- Need fast interconnects (NVLink, InfiniBand) for efficiency"
```

---

## 📊 Quick Reference: When You Need Locks in ML

| Scenario | Need Lock? | Why |
|----------|-----------|-----|
| **Inference (single request)** | ❌ No | Model weights are read-only |
| **Inference (shared cache)** | ✅ Yes | Cache is read-write |
| **Inference (metrics counter)** | ✅ Yes | Shared mutable state |
| **Training (single GPU)** | ❌ No | Single thread |
| **Training (data loading)** | ❌ No | PyTorch handles it |
| **Training (distributed)** | ✅ Yes* | Handled by DDP/NCCL |
| **Model hot-swap** | ✅ Yes | Swapping shared reference |
| **Batch prediction queue** | ✅ Yes | Shared queue |

*DDP/NCCL handle synchronization automatically

---

## 🎯 Key Takeaways

### **For Inference:**
1. Model prediction is thread-safe (read-only)
2. Shared caches/counters need locks
3. Use async + batching for high throughput
4. Scale horizontally (multiple servers)

### **For Training:**
1. Single GPU: no locks needed
2. Multi-GPU: DDP handles synchronization (like Redis for gradients)
3. Data loading: PyTorch manages threads internally
4. Distributed training = distributed systems problem

### **For Interview:**
1. Know the difference between inference and training
2. Understand when locks are needed (mutable shared state)
3. Know how to scale inference (horizontal scaling + batching)
4. Understand distributed training basics (data parallelism, AllReduce)

---

## 💡 Practice Explaining This

**Try saying out loud:**

```
"For ML inference, the model itself is thread-safe because we're only
reading weights, never writing. Multiple threads can call predict()
simultaneously without locks.

However, if we're maintaining shared state like caches or counters,
we need locks for those specific pieces.

For production serving, I'd use async inference with batching to maximize
GPU utilization, and scale horizontally across multiple servers behind
a load balancer.

For distributed training, it's similar to our distributed rate limiter
problem - instead of synchronizing token counts across servers, we
synchronize gradients across GPUs using AllReduce operations. PyTorch's
DDP handles this automatically, like how we'd use Redis for distributed
coordination."
```

---

**Does this connect the dots? Want me to dive deeper into any specific part?** 🧠
