"""
Problem 4: Product Recommendation System

Build a recommendation engine that:
- Takes user purchase history
- Finds similar products (collaborative filtering or content-based)
- Returns top N recommendations
- Handles cold start problem (new users/products)
- Scalable and efficient

Shopify Context:
- Millions of products
- Real-time recommendations
- Diverse product catalog

Interview Focus:
- Algorithm design (trade-offs between approaches)
- Scalability considerations
- Handling edge cases
- Code organization
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
import heapq


@dataclass
class Product:
    """Product information."""
    product_id: str
    name: str
    category: str
    price: float
    tags: List[str]


@dataclass
class Purchase:
    """User purchase record."""
    user_id: str
    product_id: str
    timestamp: float
    rating: Optional[float] = None


@dataclass
class Recommendation:
    """Recommendation with score."""
    product_id: str
    score: float
    reason: str  # Why this was recommended


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering: "Users who bought X also bought Y"

    Approach: Item-Item similarity
    - For each product, find other products often bought together
    - Recommend products similar to what user already bought

    Trade-offs:
    - Pros: Works without product metadata, captures implicit patterns
    - Cons: Cold start problem, needs sufficient purchase data
    - Complexity: O(n^2) for computing similarities, O(k) for recommendations

    Interview tip: "Collaborative filtering is powerful but needs historical data"
    """

    def __init__(self):
        # Store purchase history
        self.user_purchases: Dict[str, Set[str]] = defaultdict(set)
        self.product_users: Dict[str, Set[str]] = defaultdict(set)

        # Store item-item similarity
        self.item_similarity: Dict[str, Dict[str, float]] = {}

    def add_purchase(self, purchase: Purchase) -> None:
        """
        Add a purchase to the history.

        Interview tip: "Building the user-item matrix incrementally"
        """
        self.user_purchases[purchase.user_id].add(purchase.product_id)
        self.product_users[purchase.product_id].add(purchase.user_id)

    def compute_similarities(self) -> None:
        """
        Compute item-item similarities using Jaccard similarity.

        Jaccard Similarity = |users who bought both| / |users who bought either|

        Interview tip: "Computing similarities offline for faster recommendations"

        Trade-off: Jaccard vs Cosine vs Pearson
        - Jaccard: Simple, works for binary data (bought/not bought)
        - Cosine: Better with ratings
        - Pearson: Accounts for user rating bias

        For purchase data (no ratings), Jaccard is sufficient.
        """
        print("Computing item-item similarities...")

        products = list(self.product_users.keys())

        for i, product_a in enumerate(products):
            self.item_similarity[product_a] = {}

            users_a = self.product_users[product_a]

            for product_b in products[i+1:]:  # Only upper triangle (symmetric)
                users_b = self.product_users[product_b]

                # Jaccard similarity
                intersection = len(users_a & users_b)
                union = len(users_a | users_b)

                if union > 0:
                    similarity = intersection / union

                    # Only store if similarity is significant
                    if similarity > 0.1:  # Threshold to save memory
                        self.item_similarity[product_a][product_b] = similarity
                        # Symmetric
                        if product_b not in self.item_similarity:
                            self.item_similarity[product_b] = {}
                        self.item_similarity[product_b][product_a] = similarity

        print(f"Computed similarities for {len(products)} products")

    def recommend(self, user_id: str, n: int = 5,
                 exclude_purchased: bool = True) -> List[Recommendation]:
        """
        Recommend products for a user.

        Algorithm:
        1. Get products user has purchased
        2. For each purchased product, find similar products
        3. Aggregate scores
        4. Return top N

        Interview tip: "Explaining the recommendation logic step by step"
        """
        if user_id not in self.user_purchases:
            # Cold start: new user
            return self._cold_start_recommendations(n)

        user_items = self.user_purchases[user_id]

        # Aggregate scores for candidate products
        candidate_scores: Dict[str, float] = defaultdict(float)

        for purchased_item in user_items:
            if purchased_item in self.item_similarity:
                for similar_item, similarity in self.item_similarity[purchased_item].items():
                    # Skip if user already purchased
                    if exclude_purchased and similar_item in user_items:
                        continue

                    candidate_scores[similar_item] += similarity

        # Get top N
        top_items = heapq.nlargest(
            n,
            candidate_scores.items(),
            key=lambda x: x[1]
        )

        recommendations = [
            Recommendation(
                product_id=item_id,
                score=score,
                reason=f"Similar to products you purchased (score: {score:.2f})"
            )
            for item_id, score in top_items
        ]

        return recommendations

    def _cold_start_recommendations(self, n: int) -> List[Recommendation]:
        """
        Handle cold start: recommend popular products.

        Interview tip: "For new users, fall back to popularity-based recommendations"
        """
        # Count how many users bought each product
        popularity = [(product_id, len(users))
                     for product_id, users in self.product_users.items()]

        # Sort by popularity
        popularity.sort(key=lambda x: x[1], reverse=True)

        recommendations = [
            Recommendation(
                product_id=product_id,
                score=float(count),
                reason=f"Popular product (purchased by {count} users)"
            )
            for product_id, count in popularity[:n]
        ]

        return recommendations


class ContentBasedRecommender:
    """
    Content-Based Filtering: "Products similar to what you bought"

    Approach: Use product features (category, tags, price)
    - Build product feature vectors
    - Find similar products using cosine similarity

    Trade-offs:
    - Pros: No cold start for products, works with limited purchase data
    - Cons: Limited diversity (recommends very similar items)
    - Complexity: O(n) for recommendations (with indexed features)

    Interview tip: "Content-based works well when you have rich product metadata"
    """

    def __init__(self):
        self.products: Dict[str, Product] = {}

        # Index for fast lookup
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

    def add_product(self, product: Product) -> None:
        """Add product to the catalog."""
        self.products[product.product_id] = product

        # Build indices
        self.category_index[product.category].add(product.product_id)
        for tag in product.tags:
            self.tag_index[tag].add(product.product_id)

    def compute_similarity(self, product_a: Product, product_b: Product) -> float:
        """
        Compute similarity between two products.

        Features:
        - Category match: +0.5
        - Tag overlap: +0.1 per shared tag
        - Price similarity: +0.2 if within 20%

        Interview tip: "Feature weights can be tuned based on performance"
        """
        score = 0.0

        # Category match (strong signal)
        if product_a.category == product_b.category:
            score += 0.5

        # Tag overlap (moderate signal)
        tags_a = set(product_a.tags)
        tags_b = set(product_b.tags)
        tag_overlap = len(tags_a & tags_b)
        score += tag_overlap * 0.1

        # Price similarity (weak signal)
        if product_a.price > 0 and product_b.price > 0:
            price_ratio = min(product_a.price, product_b.price) / max(product_a.price, product_b.price)
            if price_ratio > 0.8:  # Within 20%
                score += 0.2

        return score

    def find_similar_products(self, product_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Find N most similar products.

        Optimization: Only compare within same category (reduces search space)

        Interview tip: "Using category as a filter to improve efficiency"
        """
        if product_id not in self.products:
            return []

        target_product = self.products[product_id]

        # Only compare within same category (optimization)
        candidate_ids = self.category_index[target_product.category]

        similarities = []
        for candidate_id in candidate_ids:
            if candidate_id == product_id:
                continue

            candidate_product = self.products[candidate_id]
            similarity = self.compute_similarity(target_product, candidate_product)

            if similarity > 0:  # Only include if some similarity
                similarities.append((candidate_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:n]

    def recommend(self, user_purchases: List[str], n: int = 5) -> List[Recommendation]:
        """
        Recommend products based on user's purchase history.

        Algorithm:
        1. For each purchased product, find similar products
        2. Aggregate scores
        3. Return top N

        Interview tip: "Aggregating across multiple purchased items"
        """
        candidate_scores: Dict[str, float] = defaultdict(float)

        for purchased_id in user_purchases:
            similar_products = self.find_similar_products(purchased_id, n=20)

            for similar_id, similarity in similar_products:
                # Skip if already purchased
                if similar_id in user_purchases:
                    continue

                candidate_scores[similar_id] += similarity

        # Get top N
        top_items = heapq.nlargest(
            n,
            candidate_scores.items(),
            key=lambda x: x[1]
        )

        recommendations = [
            Recommendation(
                product_id=item_id,
                score=score,
                reason=f"Similar to your purchases (score: {score:.2f})"
            )
            for item_id, score in top_items
        ]

        return recommendations


class HybridRecommender:
    """
    Hybrid Recommender: Combines collaborative and content-based.

    Strategy: Weighted average of both approaches
    - Collaborative: 70% weight (captures user behavior)
    - Content-based: 30% weight (adds diversity, handles cold start)

    Interview tip: "Hybrid systems combine strengths of multiple approaches"

    Trade-off: Complexity vs Performance
    - More complex to build and maintain
    - Better recommendations
    - Can tune weights based on A/B testing
    """

    def __init__(self, cf_recommender: CollaborativeFilteringRecommender,
                 cb_recommender: ContentBasedRecommender,
                 cf_weight: float = 0.7):
        """
        Initialize hybrid recommender.

        Args:
            cf_recommender: Collaborative filtering recommender
            cb_recommender: Content-based recommender
            cf_weight: Weight for collaborative filtering (0-1)
        """
        self.cf = cf_recommender
        self.cb = cb_recommender
        self.cf_weight = cf_weight
        self.cb_weight = 1 - cf_weight

    def recommend(self, user_id: str, n: int = 5) -> List[Recommendation]:
        """
        Generate hybrid recommendations.

        Interview tip: "Combining scores from multiple sources"
        """
        # Get recommendations from both systems
        cf_recs = self.cf.recommend(user_id, n=n*2)  # Get more for diversity

        # For content-based, need user's purchase history
        user_purchases = list(self.cf.user_purchases.get(user_id, []))
        cb_recs = self.cb.recommend(user_purchases, n=n*2) if user_purchases else []

        # Combine scores
        combined_scores: Dict[str, float] = defaultdict(float)

        for rec in cf_recs:
            combined_scores[rec.product_id] += rec.score * self.cf_weight

        for rec in cb_recs:
            combined_scores[rec.product_id] += rec.score * self.cb_weight

        # Get top N
        top_items = heapq.nlargest(
            n,
            combined_scores.items(),
            key=lambda x: x[1]
        )

        recommendations = [
            Recommendation(
                product_id=item_id,
                score=score,
                reason=f"Hybrid recommendation (CF: {self.cf_weight}, CB: {self.cb_weight})"
            )
            for item_id, score in top_items
        ]

        return recommendations


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PRODUCT RECOMMENDATION SYSTEM")
    print("="*60 + "\n")

    # Create sample products
    products = [
        Product("p1", "Laptop", "Electronics", 1000, ["computer", "work"]),
        Product("p2", "Mouse", "Electronics", 25, ["computer", "accessories"]),
        Product("p3", "Keyboard", "Electronics", 75, ["computer", "accessories"]),
        Product("p4", "Monitor", "Electronics", 300, ["computer", "display"]),
        Product("p5", "Headphones", "Electronics", 150, ["audio", "accessories"]),
        Product("p6", "Desk Lamp", "Furniture", 50, ["office", "lighting"]),
        Product("p7", "Office Chair", "Furniture", 200, ["office", "seating"]),
    ]

    # Create sample purchases
    purchases = [
        Purchase("user1", "p1", 1.0),  # Laptop
        Purchase("user1", "p2", 2.0),  # Mouse
        Purchase("user2", "p1", 1.5),  # Laptop
        Purchase("user2", "p3", 2.5),  # Keyboard
        Purchase("user3", "p1", 1.0),  # Laptop
        Purchase("user3", "p2", 1.5),  # Mouse
        Purchase("user3", "p4", 2.0),  # Monitor
        Purchase("user4", "p6", 1.0),  # Desk Lamp
        Purchase("user4", "p7", 2.0),  # Office Chair
    ]

    # Demo 1: Collaborative Filtering
    print("1. COLLABORATIVE FILTERING")
    print("-" * 40)

    cf_recommender = CollaborativeFilteringRecommender()

    # Add purchases
    for purchase in purchases:
        cf_recommender.add_purchase(purchase)

    # Compute similarities
    cf_recommender.compute_similarities()

    # Recommend for user1 (bought laptop and mouse)
    print("\nUser1 bought: Laptop, Mouse")
    recs = cf_recommender.recommend("user1", n=3)
    print("Recommendations:")
    for rec in recs:
        print(f"  - Product {rec.product_id}: {rec.reason}")

    # Demo 2: Content-Based
    print("\n\n2. CONTENT-BASED FILTERING")
    print("-" * 40)

    cb_recommender = ContentBasedRecommender()

    # Add products
    for product in products:
        cb_recommender.add_product(product)

    # Recommend for user who bought laptop
    print("\nUser bought: Laptop")
    recs = cb_recommender.recommend(["p1"], n=3)
    print("Recommendations:")
    for rec in recs:
        product = cb_recommender.products[rec.product_id]
        print(f"  - {product.name} ({product.category}): {rec.reason}")

    # Demo 3: Hybrid
    print("\n\n3. HYBRID RECOMMENDER")
    print("-" * 40)

    hybrid_recommender = HybridRecommender(cf_recommender, cb_recommender)

    print("\nUser1 bought: Laptop, Mouse")
    recs = hybrid_recommender.recommend("user1", n=3)
    print("Hybrid Recommendations:")
    for rec in recs:
        product = cb_recommender.products.get(rec.product_id)
        if product:
            print(f"  - {product.name}: {rec.reason}")

    # Demo 4: Cold Start
    print("\n\n4. COLD START (New User)")
    print("-" * 40)

    print("\nNew user (no purchase history)")
    recs = cf_recommender.recommend("new_user", n=3)
    print("Recommendations (fallback to popular):")
    for rec in recs:
        print(f"  - Product {rec.product_id}: {rec.reason}")

    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    print("""
    1. COLLABORATIVE FILTERING:
       Pros:
       - No product metadata needed
       - Captures implicit patterns
       - Provides serendipitous recommendations

       Cons:
       - Cold start problem (new users/products)
       - Requires significant data
       - Sparsity issues (most users buy few products)

       Best for: Mature platforms with lots of data

    2. CONTENT-BASED:
       Pros:
       - Works with limited data
       - No cold start for products
       - Explainable recommendations

       Cons:
       - Requires rich metadata
       - Limited diversity (filter bubble)
       - Doesn't capture behavior patterns

       Best for: New platforms, diverse catalogs

    3. HYBRID:
       Pros:
       - Combines strengths of both
       - Better recommendations
       - Handles cold start better

       Cons:
       - More complex
       - Requires tuning weights

       Best for: Production systems

    SHOPIFY CONSIDERATIONS:
    - Millions of products → Need efficient indexing
    - Real-time recommendations → Pre-compute similarities
    - Diverse catalog → Content-based helps with diversity
    - Scale → Use approximate nearest neighbors (ANN) like FAISS
    """)

    print("\n" + "="*60)
    print("PRODUCTION OPTIMIZATIONS")
    print("="*60)
    print("""
    1. PRE-COMPUTATION:
       - Compute item-item similarities offline (daily batch job)
       - Store in cache/database for fast lookup
       - Trade-off: Freshness vs speed

    2. INDEXING:
       - Use inverted indices for tags/categories
       - Use approximate nearest neighbors (FAISS, Annoy)
       - Trade-off: Accuracy vs speed

    3. CACHING:
       - Cache user recommendations (TTL: 5-10 minutes)
       - Cache product similarities
       - Trade-off: Memory vs speed

    4. FILTERING:
       - Filter out out-of-stock products
       - Apply business rules (margins, promotions)
       - Personalize based on user tier

    5. A/B TESTING:
       - Test different algorithms
       - Test different weights in hybrid
       - Measure: CTR, conversion rate, revenue

    6. SCALABILITY:
       - Use matrix factorization (SVD, ALS) for large scale
       - Distributed computation (Spark)
       - Model serving infrastructure (TensorFlow Serving)
    """)
