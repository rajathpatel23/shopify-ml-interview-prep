# Shopify Senior ML Engineer - Interview Day Guide

## Quick Reference (Keep this open during interview)

### 1. Opening (First 5 minutes)

**Always start with:**
```
"Let me make sure I understand the requirements..."
[Restate the problem in your own words]

"I have a few clarifying questions:"
- What format is the input data?
- Are there any constraints (time/memory)?
- What edge cases should I handle?
- What should happen if [edge case]?
- Are we optimizing for speed or readability?
```

### 2. Planning (5-10 minutes)

**Think aloud:**
```
"I'm thinking we could approach this by:
1. [Step 1]
2. [Step 2]
3. [Step 3]

The trade-off here is [X vs Y]...

Does this approach make sense?"
```

**Get buy-in before coding!**

### 3. While Coding (40-50 minutes)

**Narrate constantly:**
- "I'm creating this function to handle..."
- "I'm using a dictionary here because O(1) lookup..."
- "Let me add validation to handle null values..."
- "I'm going to use Cursor to help with..."

**Using AI (they encourage it!):**
- "Let me ask Cursor to generate the boilerplate..."
- "I'll use AI to check this regex pattern..."
- Then EXPLAIN the generated code!

**If stuck:**
- "I'm thinking through two approaches - X vs Y"
- "What do you think about this direction?"
- DON'T go silent!

### 4. Testing (10-15 minutes)

**Walk through examples:**
```
"Let me test this with a few cases:
- Normal case: [example]
- Edge case: empty input
- Edge case: null values
- Edge case: very large input"
```

**Discuss improvements:**
- "Given more time, I'd add..."
- "For production, we'd need..."
- "We could optimize this by..."

---

## Common Patterns to Know

### Input Validation
```python
def process_data(data: List[Dict]) -> pd.DataFrame:
    """Process user data."""
    # 1. Check for None/empty
    if not data:
        raise ValueError("Data cannot be empty")

    # 2. Validate structure
    required_fields = ["user_id", "timestamp"]
    for item in data:
        if not all(field in item for field in required_fields):
            raise ValueError(f"Missing required fields")

    # 3. Validate types/values
    for item in data:
        if not isinstance(item["timestamp"], (int, float)):
            raise ValueError("Invalid timestamp")
```

### Error Handling
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Failed: {e}")
    # Decide: fail fast or continue?
    raise  # or return default value
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise CustomError(f"Operation failed: {e}")
```

### Caching Pattern
```python
def predict(features: Dict) -> float:
    # Check cache first
    cache_key = create_key(features)
    if cache_key in cache:
        return cache[cache_key]

    # Compute
    result = expensive_computation(features)

    # Store in cache
    cache[cache_key] = result
    return result
```

---

## Key Trade-offs to Mention

### 1. Time vs Space
- "We could cache this for O(1) lookup, but it uses more memory"
- "Sorting takes O(n log n) but simplifies the logic"

### 2. Accuracy vs Performance
- "We could use exact algorithm (slower) or approximate (faster)"
- "Increasing cache size improves hit rate but uses more memory"

### 3. Complexity vs Maintainability
- "This could be one-lined, but separate functions are clearer"
- "Adding abstraction makes it extensible but more complex"

### 4. Synchronous vs Asynchronous
- "Sync is simpler, async handles more concurrent requests"
- "For high load, we'd use message queue (Celery/RabbitMQ)"

### 5. Fail Fast vs Graceful Degradation
- "Strict validation fails early but rejects valid-ish data"
- "Lenient handling is more robust but might hide issues"

---

## Python Best Practices

### Type Hints (Use them!)
```python
from typing import List, Dict, Optional, Union

def process(
    data: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Optional[pd.DataFrame]:
    """
    Process data and return DataFrame.

    Args:
        data: List of records
        threshold: Filter threshold

    Returns:
        Processed DataFrame or None if empty
    """
```

### Logging (Show you think about observability)
```python
import logging
logger = logging.getLogger(__name__)

def predict(features):
    logger.info(f"Making prediction for {len(features)} features")
    try:
        result = model.predict(features)
        logger.info(f"Prediction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
```

### Testing Mindset
```python
# While coding, mention:
"I'd write tests for:
- Normal case with valid input
- Edge case: empty data
- Edge case: missing fields
- Edge case: invalid types
- Error handling: network failure
"
```

---

## Common ML Engineering Questions

### 1. Data Processing
- Handling missing values (drop vs impute)
- Feature scaling (normalize vs standardize)
- Encoding categoricals (one-hot vs label)

### 2. Model Serving
- Prediction API design
- Caching strategies
- Error handling
- Performance monitoring

### 3. System Design
- Rate limiting
- Caching layers
- Load balancing
- Scaling strategies

---

## Communication Checklist

Before interview:
- [ ] Test screen share with line numbers visible
- [ ] Test audio/video
- [ ] Close unnecessary apps
- [ ] Have water ready
- [ ] IDE ready (Cursor with AI enabled)
- [ ] GitHub credentials ready

During interview:
- [ ] Restate problem before starting
- [ ] Ask clarifying questions
- [ ] Discuss approach before coding
- [ ] Think aloud constantly
- [ ] Use AI tools naturally
- [ ] Test with examples
- [ ] Mention trade-offs
- [ ] Stay calm if stuck

After coding:
- [ ] Walk through test cases
- [ ] Discuss improvements
- [ ] Mention production considerations
- [ ] Ask if they want to see anything else

---

## Phrases to Use

**Understanding:**
- "Let me make sure I understand..."
- "Just to clarify..."
- "Would it be correct to assume...?"

**Planning:**
- "I'm thinking we could approach this by..."
- "The trade-off here is..."
- "We have a few options: X or Y..."

**Coding:**
- "I'm adding this validation because..."
- "Using a dictionary here for O(1) lookup..."
- "Let me use Cursor to help with..."
- "I'm thinking about the edge case where..."

**Stuck:**
- "I'm weighing two approaches..."
- "What do you think about...?"
- "Let me think through this out loud..."

**Testing:**
- "Let's test with a few cases..."
- "An edge case I'm thinking about is..."
- "This should handle..."

**Production:**
- "For production, we'd need..."
- "We could optimize this by..."
- "In a real system, we'd also..."

---

## Don't Forget

1. **Communication > Completion** - They said it in the guide!
2. **Use AI naturally** - They want to see your workflow
3. **Ask questions** - Shows thoughtfulness
4. **Explain trade-offs** - Shows senior thinking
5. **Handle errors** - Production mindset
6. **Think about scale** - Not just correctness
7. **Be yourself** - They want to work with you!

---

## If You Get Stuck

1. **Don't panic** - Take a breath
2. **Think aloud** - "I'm considering..."
3. **Ask for help** - "What do you think about...?"
4. **Simplify** - "Let me start with a simpler version..."
5. **Pivot if needed** - "Maybe a different approach..."

Remember: They're evaluating how you work, not just if you finish!

---

## Last Minute Checklist (30 min before)

```bash
# 1. Test screen share
# Open Cursor, join Google Meet test, share screen

# 2. Check line numbers
# Cmd+, → search "line numbers" → enable

# 3. Test GitHub
ssh -T git@github.com

# 4. Test AI tools
# Open Cursor, test Cmd+K and Cmd+L

# 5. Close distractions
# Slack, email, notifications OFF

# 6. Environment ready
# Water, bathroom, quiet space

# 7. Open this guide (off-screen but visible)
```

**You've got this! 🚀**
