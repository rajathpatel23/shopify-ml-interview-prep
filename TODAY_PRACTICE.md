# Your First Practice Session - TODAY (Nov 14)

## ✅ Setup Complete!

You've successfully:
- [x] Created virtual environment
- [x] Installed all dependencies
- [x] Tested all problems run correctly
- [x] Repository pushed to GitHub

**GitHub URL:** https://github.com/rajathpatel23/shopify-ml-interview-prep

---

## 🎯 Practice Session 1 - Feature Engineering Pipeline (Next 60 min)

### Step 1: Understand the Problem (10 min)

**Read the problem carefully:**
```bash
# Open the file in Cursor
code problem_1_feature_pipeline/feature_pipeline.py
```

**Key requirements:**
1. Load data from JSON
2. Handle missing values
3. Encode categorical variables
4. Normalize numerical features
5. Make it reusable (fit/transform pattern)

### Step 2: Run the Example (5 min)

```bash
source venv/bin/activate
python problem_1_feature_pipeline/feature_pipeline.py
```

**Observe:**
- How does it handle missing values?
- What happens with categorical encoding?
- How does normalization work?
- What edge cases are handled?

### Step 3: Code It Yourself (45 min)

**Create a new file:**
```bash
touch practice_session_1.py
code practice_session_1.py
```

**Your task:**
```python
"""
Build a FeaturePipeline class that:
1. Loads JSON data
2. Handles missing values (mean for numerical, mode for categorical)
3. Encodes categorical variables
4. Normalizes numerical features

Test with this data:
data = [
    {"age": 25, "income": 50000, "category": "A"},
    {"age": None, "income": 60000, "category": "B"},
    {"age": 35, "income": None, "category": "A"},
]

Expected behavior:
- Missing age filled with mean (30)
- Missing income filled with mean (55000)
- Categories encoded to numbers
- Features normalized (StandardScaler)
"""
```

**IMPORTANT - Practice Interview Skills:**

1. **Talk Out Loud** (This is critical!)
   ```
   "Okay, let me start by understanding what we need to build...

   I see we need to handle missing values. The trade-off here is
   between dropping rows vs imputing. I'll choose imputation to
   preserve all data...

   For categorical encoding, I'm using LabelEncoder because it's
   simple and works for tree-based models. In production, we might
   use one-hot encoding for linear models...

   Let me add validation here to handle edge cases like empty data..."
   ```

2. **Use Cursor/AI Naturally**
   - Cmd+K for inline suggestions
   - Cmd+L for chat
   - Say: "Let me use Cursor to help with the pandas syntax..."
   - Then EXPLAIN what the AI suggested

3. **Think About Edge Cases**
   - What if data is empty?
   - What if a column is all NaN?
   - What if we see new categories during transform?
   - What if columns are missing?

4. **Test As You Go**
   ```python
   # After each function, test it
   if __name__ == "__main__":
       # Test with simple data first
       # Then test edge cases
   ```

### Step 4: Compare With Solution (After your attempt)

After you've coded your solution, compare with the provided code:
- What did you do differently?
- What edge cases did you miss?
- How is the code organized?
- What can you learn for next time?

---

## 🎯 Practice Session 2 - Prediction Service (Next 60 min)

### The Problem

Build a production-ready ML prediction service with:
- Input validation
- Caching (for performance)
- Error handling
- Monitoring (metrics)
- Health check endpoint

### Your Approach

```python
"""
Create a PredictionService class:

Requirements:
1. Validate input features
2. Make predictions (use a mock model)
3. Cache predictions (avoid recomputing)
4. Handle errors gracefully
5. Track metrics (prediction count, errors)
6. Provide health check

Think about:
- What if input is invalid? (raise ValidationError)
- What if prediction fails? (log and raise PredictionError)
- How to cache? (dict with key from features)
- Thread safety? (not required for now, but mention it)
"""
```

**Interview Discussion Points:**

1. **Caching Trade-offs:**
   - "I'm implementing caching because repeated predictions are common
     in production. The trade-off is memory usage vs speed..."
   - "TTL ensures we don't serve stale predictions..."

2. **Error Handling:**
   - "I'm using custom exceptions to make errors more specific..."
   - "For batch predictions, I'm continuing even if one fails,
     which gives partial results vs failing completely..."

3. **Validation:**
   - "Validating early fails fast and prevents bad data from
     reaching the model..."

4. **Production Thinking:**
   - "For production, I'd add:
     - Async processing for high load
     - Distributed caching with Redis
     - Proper monitoring (Datadog, Prometheus)
     - Rate limiting per user"

---

## 🎤 Communication Checklist

While coding, make sure you:
- [ ] Explain your approach before coding
- [ ] Think aloud constantly
- [ ] Explain trade-offs (time vs space, accuracy vs performance)
- [ ] Mention edge cases
- [ ] Use AI and explain what it suggested
- [ ] Test with examples
- [ ] Discuss production considerations

**Phrases to use:**
- "I'm thinking about..."
- "The trade-off here is..."
- "An edge case I need to handle is..."
- "Let me use Cursor to help with..."
- "For production, we'd also need..."

---

## 📊 After Both Sessions

### Self-Review Questions:

1. **Communication:**
   - Did I explain my thinking out loud?
   - Did I ask clarifying questions (to myself)?
   - Did I explain trade-offs?

2. **Code Quality:**
   - Is my code readable?
   - Did I use type hints?
   - Did I handle errors?
   - Did I add docstrings?

3. **Edge Cases:**
   - Empty data?
   - Missing values?
   - Invalid types?
   - Unexpected values?

4. **Time Management:**
   - Did I finish in time?
   - Did I get stuck anywhere?
   - What would I do differently?

### Recording Exercise (Optional but Powerful)

If you have time, record yourself:
1. Pick Problem 1 or 2
2. Start fresh in a new file
3. Record your screen with audio
4. Code while talking out loud (45 min)
5. Watch the recording - where did you go silent?

This is THE BEST way to improve your communication!

---

## 🌙 Tonight

- [ ] Review what you practiced today
- [ ] Note what went well
- [ ] Note what to improve
- [ ] Get good sleep
- [ ] DON'T code late into the night

---

## 📅 Tomorrow (Friday, Nov 15)

Morning:
- Rate Limiter System
- Cache System

Afternoon:
- Recommendation System (75 min - full interview length!)

Follow PRACTICE_SCHEDULE.md for the full plan.

---

## 💡 Tips from Today

**Most Important Lesson:**
Communication > Perfect Code

**What Shopify Wants to See:**
- How you think
- How you approach problems
- How you handle ambiguity
- How you collaborate
- How you use tools (AI!)

**Remember:**
- It's okay to not finish
- It's okay to ask for help
- It's NOT okay to go silent
- It's NOT okay to not explain your thinking

---

**You're doing great! Keep practicing and you'll crush this interview!** 🚀

Time to start coding! Open `practice_session_1.py` and begin! ⏰
