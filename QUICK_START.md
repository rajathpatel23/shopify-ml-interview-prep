# Quick Start Guide

## Setup (Do this NOW - 10 minutes)

```bash
# 1. Navigate to the repository
cd ~/work/shopify-ml-interview-prep

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test that everything works
python problem_1_feature_pipeline/feature_pipeline.py
```

You should see the feature pipeline running with sample data.

## What You Have

### 5 Complete Production-Quality Problems:

1. **Feature Engineering Pipeline** (`problem_1_feature_pipeline/`)
   - Data loading, validation, transformation
   - Handles missing values, encoding, normalization
   - Time: Practice in 45 minutes

2. **Model Prediction Service** (`problem_2_prediction_service/`)
   - Production ML service with caching
   - Error handling, monitoring, health checks
   - Time: Practice in 60 minutes

3. **Product Recommendation System** (`problem_4_recommendations/`)
   - Collaborative filtering + Content-based
   - Handles cold start, scales to large catalogs
   - Time: Practice in 75 minutes (full interview!)

4. **Rate Limiter** (`problem_7_rate_limiter/`)
   - Token Bucket, Sliding Window, Fixed Window algorithms
   - Thread-safe, production-ready
   - Time: Practice in 45 minutes

5. **Cache System** (`problem_8_cache_system/`)
   - LRU and LFU eviction policies
   - TTL support, thread-safe
   - Time: Practice in 45 minutes

### 3 Essential Guides:

1. **INTERVIEW_GUIDE.md** - Keep this open during the interview!
   - Communication checklist
   - Common patterns
   - Trade-offs to mention
   - Phrases to use

2. **PRACTICE_SCHEDULE.md** - Your 5-day plan
   - Day-by-day breakdown
   - Time allocations
   - What to focus on each day

3. **README.md** - Overview and setup instructions

## How to Practice

### Today (Thursday) - Run Problem 1

```bash
# 1. Read the problem description at the top of the file
cat problem_1_feature_pipeline/feature_pipeline.py | head -20

# 2. Run the example to see how it works
python problem_1_feature_pipeline/feature_pipeline.py

# 3. Now try to code it yourself from scratch!
# Create a new file and implement the same functionality
# Talk out loud the entire time
# Time yourself: 45 minutes
```

### Key Practice Tips

**1. Always Talk Out Loud:**
```
"I'm creating this function to handle missing values...
The trade-off here is between dropping rows vs imputing...
I'll use mean imputation because it preserves all data...
Let me add validation to handle edge cases..."
```

**2. Use AI Naturally:**
```
"Let me use Cursor to help with the pandas syntax..."
[Cmd+K to get AI help]
"The AI suggested this approach, which makes sense because..."
```

**3. Think About Trade-offs:**
- Time vs Space complexity
- Accuracy vs Performance
- Simplicity vs Flexibility
- Fail Fast vs Graceful Degradation

**4. Test With Examples:**
```python
# Always test edge cases
test_cases = [
    {"normal": "standard input"},
    {"edge": "empty list"},
    {"edge": "null values"},
    {"edge": "invalid types"}
]
```

## Run All Examples

```bash
# Feature Engineering
python problem_1_feature_pipeline/feature_pipeline.py

# Prediction Service
python problem_2_prediction_service/prediction_service.py

# Recommendations
python problem_4_recommendations/recommendation_system.py

# Rate Limiter
python problem_7_rate_limiter/rate_limiter.py

# Cache System
python problem_8_cache_system/cache_system.py
```

Each will run with sample data and show expected output.

## For Your Interview (Nov 19)

**30 Minutes Before:**
1. Test Google Meet screen share with Cursor
2. Verify line numbers are visible
3. Test GitHub SSH: `ssh -T git@github.com`
4. Close all apps except Cursor and browser
5. Open INTERVIEW_GUIDE.md (off-screen)
6. Take 3 deep breaths - you've got this!

**During Interview:**
1. ✓ Restate the problem
2. ✓ Ask clarifying questions
3. ✓ Discuss approach BEFORE coding
4. ✓ Think aloud constantly
5. ✓ Use AI naturally
6. ✓ Test with examples

## Need Help?

All code includes:
- Detailed comments explaining the thought process
- Interview tips inline
- Trade-off discussions
- Edge case handling
- Production considerations

Just read through the code and comments - they're designed as a learning resource!

## Repository URL

https://github.com/rajathpatel23/shopify-ml-interview-prep

Clone anywhere:
```bash
git clone https://github.com/rajathpatel23/shopify-ml-interview-prep.git
```

---

**Remember:** Communication > Completion. They want to see how you think and work!

Good luck! 🚀
