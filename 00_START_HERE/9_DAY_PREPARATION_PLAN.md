# 9-Day Preparation Plan for Shopify Interview
**Interview Date: November 25, 2025**

## Current Status (Nov 16)
- ✓ Coding ability: 7/10
- ✓ Planning & structure: 9/10
- ✓ Testing: 9/10
- ✗ **Communication: 3/10** ← Primary focus

**Goal:** Get communication to 9/10 in 9 days

---

## Week 1: Foundation & Communication Training (Nov 16-20)

### **Day 1 - Today (Nov 16): Rate Limiter Redux**
**Goal:** Master talking while coding

**Morning Session (2 hours):**
1. Re-read `03_guides/PRACTICE_2_FEEDBACK.md`
2. Fix the 5 bugs in your `practice_session_2_proper.py`
3. Compare with reference solution

**Afternoon Session (2.5 hours):**
4. **NEW PRACTICE:** Implement rate limiter from scratch
   - Record yourself (video or audio)
   - Set timer for 45 minutes
   - **Talk constantly** - pretend I'm sitting next to you
   - Use Cursor like you will in interview

5. **Review recording (30 min):**
   - Count silent periods >30 seconds
   - Write down what you should have said
   - Note: Where did you get stuck?

**Evening (30 min):**
6. Read `problem_8_cache_system/cache_system.py`
7. Review LRU eviction algorithm

**Success Criteria:**
- [ ] Fixed all 5 bugs from practice session 2
- [ ] Recorded yourself coding
- [ ] Silent <50% of time (down from ~70%)
- [ ] Working rate limiter in 45 min

---

### **Day 2 (Nov 17): Cache System**
**Goal:** Reduce silence to <30%

**Session (3 hours):**
1. **Practice Problem 8 - LRU Cache**
   - Record yourself again
   - 45-minute timer
   - Focus on narration:
     - "I'm using OrderedDict because..."
     - "This will be O(1) because..."
     - "I need a lock here because..."

2. **Write test cases as you code:**
   ```python
   # Test 1: Basic get/put
   cache = LRUCache(capacity=2)
   cache.put("a", 1)
   cache.put("b", 2)
   assert cache.get("a") == 1

   # Test 2: Eviction
   cache.put("c", 3)  # Should evict "b"
   assert cache.get("b") is None
   ```

3. **Review recording:**
   - Mark every time you're silent >20 seconds
   - For each silence: what were you thinking? Say it next time!

**Evening (1 hour):**
4. Watch your Day 1 and Day 2 recordings back-to-back
5. Note improvements
6. Read `problem_4_recommendations/recommendation_system.py`

**Success Criteria:**
- [ ] Working LRU cache in 45 min
- [ ] Silent <30% of time
- [ ] At least 5 test cases written incrementally
- [ ] Used lock correctly

---

### **Day 3 (Nov 18): ML Problem - Feature Pipeline**
**Goal:** Practice ML problem + communication

**Session (3 hours):**
1. **Practice Problem 1 - Feature Pipeline**
   - This is the one you did before, but do it BETTER
   - Record yourself
   - Focus on explaining ML concepts:
     - "StandardScaler needs 2D array because it expects (n_samples, n_features)"
     - "I'm using fit/transform pattern like sklearn for consistency"
     - "LabelEncoder is for categorical features with no order"

2. **Test with realistic data:**
   ```python
   import pandas as pd
   df = pd.DataFrame({
       'age': [25, 30, None, 40],
       'city': ['NYC', 'SF', 'LA', 'NYC'],
       'income': [50000, 75000, 60000, 80000]
   })
   ```

3. **Discuss trade-offs:**
   - Why StandardScaler vs MinMaxScaler?
   - What about missing values - mean vs median?
   - How to handle new categories at inference?

**Evening (1 hour):**
4. Review recording - focus on ML explanations
5. Read `problem_3_data_processing/data_pipeline.py`

**Success Criteria:**
- [ ] No bugs (learned from practice session 1)
- [ ] Silent <20% of time
- [ ] Explained ML concepts clearly
- [ ] Handled missing values correctly

---

### **Day 4 (Nov 19): Data Processing & Pandas**
**Goal:** Master pandas operations

**Session (3 hours):**
1. **Practice Problem 3 - Data Pipeline**
   - Focus on explaining pandas operations:
     - "I'm using groupby().size() to count events per day"
     - "pivot() reshapes data from long to wide format"
     - "I'm sorting by timestamp first because time-series data needs chronological order"

2. **Implement one anomaly detection method:**
   - Just z-score is fine (don't need all 3)
   - Explain: "Z-score measures how many standard deviations from mean"
   - Test with synthetic data that has obvious anomaly

3. **Talk about scalability:**
   - "For millions of rows, I'd use chunking: pd.read_csv(chunksize=10000)"
   - "Parquet is faster than CSV for large files"
   - "For distributed processing, I'd consider Dask or Spark"

**Evening (1 hour):**
4. Review all your recordings from Days 1-4
5. Note pattern: Are you still getting stuck in same places?
6. Read `problem_6_api_integration/api_service.py`

**Success Criteria:**
- [ ] Explained pandas operations clearly
- [ ] Working anomaly detection
- [ ] Discussed scalability
- [ ] Silent <20% of time

---

### **Day 5 (Nov 20): API Integration - Combining Concepts**
**Goal:** Integrate multiple patterns (cache + rate limit)

**Session (3 hours):**
1. **Practice Problem 6 - API Client**
   - This combines caching, rate limiting, retry logic
   - You've practiced these separately, now together
   - Focus on explaining the flow:
     - "Request flow: check cache → check rate limit → make request → cache response"
     - "Only cache GET requests because they're idempotent"
     - "Exponential backoff: 100ms, 200ms, 400ms to avoid overwhelming failed service"

2. **Implement core version (~100 lines):**
   - Basic APIClient class
   - Token bucket rate limiter (you know this!)
   - Simple LRU cache (you know this!)
   - Retry logic with backoff

3. **Test edge cases:**
   - What if cache is full?
   - What if rate limit exceeded?
   - What if all retries fail?

**Evening (1 hour):**
4. Watch all 5 days of recordings (on 2x speed)
5. Calculate: % time silent across all sessions
6. Target: Should be trending down: 70% → 50% → 30% → 20% → 15%

**Success Criteria:**
- [ ] Integrated cache + rate limit + retry
- [ ] Silent <15% of time
- [ ] Explained design decisions clearly
- [ ] Handled errors gracefully

---

## Weekend Break (Nov 21-22)
**Goal:** Rest & internalize

### **Saturday (Nov 21) - Light Review (2 hours)**
- Review all your practice code
- Fix any remaining bugs
- Read through all guides:
  - `PROPER_PRACTICE_PROTOCOL.md`
  - `THREADING_IN_ML_SYSTEMS.md`
  - `THREADING_AND_DISTRIBUTED_SYSTEMS.md`
- **No coding** - just reading and thinking

### **Sunday (Nov 22) - Active Rest (1 hour)**
- Go through Shopify PDF again
- Review what they're looking for
- Think about questions you might ask them
- **No coding** - mental rest

---

## Week 2: Advanced Practice & Mock Interviews (Nov 23-25)

### **Day 6 (Nov 23): Mock Interview #1**
**Goal:** Full interview simulation

**Morning (1 hour prep):**
1. Set up like real interview:
   - Cursor ready
   - Timer set for 45 minutes
   - Record video (screen + audio)
   - Wear what you'll wear in interview

**Mock Interview (45 min):**
2. **Problem: Recommendation System (Problem 4)**
   - You haven't practiced this one yet (good!)
   - Simulates real interview where problem is new
   - Talk through:
     - "I'll use collaborative filtering because..."
     - "Jaccard similarity: |intersection| / |union|"
     - "Cold start problem: need content-based fallback"

**Post-Interview Review (2 hours):**
3. Watch entire recording
4. Score yourself:
   - Communication: ?/10
   - Problem solving: ?/10
   - Code quality: ?/10
   - Testing: ?/10
   - Time management: ?/10

5. Write down:
   - What went well
   - What needs improvement
   - Specific phrases that worked
   - Moments of silence (why?)

**Success Criteria:**
- [ ] Completed in 45 minutes
- [ ] Working recommendation system (basic)
- [ ] Communication >7/10
- [ ] Identified specific improvements

---

### **Day 7 (Nov 24): Mock Interview #2 + Polish**
**Goal:** Get to 9/10 communication

**Morning Mock Interview (45 min):**
1. **Problem: A/B Testing (Problem 5)**
   - Another unseen problem
   - Focus on explaining statistics:
     - "Two-proportion z-test compares conversion rates"
     - "p-value < 0.05 means statistically significant"
     - "Need to calculate sample size for 80% power"
   - Record yourself

**Review (1 hour):**
2. Compare Mock #1 vs Mock #2
3. Is communication improving?
4. Are you still making same mistakes?

**Afternoon Polish Session (2 hours):**
5. Pick your weakest problem from Days 1-5
6. Re-do it one more time
7. Aim for PERFECT communication
8. Record and confirm: Silent <10% of time

**Evening Prep (1 hour):**
9. Prepare questions to ask interviewer:
   - "What does the ML Infra team work on day-to-day?"
   - "What's the biggest technical challenge you're facing?"
   - "How does the team approach A/B testing at Shopify's scale?"

**Success Criteria:**
- [ ] Mock #2 better than Mock #1
- [ ] Communication >8/10
- [ ] Can explain statistical concepts
- [ ] Have 3-5 questions ready for interviewer

---

### **Day 8 (Nov 25 - Interview Day): Light Review + Interview**
**Goal:** Stay calm, execute what you practiced

**Morning (1-2 hours before interview):**
1. **Don't code!** Just review:
   - Look at your best practice code
   - Review common patterns (locks, caching, validation)
   - Read your notes on trade-offs

2. **Quick mental checklist:**
   ```
   ✓ Talk constantly - explain every decision
   ✓ Test incrementally - after each function
   ✓ Validate inputs - check None, empty, negative
   ✓ Use Cursor - ask for help when stuck
   ✓ Think out loud - even when debugging
   ✓ Discuss trade-offs - time/space, accuracy/latency
   ✓ Mention production - logging, metrics, errors
   ```

3. **Warm up voice:**
   - Practice explaining one algorithm out loud
   - Get comfortable talking to yourself
   - Imagine interviewer is friendly and helpful (they are!)

**During Interview:**
4. **First 2 minutes:**
   - Listen carefully to problem
   - Ask clarifying questions
   - Repeat problem back to confirm understanding

5. **Next 3 minutes:**
   - Talk through approach
   - Outline classes/functions
   - Discuss trade-offs of different approaches
   - Get interviewer buy-in before coding

6. **Next 30 minutes:**
   - Code while narrating constantly
   - Test each function as you write it
   - Use Cursor when stuck
   - Think out loud when debugging

7. **Last 10 minutes:**
   - Walk through test cases
   - Discuss improvements
   - Ask questions about Shopify

**After Interview:**
8. Don't stress! You prepared well.
9. Write down what happened (for learning)

---

## Daily Communication Checklist

Use this EVERY practice session:

### **While Coding, Say:**
- [ ] "I'm choosing X because Y"
- [ ] "This will be O(n) because we iterate once"
- [ ] "I need a lock here because multiple threads access this"
- [ ] "Let me test this with a simple example"
- [ ] "This edge case is important because..."
- [ ] "In production, I would also add..."
- [ ] "The trade-off here is X vs Y"
- [ ] "I'm stuck here, let me think... [think out loud]"

### **When Testing, Say:**
- [ ] "Let me test the happy path first"
- [ ] "Now let's test edge case: empty input"
- [ ] "This should return X because Y"
- [ ] "Ah, this failed because... let me fix it"

### **When Stuck, Say:**
- [ ] "I'm thinking about two approaches: A and B"
- [ ] "Let me use Cursor to check the syntax"
- [ ] "I want to verify my understanding of..."
- [ ] "Can I talk through my logic with you?"

---

## Success Metrics by Day

Track your improvement:

| Day | Problem | Time (min) | Silent % | Bugs | Communication /10 |
|-----|---------|-----------|----------|------|-------------------|
| 1   | Rate Limiter | 45 | 50% | 2 | 4/10 |
| 2   | Cache | 45 | 30% | 1 | 6/10 |
| 3   | Features | 45 | 20% | 0 | 7/10 |
| 4   | Data Pipeline | 45 | 20% | 0 | 7/10 |
| 5   | API Client | 45 | 15% | 0 | 8/10 |
| 6   | Mock #1 | 45 | 15% | 1 | 7/10 |
| 7   | Mock #2 | 45 | 10% | 0 | 9/10 |

**Target for interview day:** <10% silent, 9/10 communication

---

## Key Reminders

### **What Got You From 0% → 70%:**
- ✓ Good problem-solving skills
- ✓ Can write working code
- ✓ Understand algorithms and data structures

### **What Will Get You From 70% → 95%:**
- ✓ **Talking constantly** (this is 90% of remaining gap)
- ✓ Recording and reviewing yourself
- ✓ Explaining trade-offs
- ✓ Using Cursor effectively

### **Interview Day Mantras:**
1. **"The interviewer wants me to succeed"** (they do!)
2. **"Silence is the enemy"** (talk through everything)
3. **"Working code + no communication = fail"** (communication matters more)
4. **"Bugs are okay if I find and fix them"** (shows debugging skills)
5. **"I've practiced this 7+ times"** (you're ready!)

---

## Resources Quick Links

**Before Each Practice:**
- [ ] Read: `03_guides/PROPER_PRACTICE_PROTOCOL.md`
- [ ] Set timer: 45 minutes
- [ ] Start recording: Video/audio
- [ ] Open: Cursor + problem file

**After Each Practice:**
- [ ] Watch recording
- [ ] Count silent periods
- [ ] Update metrics table above
- [ ] Write 3 things to improve tomorrow

**Before Interview:**
- [ ] Review: Your best practice code
- [ ] Review: Common patterns (locks, validation, caching)
- [ ] Review: Trade-offs you discussed
- [ ] Warm up: Talk through one algorithm out loud

---

## You Got This! 🚀

**You rescheduled for a reason - use these 9 days wisely.**

Remember:
- Days 1-5: Practice problems with heavy focus on talking
- Days 6-7: Mock interviews to simulate real pressure
- Day 8: Light review + crush the interview

**Your biggest advantage:** You know your weakness (communication) and have 9 days to fix it.

**My prediction:** With this plan, you'll be at 9/10 communication by Nov 24.

Let's start with Day 1 right now! 💪
