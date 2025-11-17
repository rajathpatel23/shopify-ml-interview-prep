# The Proper Practice Protocol

## 🎯 How to Practice Like You'll Interview

This is the ONLY way to practice that will prepare you for the real interview.

---

## Before You Start (5 minutes)

### 1. **Physical Setup**
```bash
cd ~/work/shopify-ml-interview-prep

# Open the practice file
code practice_session_2_proper.py

# Have a timer visible
# Have water ready
# Close Slack, email, everything else
```

### 2. **Mental Setup**
- [ ] Pretend interviewer is watching
- [ ] You MUST talk out loud
- [ ] This is the real thing
- [ ] No shortcuts

### 3. **Recording Setup (CRITICAL)**

**Option A: Screen + Audio Recording**
```
Mac: QuickTime Player -> New Screen Recording
- Check "Show Mouse Clicks"
- Enable microphone
- Start recording
```

**Option B: Just Audio** (minimum requirement)
```
Mac: Voice Memos app
iPhone: Voice Memos
- Start recording
- Put phone next to you
```

**WHY:** You MUST review yourself to improve. No recording = no improvement.

---

## The Protocol (45 minutes)

### **Phase 1: Understand & Plan (5 minutes)**

**TALK OUT LOUD:**
```
"Okay, let me read the problem carefully...

[Read problem]

So I need to build a rate limiter using Token Bucket algorithm.
The key requirements are:
1. Limit requests per user
2. Configurable max_requests and window
3. Token bucket algorithm
4. Clear error messages

Let me think about edge cases before I start coding:
- What if user_id is empty or None?
- What if max_requests is 0 or negative?
- What if window_seconds is 0?
- What about memory - do I need to clean up old users?
- What about thread safety? I'll mention it but might not implement.

My approach will be:
1. Store buckets dict: {user_id: (tokens, last_refill_time)}
2. On each request:
   - Calculate time elapsed
   - Refill tokens based on rate
   - Check if >= 1 token available
   - If yes: consume and allow
   - If no: deny

The refill rate will be: max_requests / window_seconds tokens per second

Let me start coding..."
```

**✅ REQUIRED:** Say all of this OUT LOUD before typing a single line of code.

---

### **Phase 2: Code + Test Incrementally (35 minutes)**

#### **2.1: Write `__init__` (3 minutes)**

**TALK OUT LOUD:**
```
"I'm starting with __init__. I need to store:
- max_requests: the bucket capacity
- window_seconds: to calculate refill rate
- buckets: dict to track each user's tokens and time
- refill_rate: tokens per second, calculated as max/window

I should validate inputs here. Let me add that...

[Type code]

Actually, I'm also going to calculate refill_rate here so I
don't recalculate it every time. That's a small optimization.

Let me test this..."
```

**CODE:**
```python
def __init__(self, max_requests: int, window_seconds: int):
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    self.max_requests = max_requests
    self.window_seconds = window_seconds
    self.refill_rate = max_requests / window_seconds
    self.buckets: Dict[str, Tuple[float, float]] = {}
```

**TEST IMMEDIATELY:**
```python
# At the bottom of the file
if __name__ == "__main__":
    print("Testing __init__...")

    # Test normal case
    limiter = RateLimiter(10, 60)
    print(f"✓ Created limiter: {limiter.max_requests} req / {limiter.window_seconds}s")
    print(f"  Refill rate: {limiter.refill_rate} tokens/sec")

    # Test edge case: negative
    try:
        bad_limiter = RateLimiter(-5, 60)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught error: {e}")

    print()
```

**RUN IT:**
```bash
python practice_session_2_proper.py
```

**VERIFY IT WORKS BEFORE MOVING ON!**

---

#### **2.2: Write `_refill_tokens` (5 minutes)**

**TALK OUT LOUD:**
```
"Now I'll write the refill logic. This calculates how many
tokens to add based on time elapsed.

If user doesn't exist, I'll initialize with full bucket.
Otherwise, I calculate:
- Time elapsed since last refill
- Tokens to add = elapsed * refill_rate
- New token count = min(current + added, max)

I'm capping at max to prevent infinite accumulation.

Let me code this and test..."
```

**CODE:**
```python
def _refill_tokens(self, user_id: str) -> float:
    current_time = time.time()

    if user_id not in self.buckets:
        # New user, give full bucket
        self.buckets[user_id] = (self.max_requests, current_time)
        return self.max_requests

    tokens, last_refill = self.buckets[user_id]

    # Calculate tokens to add
    time_elapsed = current_time - last_refill
    tokens_to_add = time_elapsed * self.refill_rate

    # Update tokens (cap at max)
    new_tokens = min(self.max_requests, tokens + tokens_to_add)
    self.buckets[user_id] = (new_tokens, current_time)

    return new_tokens
```

**TEST IMMEDIATELY:**
```python
if __name__ == "__main__":
    # ... previous tests ...

    print("Testing _refill_tokens...")

    limiter = RateLimiter(10, 10)  # 10 requests per 10 seconds = 1 per second

    # Test 1: New user gets full bucket
    tokens = limiter._refill_tokens("user1")
    print(f"✓ New user gets {tokens} tokens (expected 10)")

    # Test 2: Immediate refill gives same amount
    tokens = limiter._refill_tokens("user1")
    print(f"✓ Immediate refill: {tokens} tokens (expected ~10)")

    # Test 3: After 1 second, should stay at max (already full)
    time.sleep(1.1)
    tokens = limiter._refill_tokens("user1")
    print(f"✓ After 1s (still full): {tokens} tokens (expected 10)")

    # Test 4: Consume some, then refill
    limiter.buckets["user1"] = (5.0, time.time() - 2.0)  # 5 tokens, 2 seconds ago
    tokens = limiter._refill_tokens("user1")
    print(f"✓ After consuming 5, wait 2s: {tokens} tokens (expected ~7)")

    print()
```

**RUN AND VERIFY!**

---

#### **2.3: Write `allow_request` (5 minutes)**

**TALK OUT LOUD:**
```
"Now the main method. This needs to:
1. Refill tokens for the user
2. Check if they have at least 1 token
3. If yes: consume 1 token and return True
4. If no: return False

I also need to validate user_id isn't empty.

Let me implement..."
```

**CODE:**
```python
def allow_request(self, user_id: str) -> bool:
    if not user_id:
        raise ValueError("user_id cannot be empty")

    # Refill tokens
    tokens = self._refill_tokens(user_id)

    # Check if can consume
    if tokens >= 1.0:
        # Consume one token
        current_time = time.time()
        self.buckets[user_id] = (tokens - 1.0, current_time)
        return True
    else:
        return False
```

**TEST IMMEDIATELY:**
```python
if __name__ == "__main__":
    # ... previous tests ...

    print("Testing allow_request...")

    limiter = RateLimiter(5, 10)  # 5 requests per 10 seconds

    # Test 1: First 5 requests should succeed
    user = "test_user"
    results = []
    for i in range(7):
        allowed = limiter.allow_request(user)
        results.append(allowed)
        print(f"  Request {i+1}: {'✓ Allowed' if allowed else '✗ Denied'}")

    expected = [True, True, True, True, True, False, False]
    if results == expected:
        print(f"✓ Rate limiting works correctly!")
    else:
        print(f"✗ Expected {expected}, got {results}")

    # Test 2: Empty user_id
    try:
        limiter.allow_request("")
        print("✗ Should have raised ValueError")
    except ValueError:
        print("✓ Caught empty user_id error")

    print()
```

**RUN AND VERIFY!**

---

#### **2.4: Add More Tests (10 minutes)**

**TALK OUT LOUD:**
```
"Now let me add comprehensive tests to verify:
- Token refill over time
- Multiple users are independent
- Edge cases

Let me write these tests..."
```

**Add these tests and VERIFY each one works.**

---

### **Phase 3: Review & Improve (5 minutes)**

**TALK OUT LOUD:**
```
"Let me review what I built:

[Read through code]

Things that work well:
- Token bucket algorithm implemented correctly
- Input validation
- Tested incrementally
- Handles multiple users

Things I'd improve for production:
1. Thread safety - would need locks
2. Memory cleanup - remove inactive users
3. Better error messages - tell user when they can retry
4. Monitoring - track rate limit hits
5. Different limits per user tier

Let me add a helper method to get remaining tokens..."
```

**ONLY add improvements if time permits and core works perfectly.**

---

## After Practice (15 minutes - CRITICAL!)

### **1. Watch/Listen to Your Recording**

**Score yourself:**

| Metric | Score (1-10) | Notes |
|--------|--------------|-------|
| Talked continuously | ___ | Seconds of silence? |
| Explained decisions | ___ | Said "because..."? |
| Tested incrementally | ___ | After each function? |
| Caught own bugs | ___ | Fixed before running? |
| Handled edge cases | ___ | Thought about them? |
| Stayed calm | ___ | Panicked or steady? |

### **2. Compare with Reference**

```bash
# Compare your solution with the reference
code practice_session_2_proper.py
code problem_7_rate_limiter/rate_limiter.py
```

**Ask yourself:**
- What did they do that I didn't?
- What edge cases did I miss?
- How is their code organized differently?
- What can I learn?

### **3. Write Down Learnings**

```
WHAT I DID WELL:
-
-
-

WHAT I NEED TO IMPROVE:
-
-
-

SPECIFIC ACTIONS FOR NEXT PRACTICE:
-
-
-
```

---

## The Non-Negotiables

### **❌ DON'T START if you won't:**
- Talk out loud the entire time
- Record yourself
- Test after each function
- Review the recording after

### **✅ DO THIS EVERY PRACTICE:**
1. Set timer (visible)
2. Start recording (audio minimum)
3. Talk continuously (narrate everything)
4. Test incrementally (after each function)
5. Think edge cases (before coding)
6. Review recording (score yourself)

---

## Practice Schedule Using This Protocol

### **Today (Thursday):**
- **Now:** Rate Limiter (this file) - 45 min
- **Review:** Watch recording, score yourself - 15 min
- **Tonight:** Re-do Rate Limiter from scratch - 45 min

### **Tomorrow (Friday):**
- **Morning:** Cache System - 45 min
- **Afternoon:** Recommendation System - 75 min (full interview length)
- **Review each:** 15 min

### **Saturday:**
- **Morning:** Data Processing Pipeline - 60 min
- **Afternoon:** Create your own problem and solve it - 60 min

### **Sunday:**
- **Morning:** Mock interview (find a friend) - 75 min
- **Afternoon:** Second mock (by yourself, recorded) - 75 min
- **Review:** Deep review of both mocks

### **Monday:**
- **Morning:** Light review, run all examples
- **Afternoon:** Read interview guide, relaxrelax
- **Evening:** Early sleep!

---

## Success Criteria

**You're ready when:**
- ✅ You can talk continuously for 45 minutes while coding
- ✅ Your recording has < 10 seconds of total silence
- ✅ You test after every function automatically
- ✅ You catch bugs before running code
- ✅ You list edge cases before coding
- ✅ Your code works on first run (or second)

**You're NOT ready if:**
- ❌ You code in silence
- ❌ You write all code then test
- ❌ You don't think about edge cases
- ❌ Your code has obvious bugs
- ❌ You skip testing

---

## 🚀 START NOW

**Right now:**

1. **Set a 45-minute timer**
2. **Start recording (audio minimum)**
3. **Open practice_session_2_proper.py**
4. **Start timer and BEGIN**
5. **TALK OUT LOUD from second 1**

**After 45 minutes:**
- Stop
- Watch/listen to recording
- Score yourself
- Read CRITICAL_FEEDBACK.md

---

**Remember:** Perfect practice makes perfect. Sloppy practice makes sloppy.

**This is the only way to prepare properly.**

**Now go! Start your timer and BEGIN!** ⏰
