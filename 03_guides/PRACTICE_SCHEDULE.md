# 5-Day Practice Schedule (Nov 14-18)

## Today (Thursday, Nov 14) - Setup & First Practice

### Morning (2 hours)
- [x] Review Shopify interview guide
- [x] Set up repository
- [ ] **Test your environment:**
  ```bash
  cd ~/work/shopify-ml-interview-prep
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Test Google Meet screen share with Cursor
- [ ] Enable line numbers in Cursor (Cmd+,)
- [ ] Test GitHub SSH connection

### Afternoon (3 hours)
- [ ] **Problem 1: Feature Engineering Pipeline**
  - Run: `python problem_1_feature_pipeline/feature_pipeline.py`
  - **RECORD YOURSELF** solving it from scratch
  - **TALK OUT LOUD** the entire time
  - Time yourself: 45 minutes max

- [ ] **Problem 2: Prediction Service**
  - Run: `python problem_2_prediction_service/prediction_service.py`
  - Focus on: Error handling, caching, monitoring
  - Practice explaining design decisions

### Evening (1 hour)
- [ ] Watch your recording
- [ ] Note areas where you didn't communicate enough
- [ ] Review code - any bugs or improvements?

**Goal for today:** Get comfortable talking while coding!

---

## Friday (Nov 15) - Backend Systems & ML

### Morning (2-3 hours)
- [ ] **Problem 7: Rate Limiter**
  - Run: `python problem_7_rate_limiter/rate_limiter.py`
  - Time: 60 minutes
  - **Practice explaining trade-offs:**
    - Why Token Bucket vs Sliding Window?
    - Time vs Space complexity?
    - How would you scale this?

- [ ] **Problem 8: Cache System**
  - Run: `python problem_8_cache_system/cache_system.py`
  - Time: 60 minutes
  - **Practice explaining:**
    - LRU vs LFU - when to use each?
    - How does threading work?
    - Production considerations?

### Afternoon (2-3 hours)
- [ ] **Problem 4: Recommendation System**
  - Run: `python problem_4_recommendations/recommendation_system.py`
  - Time: 75 minutes (full interview length!)
  - **This is important!** Shopify is e-commerce
  - Practice explaining:
    - Collaborative vs Content-based
    - Cold start problem
    - Scalability

### Evening (1 hour)
- [ ] Create your own problem
- [ ] Think of edge cases for each problem
- [ ] Review Python best practices

**Goal for today:** Master system design thinking and trade-offs!

---

## Saturday (Nov 16) - Integration & Scenarios

### Morning (2 hours)
- [ ] **Combined Exercise:**
  - Build a simple API that uses:
    - Rate limiter
    - Cache
    - Prediction service
  - Time: 90 minutes
  - This simulates a real production system

### Afternoon (2 hours)
- [ ] **Shopify-Specific Scenarios:**

  **Scenario 1: Product Search Ranking (60 min)**
  ```
  Build a service that ranks search results.
  Input: Search query, user context
  Output: Ranked product IDs
  Consider: Relevance, personalization, business rules
  ```

  **Scenario 2: Inventory Prediction (45 min)**
  ```
  Predict if a product will be out of stock.
  Input: Product ID, historical sales
  Output: Risk score, recommended reorder
  Handle: Seasonal trends, new products
  ```

### Evening (1 hour)
- [ ] Review all code written so far
- [ ] Can you explain every design decision?
- [ ] Practice explaining to a friend/rubber duck

**Goal for today:** Build real-world integration skills!

---

## Sunday (Nov 17) - Mock Interview Day

### Morning (2 hours)
**Find a developer friend for this!** If not available, record yourself.

- [ ] **Full 75-Minute Mock Interview:**

  **Problem: Build an A/B Testing Analysis Service**
  ```
  Requirements:
  - Accept experiment data (control vs treatment)
  - Calculate key metrics (conversion rate, avg order value)
  - Determine statistical significance
  - Handle edge cases (small sample sizes, outliers)
  - Return recommendation (launch/don't launch)
  - Bonus: Implement early stopping

  Time: 75 minutes
  Focus: Communication, code quality, testing
  ```

- [ ] Have your friend give you feedback on:
  - Did you ask enough questions?
  - Did you explain your thinking?
  - Was your code readable?
  - Did you handle edge cases?
  - How was time management?

### Afternoon (2 hours)
- [ ] **Second Mock (by yourself):**

  **Problem: Real-time Fraud Detection Service**
  ```
  Build a service that scores transactions for fraud.
  Input: Transaction details (amount, location, time, user history)
  Output: Fraud score (0-1), risk level, reason
  Requirements:
  - Real-time scoring (< 100ms)
  - Handle missing data
  - Rate limiting (prevent abuse)
  - Logging for audit trail
  ```

- [ ] Record yourself
- [ ] Review recording

### Evening (1 hour)
- [ ] List your strengths from today's mocks
- [ ] List areas to improve
- [ ] Practice those weak areas

**Goal for today:** Simulate real interview conditions!

---

## Monday (Nov 18) - Final Prep & Relaxation

### Morning (2 hours)
- [ ] **Light review - NO new coding!**
- [ ] Re-run all the examples
- [ ] Read through the code
- [ ] Review INTERVIEW_GUIDE.md

### Afternoon (1-2 hours)
- [ ] **Final technical checklist:**
  ```bash
  # Test everything one last time
  cd ~/work/shopify-ml-interview-prep

  # 1. Python environment
  python --version
  python problem_1_feature_pipeline/feature_pipeline.py

  # 2. Google Meet screen share
  # Open Cursor, join test meeting, verify:
  # - Line numbers visible
  # - Font size readable
  # - Can share screen

  # 3. GitHub
  ssh -T git@github.com
  git config --list | grep user

  # 4. Cursor AI
  # Open Cursor, test Cmd+K (inline) and Cmd+L (chat)

  # 5. Clean workspace
  # Close: Slack, email, social media
  # Open only: Cursor, browser (Google Meet)
  ```

- [ ] **Create your interview cheat sheet:**
  ```
  Keep visible (off-screen):
  1. CLARIFY → PLAN → CODE → TEST
  2. Think aloud constantly!
  3. Ask questions
  4. If stuck: communicate
  5. Use AI naturally
  6. Explain trade-offs
  ```

### Late Afternoon (DO NOT CODE)
- [ ] Go for a walk
- [ ] Exercise (light)
- [ ] Eat a good meal
- [ ] Prepare for tomorrow:
  - Clean workspace
  - Test audio/video
  - Charge laptop
  - Have water ready

### Evening
- [ ] **Early to bed!** Get good sleep
- [ ] No coding tonight
- [ ] No cramming
- [ ] Relax and trust your preparation

**Goal for today:** Be rested and confident!

---

## Tuesday (Nov 19) - INTERVIEW DAY! 🚀

### 2 Hours Before Interview
- [ ] Eat light (avoid heavy meal)
- [ ] Use bathroom
- [ ] Dress comfortably (video call)
- [ ] Test audio/video one final time

### 30 Minutes Before
- [ ] Close ALL apps except:
  - Cursor
  - Browser (Google Meet)
  - Terminal (if needed)

- [ ] Open INTERVIEW_GUIDE.md (off-screen but visible)

- [ ] Final Google Meet test:
  - Share Cursor screen
  - Verify line numbers visible
  - Check font size readable

- [ ] Take 3 deep breaths
- [ ] You've got this!

### During Interview

**Remember:**
1. ✓ Restate problem
2. ✓ Ask clarifying questions
3. ✓ Discuss approach BEFORE coding
4. ✓ Think aloud constantly
5. ✓ Use AI naturally
6. ✓ Test with examples
7. ✓ Stay calm and positive

**If stuck:**
- Take a breath
- Think aloud
- Ask for guidance
- It's okay!

### After Interview
- Write down what went well
- Write down what could improve
- Follow up with recruiter if needed
- Relax - you did your best!

---

## Daily Reminders

**Every Practice Session:**
1. Set a timer (respect time limits!)
2. Talk out loud the ENTIRE time
3. Use Cursor/AI naturally
4. Explain your trade-offs
5. Test with examples
6. Review your code after

**Key Questions to Ask Yourself:**
- Can I explain why I made each decision?
- Did I handle edge cases?
- Is my code readable?
- Would this work in production?
- What would I improve given more time?

**Remember:**
- Communication > Completion
- Process > Perfect solution
- Calm > Panicked
- Questions > Assumptions

---

## Progress Tracker

Track your practice:

**Problems Solved:**
- [ ] Feature Engineering Pipeline
- [ ] Prediction Service
- [ ] Rate Limiter
- [ ] Cache System
- [ ] Recommendation System
- [ ] Mock Interview 1
- [ ] Mock Interview 2

**Skills Practiced:**
- [ ] Thinking aloud while coding
- [ ] Asking clarifying questions
- [ ] Explaining trade-offs
- [ ] Using AI tools
- [ ] Handling edge cases
- [ ] Time management
- [ ] Staying calm under pressure

**You've got this! The preparation is in the practice.** 💪
