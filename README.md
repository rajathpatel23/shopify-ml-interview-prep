# Shopify Senior ML Engineer - Interview Prep

**Interview Date:** November 19, 2025 (5 days!)
**Format:** 75-minute pair programming session
**GitHub:** https://github.com/rajathpatel23/shopify-ml-interview-prep

---

## 🚀 START HERE

### **Step 1: Quick Setup (5 minutes)**

```bash
cd ~/work/shopify-ml-interview-prep

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test that everything works
python 01_reference_solutions/problem_1_feature_pipeline/feature_pipeline.py
```

### **Step 2: Read the Essential Guides**

1. **[00_START_HERE/QUICK_START.md](00_START_HERE/QUICK_START.md)** - Read this first! (5 min)
2. **[03_guides/PROPER_PRACTICE_PROTOCOL.md](03_guides/PROPER_PRACTICE_PROTOCOL.md)** - How to practice correctly (10 min)
3. **[03_guides/INTERVIEW_GUIDE.md](03_guides/INTERVIEW_GUIDE.md)** - Keep open during interview (reference)

### **Step 3: Start Practicing**

```bash
# Open your first practice problem
code 02_your_practice/practice_session_2_proper.py

# Follow PROPER_PRACTICE_PROTOCOL.md:
# 1. Start recording (audio/video)
# 2. Set 45-minute timer
# 3. TALK OUT LOUD the entire time
# 4. Test after each function
# 5. Handle edge cases
```

---

## 📁 Repository Structure

```
shopify-ml-interview-prep/
│
├── 00_START_HERE/              ← Read these first
│   ├── QUICK_START.md         Quick setup and overview
│   └── README.md              Original README
│
├── 01_reference_solutions/     ← Study these (DON'T copy!)
│   ├── problem_1_feature_pipeline/
│   ├── problem_2_prediction_service/
│   ├── problem_4_recommendations/
│   ├── problem_7_rate_limiter/
│   └── problem_8_cache_system/
│
├── 02_your_practice/           ← Your practice files
│   ├── practice_session_1.py
│   ├── practice_session_1_complete.py
│   └── practice_session_2_proper.py  ← Start here!
│
├── 03_guides/                  ← Interview prep guides
│   ├── PROPER_PRACTICE_PROTOCOL.md  ← How to practice
│   ├── INTERVIEW_GUIDE.md           ← Interview day guide
│   ├── PRACTICE_SCHEDULE.md         ← 5-day schedule
│   ├── TODAY_PRACTICE.md            ← Today's plan
│   └── PRACTICE_1_FEEDBACK.md       ← Your code review
│
├── requirements.txt            Python dependencies
└── README.md                   This file
```

---

## 🎯 What This Repo Contains

### **5 Production-Quality Reference Solutions:**

| Problem | Time | Focus Area | Difficulty |
|---------|------|------------|------------|
| **1. Feature Pipeline** | 45 min | Data processing, pandas, sklearn | Medium |
| **2. Prediction Service** | 60 min | System design, caching, error handling | Medium-Hard |
| **4. Recommendation System** | 75 min | ML algorithms, scalability | Hard |
| **7. Rate Limiter** | 45 min | Algorithms, trade-offs, concurrency | Medium |
| **8. Cache System** | 45 min | Data structures, eviction policies | Medium |

### **Comprehensive Guides:**

- **Practice Protocol** - The ONLY way to practice that works
- **Interview Guide** - Cheat sheet for interview day
- **5-Day Schedule** - Structured plan
- **Code Reviews** - Detailed feedback on your practice

---

## ⚠️ CRITICAL: How to Use This Repo

### **❌ WRONG WAY (Won't help you):**
1. Read the reference solutions
2. Copy-paste code
3. Don't talk out loud
4. Don't test
5. Think you're ready

### **✅ RIGHT WAY (Will prepare you):**
1. Read `PROPER_PRACTICE_PROTOCOL.md`
2. Start recording (audio/video)
3. Set timer (45-75 min depending on problem)
4. **TALK OUT LOUD the entire time**
5. Code while explaining your thinking
6. Test after each function
7. Handle edge cases
8. Review your recording
9. Compare with reference solution
10. Note what to improve
11. **Do it again**

---

## 🎤 The Most Important Thing

**COMMUNICATION > CODE QUALITY**

From Shopify's guide:
> "It's important to remember that finishing the problem isn't the only thing that counts. What really stands out is how well you **communicate technical concepts**."

**This means:**
- Talk continuously while coding
- Explain your decisions
- Discuss trade-offs
- Ask clarifying questions
- Think about edge cases out loud

**If you code in silence for 15 minutes, you fail the interview - even with perfect code.**

---

## 📅 5-Day Practice Plan

### **Today (Thursday, Nov 14):**
- [x] Set up environment
- [ ] **Do Practice Session 2** (Rate Limiter, 45 min, recorded, talking out loud)
- [ ] Review recording
- [ ] Re-do from scratch tonight

### **Friday (Nov 15):**
- [ ] Cache System (45 min)
- [ ] Recommendation System (75 min - full interview!)
- [ ] Review both

### **Saturday (Nov 16):**
- [ ] Data Processing Pipeline (60 min)
- [ ] Create your own problem and solve it

### **Sunday (Nov 17):**
- [ ] Mock interview with friend (75 min)
- [ ] Solo mock interview, recorded (75 min)
- [ ] Deep review

### **Monday (Nov 18):**
- [ ] Light review (NO new coding)
- [ ] Test environment one final time
- [ ] Read interview guide
- [ ] Early sleep!

### **Tuesday (Nov 19):**
- [ ] **INTERVIEW DAY** 🚀

---

## ✅ You're Ready When...

- [x] You can talk continuously for 45+ minutes while coding
- [x] Your recording has < 10 seconds total silence
- [x] You test after every function automatically
- [x] You catch bugs before running code
- [x] You list edge cases before coding
- [x] You explain trade-offs naturally
- [x] You use AI tools (Cursor) and explain what they suggest

---

## 🔥 Quick Commands

```bash
# Activate environment
source venv/bin/activate

# Run reference solutions to see how they work
python 01_reference_solutions/problem_7_rate_limiter/rate_limiter.py
python 01_reference_solutions/problem_8_cache_system/cache_system.py

# Practice problems
code 02_your_practice/practice_session_2_proper.py

# Read guides
code 03_guides/PROPER_PRACTICE_PROTOCOL.md
code 03_guides/INTERVIEW_GUIDE.md

# See your code review
code 03_guides/PRACTICE_1_FEEDBACK.md
```

---

## 📚 Key Files to Read (In Order)

1. **This README** (you are here) - 5 min
2. **[00_START_HERE/QUICK_START.md](00_START_HERE/QUICK_START.md)** - 5 min
3. **[03_guides/PROPER_PRACTICE_PROTOCOL.md](03_guides/PROPER_PRACTICE_PROTOCOL.md)** - 10 min ⭐ MOST IMPORTANT
4. **[03_guides/PRACTICE_1_FEEDBACK.md](03_guides/PRACTICE_1_FEEDBACK.md)** - 10 min (your code review)
5. **[03_guides/INTERVIEW_GUIDE.md](03_guides/INTERVIEW_GUIDE.md)** - Keep open during practice and interview

---

## 💡 Interview Tips Summary

### **Before Coding:**
- Restate the problem
- Ask clarifying questions
- List edge cases
- Discuss approach
- Get buy-in

### **While Coding:**
- Think out loud constantly
- Use AI (Cursor) naturally
- Test after each function
- Explain trade-offs
- Handle errors

### **After Coding:**
- Walk through test cases
- Discuss improvements
- Mention production considerations
- Ask what interviewer wants to see next

---

## 🎯 What Shopify Looks For

From the official guide:

1. **Communication** - How you explain technical concepts
2. **Problem-solving** - How you approach challenges
3. **Code quality** - Clean, maintainable code
4. **Collaboration** - How you work with interviewer
5. **Edge cases** - Do you think about failure modes?
6. **Trade-offs** - Can you discuss pros/cons?

**They explicitly say:** Use AI tools (ChatGPT, Cursor, Copilot)!

---

## 🚨 Common Mistakes to Avoid

1. ❌ Coding in silence
2. ❌ Not testing until the end
3. ❌ Ignoring edge cases
4. ❌ Over-engineering (bonus features before core works)
5. ❌ Not asking clarifying questions
6. ❌ Giving up when stuck
7. ❌ Not explaining AI-generated code

---

## 📞 Support

- **Issues with code?** Check the reference solutions in `01_reference_solutions/`
- **Not sure how to practice?** Read `03_guides/PROPER_PRACTICE_PROTOCOL.md`
- **Interview day questions?** See `03_guides/INTERVIEW_GUIDE.md`

---

## 🎉 You've Got This!

You have:
- ✅ 5 complete practice problems
- ✅ Comprehensive guides
- ✅ Detailed feedback on your work
- ✅ 5-day structured plan
- ✅ All the tools you need

**What you need to do:**
- 🎯 Practice the RIGHT way (protocol!)
- 🎯 Talk out loud constantly
- 🎯 Test incrementally
- 🎯 Review and improve

**5 days is enough if you practice properly.**

---

## 🚀 Next Steps

**Right now:**
1. Read `03_guides/PROPER_PRACTICE_PROTOCOL.md` (10 min)
2. Open `02_your_practice/practice_session_2_proper.py`
3. Start recording
4. Set 45-minute timer
5. **BEGIN AND TALK FROM SECOND 1**

**After practice:**
1. Review recording
2. Score yourself
3. Compare with reference
4. Note improvements
5. Do it again

---

**Now stop reading and start practicing!** ⏰

Good luck! 🍀
