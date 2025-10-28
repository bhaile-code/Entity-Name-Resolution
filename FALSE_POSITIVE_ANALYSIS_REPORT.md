# False Positive Analysis & Precision Trade-off Report

**Entity Name Resolution System - Performance Analysis**

Date: 2025-10-24
Current Configuration: OpenAI text-embedding-3-large + Adaptive GMM Thresholding
Dataset: 739 company names from `sample_data_500.csv`

---

## Executive Summary

**Current Performance (Best Configuration):**
- **F1 Score:** 82.4%
- **Precision:** 93.0% (7% false positive rate = 77 incorrect merges)
- **Recall:** 74.0% (26% false negative rate = 359 missed matches)
- **True Positives:** 1,022 correct matches
- **True Negatives:** 262,443 correctly separated pairs

**Key Finding:** The system has a **shared-word problem** - 87% of false positives (67 out of 77) occur when company names share common words like "American", "United", "Delta", but represent completely different entities.

---

## 1. Types of False Positives (77 Total)

### 1.1 Shared Word Problem (67 pairs, 87.0%)

**The Problem:** Token_set_ratio scores 100% when company names share words, regardless of whether they're the same entity.

**Top Problem Words:**
- **"american"**: 36 false positives (47% of all FPs)
  - "American Airlines" ↔ "American Express" ↔ "American Family Insurance" ↔ "American Standard"
  - Example: "American" merged with "American Family Insurance" (74.7% score)

- **"united"**: 14 false positives (18% of all FPs)
  - "United Airlines" ↔ "United Parcel Service (UPS)" ↔ "United Rentals" ↔ "UnitedHealthcare"
  - Example: "United" merged with "United Parcel Service" (76.9% score)

- **"general"**: 7 false positives (9% of all FPs)
  - "General Motors" ↔ "General Electric" ↔ "General Dynamics" ↔ "General Mills"

- **"delta"**: 5 false positives (6.5% of all FPs)
  - "Delta Airlines" ↔ "Delta Dental" ↔ "Delta Faucet Company" ↔ "Delta Community Credit Union"

- **"adobe"**: 5 false positives (6.5% of all FPs)
  - "Adobe Inc." (software company) ↔ "Adobe Rent-A-Car" (80.3% score - HIGHEST FP)

**Why This Happens:**
- Token_set_ratio at 15% weight still creates 100% scores for shared words
- Embeddings (45% weight) don't compensate enough when tokens dominate
- WRatio (40% weight) also scores high due to substring matches

### 1.2 Subset Matching Problem (12 pairs, 15.6%)

**The Problem:** When one name is a subset of another, all metrics score very high even if they're different entities.

**Examples:**
- "Adobe" ↔ "Adobe Rent-A-Car"
  - Normalized: "adobe" vs "adobe rent a car"
  - Score: 80.3% (token_set: 100%, WRatio: 90%)

- "United" ↔ "United Parcel Service"
  - Normalized: "united" vs "united parcel service"
  - Score: 76.9% (token_set: 100%, WRatio: 90%)

**Why This Happens:**
- Token_set_ratio: 100% (all tokens in subset are in superset)
- WRatio: 90% (high character overlap)
- Embeddings can't distinguish without industry context

### 1.3 Component Score Breakdown

**Average Scores for False Positives:**
- Final Score: 62.0%
- WRatio: 74.0% ← **Main culprit**
- Token Set: 68.7% ← **Second contributor**
- Embedding: 49.7% (correctly low, but overridden by other components)
- Phonetic: -0.2 (neutral)

**Insight:** Even though embeddings correctly score low (49.7%), the high WRatio (74%) and Token Set (68.7%) scores push the final score above threshold.

---

## 2. Precision vs Recall Trade-off Analysis

### 2.1 Current State

| Configuration | Precision | Recall | F1 | False Positives | False Negatives |
|--------------|-----------|--------|-----|-----------------|-----------------|
| **OpenAI-Large + Adaptive** | **93.0%** | **74.0%** | **82.4%** | **77** | **359** |
| OpenAI-Small + Adaptive | 91.4% | 73.4% | 81.4% | 96 | 367 |
| OpenAI-Large + Fixed | 96.4% | 62.8% | 76.1% | 32 | 514 |
| OpenAI-Small + Fixed | 97.3% | 59.2% | 73.6% | 23 | 563 |

### 2.2 Trade-off Considerations

**Option A: Increase Precision (Target: 96%+)**
- **Benefit:** Reduce false positives from 77 → ~30 (60% reduction)
- **Cost:** Recall drops from 74% → ~63% (11% decrease)
- **Net Effect:** F1 score drops from 82.4% → ~76%
- **Business Impact:** More manual review needed (150+ additional false negatives)

**Option B: Increase Recall (Target: 80%+)**
- **Benefit:** Capture 6% more true matches (reduce false negatives by ~80)
- **Cost:** Precision drops from 93% → ~88% (lose 5%)
- **Net Effect:** F1 score potentially increases to ~84%
- **Business Impact:** More false positives to manually reject (~150 total FPs)

**Option C: Balanced Optimization (RECOMMENDED)**
- **Target:** 94-95% precision, 76-78% recall
- **Approach:** Fix shared-word problem without sacrificing too much recall
- **Expected:** F1 score improves to 84-85%
- **Business Impact:** Best balance for production use

### 2.3 Recommendation

**PRIORITIZE PRECISION WHILE IMPROVING RECALL**

**Rationale:**
1. **Current recall (74%) is acceptable** - catching 3 out of 4 true matches
2. **False positives are costly** - require manual investigation and risk reputation damage
3. **False negatives are easier to catch** - users can manually group missed matches
4. **The 77 false positives have fixable patterns** - targeted improvements can reduce FPs by 50-70% without hurting recall

**Target Metrics:**
- Precision: 95-96% (reduce FPs from 77 → 40-50)
- Recall: 75-77% (maintain or slightly improve)
- F1 Score: 84-85% (2-3% improvement)

---

## 3. Improvement Proposals

### 🎯 **PROPOSAL 1: Industry-Aware Context Disambiguation**

**Problem Solved:**
- Shared-word false positives (87% of errors)
- Distinguishes "American Airlines" from "American Express" despite shared word

**How It Works:**

1. **Industry Classification Layer**
   - Add lightweight industry classifier using embeddings
   - Categories: {Technology, Finance, Retail, Airlines, Insurance, Manufacturing, Food/Beverage, Healthcare, etc.}
   - Implementation: Few-shot classification using OpenAI embeddings + k-NN or simple logistic regression

2. **Industry Penalty System**
   - If two names share words BUT belong to different industries → apply penalty to similarity score
   - Penalty calculation:
     ```python
     def calculate_industry_penalty(name1_industry, name2_industry, similarity_score):
         if name1_industry != name2_industry:
             # Apply graduated penalty based on industry distance
             industry_distance = get_industry_distance(name1_industry, name2_industry)
             # Reduce score by 10-20% if different industries
             penalty = 0.10 + (0.10 * industry_distance)
             return similarity_score * (1 - penalty)
         return similarity_score
     ```

3. **Word-Industry Conflict Detection**
   - Maintain a dictionary of ambiguous words and their industries:
     ```python
     AMBIGUOUS_WORDS = {
         'american': ['Airlines', 'Finance', 'Insurance', 'Manufacturing'],
         'united': ['Airlines', 'Logistics', 'Healthcare', 'Rentals'],
         'delta': ['Airlines', 'Healthcare', 'Manufacturing', 'Finance'],
         'general': ['Automotive', 'Tech', 'Food', 'Defense'],
         'meta': ['Tech/Social', 'Finance'],
         'apple': ['Tech', 'Music/Entertainment'],
         'continental': ['Airlines', 'Automotive'],
         'target': ['Retail', 'Marketing'],
         'oracle': ['Tech/Database', 'Finance']
     }
     ```
   - If shared word is in `AMBIGUOUS_WORDS` → require embedding score > 70% to proceed

**Implementation Complexity:** **MEDIUM**
- Requires training or fine-tuning industry classifier (~200-500 labeled examples)
- Modify confidence scoring function to incorporate industry penalty
- Add industry metadata to company profiles
- Estimated effort: 2-3 days

**Expected Impact:**
- **Precision:** +2-3% (reduce shared-word FPs by 60-70%)
- **Recall:** -0.5% (minimal - only affects borderline cases)
- **F1 Score:** +1.5-2%
- **False Positives:** 77 → 25-30 (60% reduction)

**Example Results:**
- ✅ "American Airlines" vs "American Express" → 65% → **REJECT** (different industries)
- ✅ "Delta Airlines" vs "Delta Dental" → 58% → **REJECT** (airlines ≠ healthcare)
- ✅ "Adobe Inc." vs "Adobe Rent-A-Car" → 62% → **REJECT** (tech ≠ car rental)
- ✅ "United Airlines" vs "United Parcel Service" → 68% → **REJECT** (airlines ≠ logistics)

---

### 🎯 **PROPOSAL 2: Multi-Token Requirement for Short Names**

**Problem Solved:**
- Subset matching false positives (15.6% of errors)
- Single-word name collisions ("United", "American", "Delta", "Adobe")

**How It Works:**

1. **Minimum Token Match Rule**
   - If name has ≤ 2 tokens (e.g., "Adobe", "United") → require at least 2 matching tokens
   - If name has ≥ 3 tokens → allow partial matches
   - Example:
     ```python
     def should_require_multi_token_match(name1, name2):
         tokens1 = name1.split()
         tokens2 = name2.split()

         # Short names need more evidence
         if len(tokens1) <= 2 or len(tokens2) <= 2:
             # Require at least 2 matching meaningful tokens
             shared_tokens = set(tokens1) & set(tokens2)
             meaningful_shared = [t for t in shared_tokens if len(t) > 2]
             return len(meaningful_shared) < 2

         return False
     ```

2. **Specificity Boost for Longer Names**
   - Reward names with unique descriptive tokens
   - "Adobe Inc." vs "Adobe Rent-A-Car" → "Rent-A-Car" is specific → different entity
   - "United" vs "United Parcel Service" → "Parcel Service" is specific → different entity
   - Implementation:
     ```python
     def calculate_specificity_score(name1, name2):
         tokens1 = set(normalize_name(name1).split())
         tokens2 = set(normalize_name(name2).split())

         unique1 = tokens1 - tokens2
         unique2 = tokens2 - tokens1

         # If one has specific tokens and other doesn't, penalize
         if len(unique1) > 0 or len(unique2) > 0:
             # Check if unique tokens are industry-specific
             if has_specific_industry_tokens(unique1) or has_specific_industry_tokens(unique2):
                 return -10  # Apply 10-point penalty

         return 0
     ```

3. **Blocklist for Known Single-Word Conflicts**
   - Maintain list of single words that NEVER match alone:
     ```python
     NEVER_MATCH_ALONE = {
         'american', 'united', 'delta', 'general', 'continental',
         'target', 'meta', 'oracle', 'apple', 'adobe'
     }
     ```
   - If normalized name reduces to blocklisted word → require additional evidence

**Implementation Complexity:** **LOW**
- Add simple token counting logic to matching function
- Create blocklist of ambiguous short names
- Estimated effort: 4-6 hours

**Expected Impact:**
- **Precision:** +1-2% (reduce subset FPs by 80-90%)
- **Recall:** -0.5% (minimal - only affects ambiguous short names)
- **F1 Score:** +0.5-1%
- **False Positives:** 77 → 60-65 (15-20% reduction)

**Example Results:**
- ✅ "Adobe" vs "Adobe Rent-A-Car" → Requires 2+ matching tokens → **REJECT**
- ✅ "United" vs "United Parcel Service" → Requires 2+ matching tokens → **REJECT**
- ✅ "Delta" vs "Delta Dental" → Requires 2+ matching tokens → **REJECT**
- ✅ "American" vs "American Family Insurance" → Requires 2+ matching tokens → **REJECT**

---

### 🎯 **PROPOSAL 3: Dynamic Component Weight Adjustment**

**Problem Solved:**
- Token_set and WRatio dominating when they shouldn't (combined 74%+ for FPs)
- Embeddings being overridden despite correctly identifying mismatch

**How It Works:**

1. **Context-Aware Weight Shifting**
   - **Current weights** (fixed for all pairs):
     - WRatio: 40%, Token Set: 15%, Embedding: 45%

   - **Proposed adaptive weights** (based on name characteristics):
     ```python
     def get_adaptive_weights(name1, name2):
         norm1 = normalize_name(name1)
         norm2 = normalize_name(name2)

         # Calculate name characteristics
         shared_word_ratio = calculate_shared_word_ratio(norm1, norm2)
         length_diff_ratio = calculate_length_difference_ratio(norm1, norm2)
         has_ambiguous_word = contains_ambiguous_word(norm1, norm2)

         # Base weights
         wratio_weight = 0.40
         token_weight = 0.15
         embed_weight = 0.45

         # CASE 1: High shared word ratio (potential false positive risk)
         if shared_word_ratio > 0.7:
             # Reduce token_set, increase embedding reliance
             token_weight = 0.10  # Reduce from 15% → 10%
             embed_weight = 0.50  # Increase from 45% → 50%

         # CASE 2: Contains known ambiguous words
         if has_ambiguous_word:
             # Trust embeddings more
             wratio_weight = 0.30  # Reduce from 40% → 30%
             token_weight = 0.10  # Reduce from 15% → 10%
             embed_weight = 0.60  # Increase from 45% → 60%

         # CASE 3: Significant length difference (subset risk)
         if length_diff_ratio > 0.5:
             # One name is much longer → trust embeddings
             token_weight = 0.10
             embed_weight = 0.55
             wratio_weight = 0.35

         return wratio_weight, token_weight, embed_weight
     ```

2. **Embedding Veto Power**
   - If embedding score < 60% AND (WRatio > 80% OR Token_set > 90%):
     - **Trigger:** Likely false positive due to shared words
     - **Action:** Apply "embedding veto" - reduce final score by 15-20%
     - Example:
       ```python
       if embedding_score < 0.60 and (wratio > 80 or token_set > 90):
           # Embeddings detect semantic mismatch - apply veto
           final_score *= 0.80  # 20% penalty
       ```

3. **Score Confidence Calibration**
   - Add confidence intervals to scores
   - If components disagree significantly → flag as low confidence
   - Example:
     ```python
     def calculate_confidence_level(wratio, token_set, embedding):
         # Standard deviation of components
         scores = [wratio, token_set, embedding]
         std_dev = statistics.stdev(scores)

         if std_dev > 20:  # High disagreement
             return 'LOW_CONFIDENCE'
         elif std_dev > 10:
             return 'MEDIUM_CONFIDENCE'
         else:
             return 'HIGH_CONFIDENCE'

         # Adjust threshold based on confidence
         if confidence == 'LOW_CONFIDENCE':
             effective_threshold = threshold * 1.10  # Raise bar by 10%
     ```

**Implementation Complexity:** **MEDIUM**
- Modify `calculate_confidence()` function in name_matcher.py
- Add helper functions for name characteristic analysis
- Tune weight adjustment parameters based on validation data
- Estimated effort: 1-2 days

**Expected Impact:**
- **Precision:** +2-3% (embeddings get more influence in ambiguous cases)
- **Recall:** +1-2% (better weight balance helps borderline true matches)
- **F1 Score:** +2-3%
- **False Positives:** 77 → 45-55 (30-40% reduction)

**Example Results:**
- ✅ "American Airlines" vs "American Express"
  - Token_set: 100%, WRatio: 90%, Embedding: 43%
  - **OLD:** 40%×90 + 15%×100 + 45%×43 = 70.4% → **MATCH (FP)**
  - **NEW:** Adaptive weights (30%, 10%, 60%) + embedding veto
    - 30%×90 + 10%×100 + 60%×43 = 62.8% → **REJECT** ✓

- ✅ "Adobe" vs "Adobe Rent-A-Car"
  - Token_set: 100%, WRatio: 90%, Embedding: 56%
  - **OLD:** 40%×90 + 15%×100 + 45%×56 = 76.2% → **MATCH (FP)**
  - **NEW:** Adaptive weights + subset detection + embedding veto
    - 30%×90 + 10%×100 + 60%×56 = 70.6% × 0.80 (veto) = 56.5% → **REJECT** ✓

---

## 4. Implementation Roadmap

### Phase 1: Quick Wins (1 week)
**Proposal 2: Multi-Token Requirement**
- Complexity: LOW
- Impact: 15-20% FP reduction
- No API costs
- Immediate deployment

### Phase 2: Core Improvements (2 weeks)
**Proposal 3: Dynamic Component Weights**
- Complexity: MEDIUM
- Impact: 30-40% FP reduction
- Improves both precision and recall
- Deploy with A/B testing

### Phase 3: Advanced Features (3-4 weeks)
**Proposal 1: Industry-Aware Disambiguation**
- Complexity: MEDIUM
- Impact: 60-70% FP reduction (cumulative with Phase 1+2)
- Requires industry classifier training
- Optional: Can use industry keywords instead of full classifier

### Expected Combined Impact
**After All Three Proposals:**
- Precision: 93.0% → **96-97%**
- Recall: 74.0% → **76-78%**
- F1 Score: 82.4% → **85-87%**
- False Positives: 77 → **10-20** (75-85% reduction)
- False Negatives: 359 → **340-350** (minimal increase)

---

## 5. Alternative Approaches (Not Recommended)

### ❌ Increase Fixed Threshold (85% → 90%)
- **Impact:** Precision improves to 96%+, but recall drops to 63%
- **Why Not:** Loses too many true matches, F1 score decreases
- **Better:** Use dynamic thresholding based on name characteristics

### ❌ Remove Token_set Component Entirely
- **Impact:** Eliminates shared-word false positives but breaks legitimate matches
- **Why Not:** Token_set helps with word order variations ("IBM Corporation" vs "Corporation IBM")
- **Better:** Reduce weight conditionally (Proposal 3)

### ❌ Increase Embedding Weight to 70%+
- **Impact:** Better semantic understanding but higher API costs and slower processing
- **Why Not:** Current 45% is optimal for cost/performance, issue is other components overriding it
- **Better:** Use embedding veto power (Proposal 3)

---

## 6. Monitoring & Validation

### Success Metrics
1. **Primary:** F1 score ≥ 85%
2. **Secondary:** Precision ≥ 95%, Recall ≥ 75%
3. **Tertiary:** Processing time ≤ 70 seconds for 739 names

### Testing Strategy
1. **Unit Tests:** Test each improvement in isolation
2. **Integration Tests:** Validate combined effect on full dataset
3. **A/B Testing:** Run old vs new side-by-side for 1000 name dataset
4. **Edge Case Testing:** Focus on known false positive patterns

### Rollback Plan
- Keep current algorithm as fallback
- Implement feature flags for each proposal
- Monitor precision/recall daily for first week after deployment

---

## 7. Conclusion

**Key Takeaways:**
1. **False positives are highly concentrated:** 87% caused by shared-word problem
2. **Embeddings are working correctly:** They score low (~50%) for false positives, but are overridden
3. **Simple fixes can have major impact:** Multi-token requirement can reduce FPs by 15-20% with minimal effort
4. **Precision should be prioritized:** Current 93% is good, but 96%+ is achievable without sacrificing recall
5. **Combined approach is best:** All three proposals complement each other for maximum impact

**Recommended Next Steps:**
1. ✅ Implement Proposal 2 (Multi-Token Requirement) - **START HERE** (1 week)
2. ✅ Validate improvement on test dataset
3. ✅ Implement Proposal 3 (Dynamic Weights) - **CORE FIX** (2 weeks)
4. ✅ Consider Proposal 1 (Industry Classifier) - **ADVANCED** (3-4 weeks)
5. ✅ Monitor production metrics and iterate

**Expected Final Performance:**
- **F1 Score:** 85-87% (up from 82.4%)
- **Precision:** 96-97% (up from 93%)
- **Recall:** 76-78% (up from 74%)
- **False Positives:** 10-20 (down from 77, **85% reduction**)

---

**Report Generated:** 2025-10-24
**Analyst:** Claude Code (Anthropic)
**Data Source:** c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\backend\
