# Tier 1 Improvements - Executive Summary

**Date:** October 27, 2025
**Status:** ⚠️ **Regression Detected** - Performance decreased instead of improving

---

## Results at a Glance

### OPENAI-LARGE + ADAPTIVE Configuration

| Metric | Baseline | After Tier 1 | Change | Status |
|--------|----------|--------------|--------|--------|
| **F1 Score** | **82.42%** | **80.89%** | **-1.53%** | 🔴 |
| **Precision** | 93.00% | 89.40% | -3.60% | 🔴 |
| **Recall** | 74.00% | 73.86% | -0.14% | 🔴 |
| **Processing Time** | 67.5s | 85.5s | +18.0s | 🔴 |
| **False Positives** | 77 | 121 | +44 | 🔴 |
| **False Negatives** | 359 | 361 | +2 | 🔴 |

---

## What Happened?

### Expected vs. Actual

**Expected**: F1 Score would improve from 82.4% → 87-90% (+5-8 points)

**Actual**: F1 Score **decreased** from 82.4% → 80.9% (-1.5 points)

### Root Cause: GMM Variability

The Gaussian Mixture Model (GMM) fitted differently than the baseline run, resulting in dramatically different thresholds:

| Threshold | Baseline | Tier 1 | Impact |
|-----------|----------|--------|--------|
| **T_HIGH** | 58.2% | 38.3% | 🔴 **-19.9 points** (too permissive) |
| **GMM "Same" Cluster Mean** | 87.8% | 72.7% | 🔴 Lower quality fit |

The combination of:
1. **Lowering posterior probability requirement** (0.98 → 0.92)
2. **Worse GMM fit** (cluster mean dropped from 87.8% → 72.7%)

...resulted in T_HIGH=38.3%, which accepted almost everything and caused **+44 false positives**.

---

## Improvements Implemented

### 1. GMM Threshold Adjustment
- **File**: `backend/app/services/gmm_threshold_service.py` (line 173)
- **Change**: `target_prob=0.98` → `target_prob=0.92`
- **Result**: ❌ **Counterproductive** - Caused over-grouping

### 2. Acronym Detection
- **File**: `backend/app/services/name_matcher.py` (lines 224-281)
- **Change**: Added initials-matching with +15 point boost
- **Result**: ⚠️ **Insufficient impact** - Drowned out by threshold issues

### 3. Multi-Token Requirement
- **File**: `backend/app/services/name_matcher.py` (lines 368-378)
- **Change**: 25% penalty for short names with <2 matching tokens
- **Result**: ❌ **Too restrictive** - Prevented legitimate matches

---

## Key Lessons

### 1. GMM is Unstable
- Different runs produce different cluster fits
- Need consistent initialization (random_state + preprocessing)
- Consider alternative: Hierarchical Agglomerative Clustering (HAC)

### 2. Test Incrementally
- Should have tested each improvement separately
- All 3 at once made it impossible to isolate issues

### 3. Precision Loss is Costly
- +44 false positives (-3.6% precision) negated any recall gains
- Need to balance precision-recall trade-off more carefully

---

## Recommendations

### Option 1: Revert and Stabilize GMM (RECOMMENDED)
1. **Revert all Tier 1 changes** to baseline
2. **Fix GMM initialization**: Add `random_state=42` to preprocessing
3. **Test improvements incrementally**:
   - Acronym detection only (1st)
   - Threshold adjustment (2nd, with stabilized GMM)
   - Multi-token requirement (3rd, with lighter penalty)

### Option 2: Try Alternative Clustering
Based on [ML_APPROACHES_RESEARCH_REPORT.md](ML_APPROACHES_RESEARCH_REPORT.md):

1. **Hierarchical Agglomerative Clustering (HAC)**
   - More stable than GMM
   - Global optimization (no order dependence)
   - Expected: +5-7% F1 improvement
   - Effort: 2-3 weeks

2. **Graph-Based Transitive Closure**
   - Captures A→B→C relationships GMM misses
   - Expected: +5-7% F1 improvement
   - Effort: 2-3 weeks

### Option 3: Tune More Conservatively
1. **Lighter threshold adjustment**: 0.98 → 0.95 (not 0.92)
2. **Lighter multi-token penalty**: 0.85 (15% penalty, not 25%)
3. **Stronger acronym boost**: +20 points (not +15)

---

## File Organization

### Baseline Results
**Location**: `backend/test_results/baseline/2025-10-27_baseline/`
- F1: 82.42%
- Precision: 93.00%
- Recall: 74.00%

### Tier 1 Results
**Location**: `backend/test_results/tier1_improvements/2025-10-27_tier1/`
- F1: 80.89% ❌
- Precision: 89.40% ❌
- Recall: 73.86% ❌

### Detailed Analysis
See: `backend/test_results/tier1_improvements/2025-10-27_tier1/COMPARISON_SUMMARY.md`

---

## Next Steps

**Decision Point**: Choose one of the 3 options above

**If Option 1 (Revert + Stabilize):**
1. Revert changes in `gmm_threshold_service.py` and `name_matcher.py`
2. Add GMM stability fixes
3. Re-test baseline to confirm consistency
4. Implement improvements one at a time

**If Option 2 (Alternative Clustering):**
1. Research HAC implementation (scipy)
2. Create prototype with sample data
3. Compare HAC vs GMM on same dataset
4. Proceed with winner

**If Option 3 (Tune Parameters):**
1. Adjust parameters as specified above
2. Re-run test
3. Iterate until F1 > 85%

---

**For detailed analysis, see:**
- Full comparison: [COMPARISON_SUMMARY.md](backend/test_results/tier1_improvements/2025-10-27_tier1/COMPARISON_SUMMARY.md)
- Test results: `backend/test_results/tier1_improvements/2025-10-27_tier1/`
- ML research: [ML_APPROACHES_RESEARCH_REPORT.md](ML_APPROACHES_RESEARCH_REPORT.md)

**Status**: Awaiting decision on next approach
