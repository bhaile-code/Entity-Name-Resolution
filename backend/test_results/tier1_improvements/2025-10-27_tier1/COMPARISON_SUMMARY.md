# Tier 1 Improvements - Performance Comparison Summary

**Test Date:** October 27, 2025
**Configuration Tested:** OPENAI-LARGE + ADAPTIVE (Best performing configuration)
**Dataset:** 739 company names, 229 ground truth groups

---

## Executive Summary

**Unexpected Result:** The Tier 1 improvements resulted in **slightly lower performance** than baseline, contrary to expectations based on false negative analysis.

### Key Findings

| Metric | Baseline | Tier 1 | Change | Status |
|--------|----------|--------|--------|--------|
| **F1 Score** | **82.42%** | **80.89%** | **-1.53%** | ⚠️ **Regression** |
| **Precision** | 93.00% | 89.40% | -3.60% | ⚠️ Decreased |
| **Recall** | 74.00% | 73.86% | -0.14% | ⚠️ Nearly unchanged |
| **Processing Time** | 67.5s | 85.5s | +18.0s (+27%) | ⚠️ Slower |
| **False Positives** | 77 | 121 | +44 (+57%) | ⚠️ Increased |
| **False Negatives** | 359 | 361 | +2 (+0.6%) | ≈ Similar |

### What Went Wrong?

1. **GMM Threshold Too Aggressive**: Reducing from 0.98 → 0.92 caused T_HIGH to drop significantly (58.2% → 38.3%), resulting in over-grouping
2. **Multi-Token Penalty Too Harsh**: The 25% penalty prevented many legitimate matches
3. **Acronym Boost Insufficient**: The +15 point boost wasn't enough to overcome the threshold issues
4. **Increased False Positives**: +44 FPs negated any recall gains, hurting precision significantly

---

## Detailed Metrics Comparison

### Confusion Matrix

| | Baseline | Tier 1 | Change |
|---|---|---|---|
| **True Positives** | 1,022 | 1,020 | -2 |
| **True Negatives** | 262,443 | 262,399 | -44 |
| **False Positives** | 77 | 121 | +44 ⚠️ |
| **False Negatives** | 359 | 361 | +2 |
| **Total Pairs** | 263,901 | 263,901 | - |

### Grouping Quality

| Metric | Baseline | Tier 1 | Change |
|--------|----------|--------|--------|
| **Purity** | 99.46% | 99.11% | -0.35% |
| **Completeness** | 93.49% | 93.38% | -0.11% |
| **Groups Created** | 289 | 285 | -4 |

---

## Improvements Implemented

### 1. GMM Threshold Adjustment ❌ **Counterproductive**
- **Change**: Reduced posterior probability requirement from 0.98 → 0.92
- **File**: `backend/app/services/gmm_threshold_service.py` (line 173)
- **Expected**: +8-10% recall
- **Actual**: -0.14% recall, -3.6% precision
- **Analysis**: The GMM fitted differently on this run, resulting in T_HIGH=38.3% instead of expected ~58%. This was too aggressive and caused over-grouping.

### 2. Acronym Detection ⚠️ **Insufficient Impact**
- **Change**: Added initials-matching logic with +15 point boost
- **File**: `backend/app/services/name_matcher.py` (lines 224-281)
- **Expected**: +5-8% recall
- **Actual**: No measurable impact
- **Analysis**: The boost wasn't large enough to overcome the lowered threshold, or acronym pairs were already being caught by embeddings.

### 3. Multi-Token Requirement ❌ **Too Restrictive**
- **Change**: 25% penalty for short names with <2 matching tokens
- **File**: `backend/app/services/name_matcher.py` (lines 368-378)
- **Expected**: +1-2% precision
- **Actual**: Likely contributed to recall drop
- **Analysis**: May have prevented legitimate short-name matches.

---

## Root Cause Analysis

### Why Did This Happen?

1. **GMM Variability**: The Gaussian Mixture Model fitting is non-deterministic and sensitive to input data order. Even with the same code, different runs can produce different cluster means and thresholds.

   **Baseline GMM**:
   - Cluster means: 0.275 (different), 0.878 (same)
   - T_HIGH: 58.2% (requires P(same|score) = 0.98)

   **Tier 1 GMM**:
   - Cluster means: 0.199 (different), 0.727 (same)
   - T_HIGH: 38.3% (requires P(same|score) = 0.92)

   The combination of lowering the posterior probability threshold (0.98 → 0.92) AND getting a worse GMM fit (lower "same" cluster mean: 0.878 → 0.727) resulted in an extremely low T_HIGH threshold of 38.3%.

2. **Threshold Too Low**: T_HIGH=38.3% accepts nearly everything, causing the +44 false positives and hurting precision.

3. **No Recall Benefit**: Despite the lower threshold, recall barely improved (-0.14%), suggesting the multi-token penalty and other factors prevented gains.

---

## Lessons Learned

### ❌ What Didn't Work

1. **Blindly adjusting GMM parameters** without accounting for cluster variability
2. **Assuming linear impact** - reducing posterior prob from 0.98 → 0.92 had non-linear effects
3. **Not testing improvements incrementally** - implementing all 3 at once made it hard to isolate issues
4. **Over-reliance on false negative analysis** from a single run without considering GMM variance

### ✅ What We Learned

1. **GMM is unstable**: Need to either:
   - Fix the cluster initialization (set random_state + consistent preprocessing)
   - Use multiple runs and average thresholds
   - Consider hybrid fixed + adaptive approach

2. **Need incremental testing**: Test each improvement separately to isolate effects

3. **False positive cost is high**: Going from 77 → 121 FPs (-3.6% precision) outweighed any recall gains

---

## Recommended Next Steps

### Option 1: Revert and Re-test (RECOMMENDED)
1. **Revert all changes** to baseline
2. **Fix GMM initialization**: Add consistent random_state and data preprocessing
3. **Test ONE improvement at a time**:
   - First: Acronym detection only
   - Then: Threshold adjustment with fixed GMM
   - Finally: Multi-token requirement

### Option 2: Tune Parameters
1. **Increase GMM posterior probability target**: Try 0.95 instead of 0.92 (less aggressive)
2. **Reduce multi-token penalty**: Change from 0.75 → 0.85 (15% penalty instead of 25%)
3. **Increase acronym boost**: Change from +15 → +20 points

### Option 3: Alternative Approaches (From ML Research)
1. **Hierarchical Agglomerative Clustering**: Replace GMM-based threshold with HAC
2. **Graph-Based Transitive Closure**: Add post-processing step to capture A→B→C relationships
3. **Hybrid Fixed + Adaptive**: Use fixed threshold (85%) for high-confidence, GMM for borderline cases

---

## Detailed Adaptive Thresholds

### Baseline Run
- **T_LOW**: 44.7% (P(same|score) = 0.02)
- **S_90**: 55.5% (P(same|score) = 0.90)
- **T_HIGH**: 58.2% (P(same|score) = 0.98)
- **GMM Means**: [0.275, 0.878]

### Tier 1 Run
- **T_LOW**: 0.0% (P(same|score) = 0.02)
- **S_90**: 38.0% (P(same|score) = 0.90)
- **T_HIGH**: 38.3% (P(same|score) = 0.92)
- **GMM Means**: [0.199, 0.727]

**Critical Issue**: T_HIGH dropped by 19.9 percentage points (58.2% → 38.3%), making the system far too permissive.

---

## Performance by Configuration (All 4 Tested)

| Configuration | F1 (Baseline) | F1 (Tier 1) | Change |
|---------------|---------------|-------------|--------|
| **OPENAI-LARGE + ADAPTIVE** | **82.42%** | **80.89%** | **-1.53%** |
| OPENAI-SMALL + ADAPTIVE | 81.41% | 80.73% | -0.68% |
| OPENAI-LARGE + FIXED | 76.05% | 48.24% | -27.81% ⚠️ |
| OPENAI-SMALL + FIXED | 73.63% | 48.11% | -25.52% ⚠️ |

**Note**: Fixed threshold modes were severely impacted, likely due to multi-token penalty and acronym boost interfering with fixed thresholding logic.

---

## Conclusion

The Tier 1 improvements **did not achieve the expected performance gains**. The primary issue was **GMM variability** combined with an overly aggressive threshold adjustment that resulted in too many false positives.

### Key Takeaways

1. **GMM-based adaptive thresholding is inherently unstable** without proper initialization
2. **Need to test improvements incrementally** to isolate effects
3. **Precision loss is expensive** - the +44 false positives negated any recall benefits
4. **Alternative approaches** (HAC, Graph-Based) may be more stable than GMM tuning

### Recommendation

**Revert to baseline** and implement a **stabilized GMM approach** OR explore **alternative clustering algorithms** (Hierarchical Agglomerative Clustering, Graph-Based Transitive Closure) that don't depend on stochastic initialization.

---

## Files Reference

**Baseline Results**: `test_results/baseline/2025-10-27_baseline/`
- performance_summary.csv
- performance_detailed_reports.md
- performance_results.json

**Tier 1 Results**: `test_results/tier1_improvements/2025-10-27_tier1/`
- performance_summary.csv
- performance_detailed_reports.md
- performance_results.json
- test_output.txt (full test log)

**Modified Files**:
- `backend/app/services/gmm_threshold_service.py` (line 173)
- `backend/app/services/name_matcher.py` (lines 224-393)

---

**Analysis Date**: October 27, 2025
**Next Review**: After stabilizing GMM or exploring alternative approaches
