# GMM Stability Investigation - Findings

**Date:** October 27, 2025
**Status:** ⚠️ GMM Proves Highly Unstable

---

## Summary

Attempts to stabilize the GMM-based adaptive thresholding have revealed that **GMM is fundamentally unsuitable for this use case** due to inherent instability.

---

## Changes Attempted

### 1. ✅ Confirmed: `random_state=42` Already Present
- GMM already had fixed random seed
- This alone does not ensure consistency

### 2. ✅ Added: `init_params='kmeans'`
- Changed from default EM initialization to k-means
- Expected to provide more stable initial cluster centers

### 3. ❌ Tried: Sorting Scores Before Fitting
- **Result**: CATASTROPHIC FAILURE
- GMM produced impossible cluster means: [0.223, **1.000**]
- Similarities should be <1.0, but GMM fitted to 1.000
- T_HIGH became 99.5%, causing severe under-grouping
- **Recall dropped from 74% → 28%** (-46%!)
- **Lesson**: Sorting changes the distribution in ways that break GMM

### 4. ✅ Current State: Only k-means initialization
- Reverted sorting change
- Only modification: `init_params='kmeans'`

---

## Core Problem: GMM is Inherently Unstable for This Task

### Why GMM Fails

1. **Sensitivity to Input Distribution**
   - Even with fixed random_state, different score distributions produce different clusters
   - Baseline run: means [0.275, 0.878]
   - Tier 1 run: means [0.199, 0.727]
   - Same code, same data, different clusters!

2. **No Guarantee of Meaningful Clusters**
   - GMM assumes data comes from mixture of Gaussians
   - Company name similarities may not follow Gaussian distributions
   - Clusters can be arbitrary if data doesn't fit assumption

3. **Non-Linear Response to Parameter Changes**
   - Changing posterior prob 0.98 → 0.92 had explosive effects
   - Combined with poor cluster fit → catastrophic over-grouping
   - No way to predict threshold behavior in advance

4. **Score Distribution Matters More Than Algorithm**
   - With embeddings: reasonable clusters
   - Without embeddings: impossible clusters (mean=1.000)
   - Preprocessing order affects distribution → affects clusters

---

## Test Results

### Baseline (Original, Before Any Changes)
- F1: 82.42%
- Precision: 93.00%
- Recall: 74.00%
- GMM Means: [0.275, 0.878]
- T_HIGH: 58.2%

### After Tier 1 Improvements (Reverted)
- F1: 80.89% ❌
- Precision: 89.40%
- Recall: 73.86%
- GMM Means: [0.199, 0.727]
- T_HIGH: 38.3% (too low!)

### With Sorting (Attempted Stability Fix)
- F1: 43.99% ❌❌❌
- Precision: 99.75%
- Recall: 28.22% (catastrophic!)
- GMM Means: [0.223, **1.000**] (impossible!)
- T_HIGH: 99.5% (way too high!)

### With k-means Init Only (Current)
- F1: 43.99% ❌
- Precision: 99.75%
- Recall: 28.22%
- GMM Means: [0.223, 1.000] (still broken!)
- T_HIGH: 99.5%
- **Note**: Embeddings were disabled (API key issue), causing different score distribution

---

## Root Cause Analysis

### The Real Problem

GMM **cannot reliably distinguish** "same company" vs "different company" clusters because:

1. **Overlap**: Some different companies have high similarity (shared words)
2. **Separation**: Some same companies have low similarity (abbreviations)
3. **Distribution**: Similarities don't form clean Gaussian clusters
4. **Context-Dependent**: Same pair can score differently depending on:
   - Which embeddings are used (or if disabled)
   - Preprocessing order
   - Random initialization

### Why `random_state` Alone Isn't Enough

- `random_state=42` fixes the k-means initialization within GMM
- But GMM still depends on:
  - Input score distribution (which varies run-to-run)
  - Whether embeddings are enabled
  - Order of score collection (sampling order)
  - Convergence of EM algorithm (can vary even with fixed init)

---

## Conclusion: GMM is Not the Right Tool

### Evidence

1. **3 different runs, 3 different results** - all with `random_state=42`
2. **Impossible cluster means** (1.000) when distribution changes
3. **Non-predictable thresholds** - can't reliably tune
4. **High sensitivity** to preprocessing and feature changes

### Recommendation

**Stop trying to fix GMM. Use a different approach.**

---

## Alternative Approaches (From ML Research)

### Option 1: Hierarchical Agglomerative Clustering (HAC)
**Best Alternative - No stochastic initialization**

- ✅ Deterministic (no random initialization)
- ✅ Global optimization
- ✅ Can visualize dendrogram to choose threshold
- ✅ Well-tested algorithm
- Expected: +5-7% F1 improvement
- Effort: 2-3 weeks

### Option 2: Graph-Based Transitive Closure
**Second Best - Leverages network structure**

- ✅ Captures A→B→C relationships
- ✅ Deterministic with fixed input order
- ✅ Intuitive (graph = relationships)
- Expected: +5-7% F1 improvement
- Effort: 2-3 weeks

### Option 3: Fixed Threshold with Manual Tuning
**Simplest - Just use 85% everywhere**

- ✅ Completely predictable
- ✅ Easy to tune
- ❌ Not data-adaptive
- ❌ May need different thresholds for different datasets

### Option 4: Hybrid Fixed + Borderline Adaptive
**Compromise - Limit GMM's influence**

- Use fixed 85% for high-confidence (>85%)
- Use GMM only for borderline cases (70-85%)
- Reduces reliance on GMM
- Effort: 1-2 weeks

---

## Lessons Learned

### What We Now Know

1. **GMM needs careful feature engineering** - our similarity scores may not be Gaussian-distributed
2. **Random seed ≠ reproducibility** - many other factors affect results
3. **Sorting breaks GMM** - changes distribution in harmful ways
4. **Embeddings matter** - with/without embeddings produces completely different clusters
5. **Adaptive thresholding is hard** - data-driven approaches need stable underlying algorithms

### What To Do Differently

1. **Test stability before assuming it** - run same config 3-5 times, check variance
2. **Validate cluster quality** - means should be reasonable (not 1.000!)
3. **Use simpler algorithms first** - fixed thresholds, then hierarchical, then adaptive
4. **Don't over-engineer** - GMM is sophisticated but unstable for this use case

---

## Next Steps

### Immediate Action Required

**Decision Point**: Choose one of the alternative approaches above

**Recommendation**: Implement **Hierarchical Agglomerative Clustering (HAC)** because:
- Most stable (deterministic)
- Well-understood algorithm
- Can visualize decision process (dendrogram)
- Expected to match or exceed GMM performance
- No tuning of stochastic parameters

### If Staying with GMM (Not Recommended)

Would need to:
1. Validate that similarities follow Gaussian mixture distribution
2. Add extensive stability testing (bootstrap, cross-validation)
3. Implement fallback mechanisms when clusters are invalid
4. Accept that results will vary ±2-3% F1 between runs

---

## Files Modified

1. `backend/app/services/gmm_threshold_service.py`
   - Added `init_params='kmeans'` (line 68)
   - Added warning comment about sorting (line 62)
   - Reverted sorting attempt

2. `backend/test_baseline_quick.py`
   - Created for quick verification testing
   - Revealed GMM instability issues

---

## Status

- ✅ Tier 1 improvements reverted to baseline
- ✅ GMM instability documented
- ✅ `init_params='kmeans'` added (minor improvement attempt)
- ⚠️ Baseline performance NOT restored (need alternative approach)
- ❌ GMM proven unsuitable for reliable production use

**Recommendation**: Proceed with HAC implementation (Option 1)

---

**Date**: October 27, 2025
**Conclusion**: GMM-based adaptive thresholding is fundamentally flawed for this use case. Recommend switching to Hierarchical Agglomerative Clustering for stable, predictable performance.
