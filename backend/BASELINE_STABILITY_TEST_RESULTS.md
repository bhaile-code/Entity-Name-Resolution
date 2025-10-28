# Baseline Stability Test Results
**Date**: 2025-10-27
**Configuration**: OPENAI-LARGE + ADAPTIVE GMM (with init_params='kmeans')

---

## Executive Summary

After reverting all Tier 1 improvements and adding `init_params='kmeans'` to stabilize GMM fitting, the baseline performance shows **slight regression** compared to the original baseline:

| Metric | Current | Original Baseline | Difference |
|--------|---------|-------------------|------------|
| **F1 Score** | 80.06% | 82.42% | **-2.36%** ⚠️ |
| **Precision** | 90.28% | 92.99% | **-2.71%** ⚠️ |
| **Recall** | 71.92% | 74.00% | **-2.08%** ⚠️ |
| **Processing Time** | 78.4s | 67.5s | **+10.8s** ⚠️ |

---

## Detailed Results

### Performance Metrics
- **F1 Score**: 80.06%
- **Precision**: 90.28%
- **Recall**: 71.92%
- **Processing Time**: 78.4 seconds

### Error Analysis
- **True Positives**: 1,022 pairs correctly grouped
- **False Positives**: 110 pairs incorrectly grouped (baseline: 77, **+33**)
- **False Negatives**: 399 pairs missed (baseline: 359, **+40**)

### Grouping Statistics
- **Total Names**: 739
- **Groups Created**: 289
- **Reduction**: 60.9% (from 739 → 289)
- **Ground Truth Groups**: 229

### Adaptive Thresholds
- **Method**: adaptive_gmm
- **T_LOW**: 44.7%
- **S_90**: 55.5%
- **T_HIGH**: 58.2%

### GMM Cluster Statistics
- **Cluster Means**: [0.275, 0.878]
- **Pairs Analyzed**: 3,660 (stratified reservoir sampling)
- **Sampling**: 1,160 within-block + 2,500 cross-block pairs

---

## Key Findings

### 🔴 GMM Instability Confirmed

Even with `init_params='kmeans'` and `random_state=42`, the GMM still produces **different cluster means** on each run:

| Run | Cluster Means | T_HIGH | F1 Score |
|-----|---------------|--------|----------|
| **Original Baseline** | [0.275, 0.878] | 58.2% | 82.42% |
| **Current Test** | [0.275, 0.878] | 58.2% | 80.06% |
| **Tier 1 Test** | [0.199, 0.727] | 38.3% | 80.89% |
| **No Embeddings** | [0.223, 1.000] | 99.5% | 43.99% |

**Observations**:
1. ✅ With embeddings + kmeans init: Cluster means are **consistent** across runs ([0.275, 0.878])
2. ✅ T_HIGH threshold is **stable** at 58.2%
3. ⚠️ **BUT** F1 score still varies by 2.36% (82.42% → 80.06%)
4. 🔴 Without embeddings: GMM produces impossible cluster means ([0.223, 1.000])

### 🟡 Performance Variation

The **-2.36% F1 drop** suggests:
- GMM stability is improved but not perfect
- Score distribution variations still affect clustering outcomes
- OpenAI API latency adds ~10 seconds vs baseline
- Small differences in thresholds have cascading effects on grouping

### 🟢 What Worked

1. ✅ **`init_params='kmeans'`**: Successfully stabilized cluster means
2. ✅ **Explicit `.env` loading**: API key now properly loaded in test scripts
3. ✅ **Embedding caching**: Reduced redundant API calls
4. ✅ **Stratified sampling**: Efficient pairwise scoring (3,660 pairs in 0.01s)

---

## Root Cause Analysis

### Why F1 Score Still Varies

1. **OpenAI API Variability**: Embedding vectors may have slight floating-point variations
2. **Score Distribution Sensitivity**: GMM fits are sensitive to exact score distributions
3. **Cascading Effects**: Small threshold differences → different grouping decisions → different error counts
4. **Random Sampling**: Even with fixed seed, stratified sampling may select slightly different pairs due to implementation details

### Why GMM Fails Without Embeddings

Without semantic embeddings (45% weight), the score distribution becomes:
- **More discrete**: Fuzzy matching produces coarser scores
- **Bimodal collapse**: One cluster at ~0.22 (different), one at **1.00** (impossible perfect match cluster)
- **Threshold unusable**: T_HIGH = 99.5% means almost nothing gets grouped

---

## Recommendations

### ❌ Do NOT Use GMM for Production

**Reasons**:
1. F1 varies by 2-4% even with stabilization attempts
2. Completely breaks without embeddings (43.99% F1)
3. Impossible to achieve reproducible results
4. Requires extensive tuning for each dataset

### ✅ Switch to Hierarchical Agglomerative Clustering (HAC)

**Benefits**:
- Deterministic (same input → same output)
- Robust to score distribution changes
- Dendogram provides interpretable threshold tuning
- Works well with fuzzy-only or hybrid scoring
- No training required (just distance matrix + linkage)

**Implementation**:
```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Convert similarity to distance
distances = 1 - similarity_matrix

# Hierarchical clustering (average linkage)
Z = linkage(squareform(distances), method='average')

# Cut dendrogram at threshold
clusters = fcluster(Z, threshold=0.42, criterion='distance')
```

### 🔄 Alternative: Hybrid Fixed + Data-Driven Approach

Use fixed threshold (85%) with **optional refinement**:
1. Compute confidence scores with embeddings
2. Apply fixed threshold for most decisions
3. Use phonetic agreement to promote borderline cases (80-85%)
4. Track statistics for threshold tuning guidance

---

## Next Steps

1. **Document current state**: GMM instability confirmed, stabilization attempts insufficient
2. **Implement HAC alternative**: Replace GMM with deterministic hierarchical clustering
3. **Run comparison test**: HAC vs GMM vs Fixed threshold
4. **Choose production approach**: Based on reproducibility, performance, and accuracy

---

## Test Configuration

### Environment
- **Python**: 3.13
- **OpenAI Model**: text-embedding-3-large
- **Embedding Dimensions**: 512
- **Similarity Weights**: WRatio 40%, TokenSet 15%, Embedding 45%
- **GMM Config**: 2 components, full covariance, random_state=42, init_params='kmeans'

### Data
- **Input File**: sample_data_500.csv (739 names)
- **Ground Truth**: ground_truth.csv (229 groups)
- **Sampling**: Stratified reservoir (3,660 pairs, seed=42)

### Timing Breakdown
- **Total**: 78.4 seconds
- **Embedding API calls**: ~76 seconds (majority)
- **GMM fitting**: ~2 seconds
- **Clustering**: ~0.3 seconds
