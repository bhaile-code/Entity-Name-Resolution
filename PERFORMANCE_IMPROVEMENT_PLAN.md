# Performance Improvement Plan
## Entity Name Resolution System

**Document Version**: 1.0
**Date**: 2025-10-24
**Current Performance**: F1 Score 82.4%, Precision 93.0%, Recall 74.0%
**Target Performance**: F1 Score 88-93%

---

## Executive Summary

This document outlines a phased approach to improve the entity name resolution system from **82.4% F1** to **88-93% F1** without requiring any machine learning training. All improvements are rule-based, parameter adjustments, or algorithmic changes that can be implemented in 2-6 weeks.

### Current Performance Analysis

**Best Configuration**: OPENAI-LARGE + ADAPTIVE
- **F1 Score**: 82.4%
- **Precision**: 93.0% (77 false positives)
- **Recall**: 74.0% (359 false negatives)
- **Processing Time**: 67.5 seconds
- **Groups Created**: 289 (vs 229 ground truth)

### Key Findings from Analysis

1. **False Negatives (359 pairs - Missing 26% of matches)**
   - 48.4% OTHER: High-scoring pairs (80-100%) rejected due to conservative threshold
   - 36.1% ABBREVIATION: "IBM" ↔ "International Business Machines"
   - 11.0% NICKNAME: "AmEx" ↔ "American Express"
   - 10.0% TRANSLITERATION: Already handled well

2. **False Positives (77 pairs - 7% incorrect merges)**
   - 87% SHARED WORD: "American Airlines" ↔ "American Express"
   - 15.6% SUBSET: "Adobe" ↔ "Adobe Rent-A-Car"
   - Root cause: Token_set (100%) and WRatio (90%) override low embedding scores (43%)

3. **Root Cause**: GMM threshold is too conservative (T_HIGH requires 98% certainty)

---

## Three-Phase Implementation Plan

### Phase 1: Quick Wins (Week 1-2) ⚡ **PRIORITY: HIGHEST**

**Goal**: Achieve 88-90% F1 in 1-2 weeks with minimal risk

#### Improvement 1.1: Dynamic Threshold Adjustment

**Problem**: 150 pairs (37.6% of false negatives) scored 80-100% but were rejected

**Solution**: Lower GMM auto-accept threshold from 98% to 92% certainty

**Files to Modify**:
```
backend/app/services/gmm_threshold_service.py
```

**Code Change**:
```python
# Line ~45 (in calculate_adaptive_thresholds method)
# OLD:
t_high = self._find_threshold_for_posterior(scores, weights, means, covariances, target_prob=0.98)

# NEW:
t_high = self._find_threshold_for_posterior(scores, weights, means, covariances, target_prob=0.92)
```

**Expected Impact**:
- Recall: +8-12% (74% → 82-86%)
- Precision: -2-3% (93% → 90-91%)
- F1: +5-7% (82.4% → 87-89%)
- Effort: 2-4 hours
- Risk: Low

**Testing**:
```bash
cd backend
python test_and_analyze_performance.py
# Verify F1 improvement and acceptable precision trade-off
```

---

#### Improvement 1.2: Multi-Token Requirement for Short Names

**Problem**: "Adobe" matches "Adobe Rent-A-Car" with 80.3% confidence (12 false positives)

**Solution**: Require at least 2 matching meaningful tokens for names with ≤2 tokens

**Files to Modify**:
```
backend/app/services/name_matcher.py
```

**Code Change**:
```python
# Add new method to NameMatcher class (around line 200)

def _count_matching_tokens(self, name1: str, name2: str) -> int:
    """Count meaningful matching tokens between two names."""
    # Normalize and tokenize
    tokens1 = set(self.normalize_name(name1).split())
    tokens2 = set(self.normalize_name(name2).split())

    # Remove common stopwords
    stopwords = {'the', 'and', 'of', 'a', 'an', 'for', 'in', 'on'}
    tokens1 = {t for t in tokens1 if t not in stopwords and len(t) > 2}
    tokens2 = {t for t in tokens2 if t not in stopwords and len(t) > 2}

    return len(tokens1 & tokens2)

# Modify calculate_confidence method (around line 180)
def calculate_confidence(self, name1: str, name2: str, norm1: str, norm2: str) -> float:
    # ... existing code ...

    # NEW: Multi-token requirement for short names
    token_count1 = len([t for t in norm1.split() if len(t) > 2])
    token_count2 = len([t for t in norm2.split() if len(t) > 2])

    if token_count1 <= 2 or token_count2 <= 2:
        matching_tokens = self._count_matching_tokens(name1, name2)
        if matching_tokens < 2:
            # Apply penalty for insufficient token matches
            base_score *= 0.75  # 25% penalty

    # ... rest of existing code ...
    return final_score
```

**Expected Impact**:
- Recall: -0.5% (small cost)
- Precision: +1-2% (93% → 94-95%)
- F1: +1% (net positive)
- Effort: 4-6 hours
- Risk: Low

**Testing**:
```python
# Test cases
assert calculate_confidence("Adobe", "Adobe Rent-A-Car") < 75.0  # Should reject
assert calculate_confidence("Adobe Inc.", "Adobe Systems Inc.") > 85.0  # Should match
assert calculate_confidence("IBM", "IBM Corp") > 85.0  # Should still match
```

---

#### Improvement 1.3: Acronym/Abbreviation Detection

**Problem**: 129 false negatives (36% of errors) are abbreviation pairs

**Solution**: Add fuzzy acronym detection to boost confidence

**Files to Modify**:
```
backend/app/services/name_matcher.py
```

**Code Change**:
```python
# Add new method to NameMatcher class (around line 250)

def _detect_acronym_match(self, name1: str, name2: str) -> bool:
    """
    Detect if one name is an acronym of another.
    Examples: "IBM" ↔ "International Business Machines"
              "GE" ↔ "General Electric"
    """
    # Normalize names
    n1 = self.normalize_name(name1).replace(' ', '')
    n2 = self.normalize_name(name2).replace(' ', '')

    # Check if one is much shorter (potential acronym)
    short, long = (n1, n2) if len(n1) < len(n2) else (n2, n1)

    # Acronym must be 2-5 characters
    if not (2 <= len(short) <= 5):
        return False

    # Check if short name matches initials of long name
    long_tokens = name2.split() if len(n1) < len(n2) else name1.split()
    long_tokens = [t for t in long_tokens if len(t) > 2]  # Skip short words

    if len(long_tokens) < len(short):
        return False

    # Extract initials from long name
    initials = ''.join([t[0].lower() for t in long_tokens])

    # Check exact match or fuzzy match (allowing 1 character difference)
    if short.lower() == initials[:len(short)]:
        return True

    # Fuzzy match: allow 1 character mismatch
    if len(short) >= 3:
        matches = sum(1 for i, c in enumerate(short.lower()) if i < len(initials) and c == initials[i])
        if matches >= len(short) - 1:
            return True

    return False

# Modify calculate_confidence method (around line 180)
def calculate_confidence(self, name1: str, name2: str, norm1: str, norm2: str) -> float:
    # ... existing code ...

    # NEW: Acronym boost
    if self._detect_acronym_match(name1, name2):
        base_score += 15.0  # Significant boost for acronym matches
        logger.debug(f"Acronym match detected: {name1} ↔ {name2}")

    # ... rest of existing code ...
    return final_score
```

**Expected Impact**:
- Recall: +5-8% (74% → 79-82%)
- Precision: 0% (no impact)
- F1: +3-5% (82.4% → 85-87%)
- Effort: 1-2 days
- Risk: Low

**Testing**:
```python
# Test cases
assert _detect_acronym_match("IBM", "International Business Machines") == True
assert _detect_acronym_match("GE", "General Electric") == True
assert _detect_acronym_match("AmEx", "American Express") == False  # Not initials
assert _detect_acronym_match("AAPL", "Apple Inc.") == False  # Stock ticker, not acronym
```

---

### Phase 1 Summary

**Combined Expected Results**:
- **Recall**: 74% → 84-88% (+10-14%)
- **Precision**: 93% → 90-92% (-1 to -3%)
- **F1 Score**: 82.4% → **88-90%** (+6-8%)
- **Total Effort**: 2-3 days of development
- **Risk**: Low
- **Cost**: $0
- **Training Required**: None ✅

**Decision Point**: After Phase 1, evaluate if 88-90% F1 meets production requirements. If yes, **STOP HERE**. If you need 90%+, proceed to Phase 2.

---

## Phase 2: Root Cause Fix (Week 3-4) 🔧 **PRIORITY: MEDIUM**

**Goal**: Push F1 from 88-90% to 90-91% by fixing shared word problem

#### Improvement 2.1: Dynamic Component Weight Adjustment

**Problem**: Token_set (100%) and WRatio (90%) override low embedding scores (43%), causing "American Airlines" ↔ "American Express" to match

**Solution**: Adaptively reduce token_set weight and increase embedding weight when shared words detected

**Files to Modify**:
```
backend/app/services/name_matcher.py
backend/app/config/settings.py
```

**Code Changes**:

1. Add shared word detection:
```python
# Add to NameMatcher class (around line 280)

def _detect_shared_common_words(self, norm1: str, norm2: str) -> bool:
    """
    Detect if names share common words that could cause false positives.
    Examples: "American Airlines" vs "American Express" (shared: "American")
    """
    # Common business words that cause false positives
    common_words = {
        'american', 'united', 'general', 'international', 'national',
        'global', 'delta', 'first', 'capital', 'bank', 'group',
        'holdings', 'services', 'corporation', 'company', 'technologies'
    }

    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    # Check if they share a common word and have other different words
    shared = tokens1 & tokens2 & common_words
    different = (tokens1 ^ tokens2) - common_words

    # If they share a common word but have 2+ different meaningful words
    if shared and len(different) >= 2:
        return True

    return False
```

2. Modify calculate_confidence with dynamic weights:
```python
# Modify calculate_confidence method (around line 180)
def calculate_confidence(self, name1: str, name2: str, norm1: str, norm2: str) -> float:
    # Calculate component scores
    wratio = fuzz.WRatio(norm1, norm2)
    token_set = fuzz.token_set_ratio(norm1, norm2)

    # Get embedding score if available
    embedding_score = 0.0
    if self.embedding_service:
        embedding_score = self.embedding_service.calculate_similarity(name1, name2) * 100

    # NEW: Detect shared word problem
    has_shared_words = self._detect_shared_common_words(norm1, norm2)

    # NEW: Dynamic weight adjustment
    if has_shared_words:
        # Reduce token_set weight, increase embedding weight
        wratio_weight = 0.30  # Reduced from 0.40
        token_set_weight = 0.10  # Reduced from 0.15
        embedding_weight = 0.60  # Increased from 0.45
        logger.debug(f"Shared word detected, adjusting weights: {name1} ↔ {name2}")
    else:
        # Use default weights
        wratio_weight = settings.WRATIO_WEIGHT
        token_set_weight = settings.TOKEN_SET_WEIGHT
        embedding_weight = settings.EMBEDDING_WEIGHT

    # Calculate base score with dynamic weights
    base_score = (
        wratio * wratio_weight +
        token_set * token_set_weight +
        embedding_score * embedding_weight
    )

    # NEW: Embedding veto
    # If embedding is low but token/wratio are high, apply penalty
    if (self.embedding_service and
        embedding_score < 60.0 and
        token_set > 90.0 and
        wratio > 85.0):
        base_score *= 0.80  # 20% penalty
        logger.debug(f"Embedding veto applied: embedding={embedding_score:.1f}, token_set={token_set:.1f}")

    # ... rest of existing code (phonetic bonus, etc.) ...
    return final_score
```

**Example Before/After**:
```
"American Airlines" vs "American Express"
BEFORE:
  WRatio=90, token_set=100, embedding=43
  Score = 0.40×90 + 0.15×100 + 0.45×43 = 70.4% → MATCH ❌

AFTER:
  Shared word detected → adjust weights
  Score = 0.30×90 + 0.10×100 + 0.60×43 = 62.8%
  Embedding veto (43 < 60) → 62.8% × 0.80 = 50.2% → REJECT ✅
```

**Expected Impact**:
- Recall: +1-2% (additional matches from better score calibration)
- Precision: +2-3% (93% → 95-96%)
- F1: +2% (88-90% → 90-91%)
- Effort: 1-2 days
- Risk: Low-Medium

**Testing**:
```python
# Test cases for shared words
test_cases = [
    ("American Airlines", "American Express", False),  # Should NOT match
    ("American Airlines", "American Airlines Inc.", True),  # Should match
    ("Delta Air Lines", "Delta Dental", False),  # Should NOT match
    ("General Motors", "General Electric", False),  # Should NOT match
    ("United Airlines", "United Parcel Service", False),  # Should NOT match
]

for name1, name2, expected_match in test_cases:
    score = matcher.calculate_confidence(name1, name2, ...)
    actual_match = score >= 85.0
    assert actual_match == expected_match, f"Failed: {name1} vs {name2}"
```

---

### Phase 2 Summary

**Combined Results (Phase 1 + Phase 2)**:
- **Recall**: 74% → 85-90% (+11-16%)
- **Precision**: 93% → 94-96% (+1-3%)
- **F1 Score**: 82.4% → **90-91%** (+8-9%)
- **Total Effort**: 3-4 days of development
- **Risk**: Low-Medium
- **Cost**: $0
- **Training Required**: None ✅

**Decision Point**: After Phase 2, evaluate if 90-91% F1 meets requirements. If yes, **STOP HERE**. If you need 92%+, proceed to Phase 3.

---

## Phase 3: Architectural Upgrade (Week 5-8) 🚀 **PRIORITY: LOW**

**Goal**: Push F1 from 90-91% to 92-93% via clustering algorithm change

### Only implement if you need 92%+ F1 score

#### Improvement 3.1: Hierarchical Agglomerative Clustering (HAC)

**Problem**: Current greedy clustering misses transitive relationships
- Example: "Microsoft Corp" matches "Microsoft Corporation" (95%)
- "Microsoft Corporation" matches "MSFT" (87%)
- But "Microsoft Corp" vs "MSFT" never compared directly
- Result: 2 separate groups instead of 1

**Solution**: Replace greedy clustering with Hierarchical Agglomerative Clustering

**Files to Modify**:
```
backend/app/services/name_matcher.py
```

**Code Changes**:

```python
# Add import at top
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Add new method to NameMatcher class (around line 400)

def _cluster_with_hac(self, names: List[str], similarity_matrix: np.ndarray, threshold: float) -> Dict[str, List[str]]:
    """
    Cluster names using Hierarchical Agglomerative Clustering.

    Args:
        names: List of company names
        similarity_matrix: NxN matrix of similarity scores (0-100)
        threshold: Minimum similarity threshold (0-100)

    Returns:
        Dictionary mapping canonical names to lists of grouped names
    """
    # Convert similarity to distance (HAC uses distance, not similarity)
    distance_matrix = 100 - similarity_matrix

    # Convert to condensed distance matrix (required by scipy)
    condensed_distances = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    # method='average' = UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    linkage_matrix = linkage(condensed_distances, method='average')

    # Cut dendrogram at threshold to get flat clusters
    distance_threshold = 100 - threshold
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    # Build groups
    groups = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        groups[label].append(names[i])

    # Select canonical name for each group (shortest, fewest words)
    result = {}
    for label, group_names in groups.items():
        canonical = min(group_names, key=lambda n: (len(n), len(n.split()), -sum(1 for c in n if c.isupper())))
        result[canonical] = group_names

    return result

# Modify process_names method to use HAC (around line 450)

def process_names(self, names: List[str], filename: str = "input.csv") -> Dict:
    # ... existing code for threshold calculation ...

    # NEW: Build full similarity matrix
    n = len(unique_names)
    similarity_matrix = np.zeros((n, n))

    logger.info(f"Building similarity matrix for {n} names...")

    for i in range(n):
        similarity_matrix[i, i] = 100.0  # Self-similarity
        for j in range(i + 1, n):
            name1, name2 = unique_names[i], unique_names[j]
            norm1, norm2 = normalized[name1], normalized[name2]
            score = self.calculate_confidence(name1, name2, norm1, norm2)
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score  # Symmetric

    # Use HAC clustering instead of greedy
    groups = self._cluster_with_hac(unique_names, similarity_matrix, t_high)

    # ... rest of existing code to build mappings and audit log ...
```

**Why This Works**:
- **Transitive Closure**: HAC automatically groups A→B→C even if A doesn't directly match C
- **Global Optimization**: Considers all pairwise similarities simultaneously
- **Dendrogram Structure**: Creates hierarchy of relationships, naturally handles edge cases
- **Proven Algorithm**: Used in bioinformatics, document clustering, well-tested

**Expected Impact**:
- Recall: +2-3% (90% → 92-93%) - Captures transitive relationships
- Precision: 0 to +1% (no regression, possibly slight improvement)
- F1: +2-3% (90-91% → 92-93%)
- Effort: 2-3 weeks
- Risk: Medium (algorithmic change)
- Processing Time: +10-20% slower (need to compute full similarity matrix)

**Testing**:
```python
# Test transitive closure
def test_transitive_clustering():
    names = ["Microsoft Corp", "Microsoft Corporation", "MSFT"]

    # Simulate similarity scores
    # Microsoft Corp ↔ Microsoft Corporation: 95%
    # Microsoft Corporation ↔ MSFT: 87%
    # Microsoft Corp ↔ MSFT: 75% (below threshold)

    result = matcher.process_names(names)

    # HAC should group all three together
    groups = {}
    for mapping in result['mappings']:
        canonical = mapping['canonical_name']
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(mapping['original_name'])

    # Should be ONE group containing all three names
    assert len(groups) == 1
    assert len(list(groups.values())[0]) == 3
```

---

### Phase 3 Summary

**Combined Results (Phase 1 + Phase 2 + Phase 3)**:
- **Recall**: 74% → 87-93% (+13-19%)
- **Precision**: 93% → 94-96% (+1-3%)
- **F1 Score**: 82.4% → **92-93%** (+10-11%)
- **Total Effort**: 5-7 weeks
- **Risk**: Medium
- **Cost**: $0
- **Training Required**: None ✅

---

## Implementation Checklist

### Phase 1 (Week 1-2)

- [ ] **Day 1**: Implement threshold adjustment
  - [ ] Modify `gmm_threshold_service.py` line 45
  - [ ] Test with `test_and_analyze_performance.py`
  - [ ] Verify recall improvement and acceptable precision trade-off

- [ ] **Day 2-3**: Implement multi-token requirement
  - [ ] Add `_count_matching_tokens()` method to `name_matcher.py`
  - [ ] Modify `calculate_confidence()` to apply penalty
  - [ ] Write unit tests for edge cases
  - [ ] Test with "Adobe" / "Adobe Rent-A-Car" examples

- [ ] **Day 4-5**: Implement acronym detection
  - [ ] Add `_detect_acronym_match()` method to `name_matcher.py`
  - [ ] Modify `calculate_confidence()` to apply boost
  - [ ] Write unit tests for IBM, GE, AmEx cases
  - [ ] Test false positive rate (ensure AmEx doesn't match American Express incorrectly)

- [ ] **Day 6**: Full regression testing
  - [ ] Run `test_and_analyze_performance.py`
  - [ ] Verify F1 ≥ 88%
  - [ ] Review false positive examples
  - [ ] Document results in performance log

### Phase 2 (Week 3-4) - Optional

- [ ] **Day 7-8**: Implement shared word detection
  - [ ] Add `_detect_shared_common_words()` method
  - [ ] Test with "American Airlines" / "American Express"
  - [ ] Refine common words list based on false positives

- [ ] **Day 9-10**: Implement dynamic weighting
  - [ ] Modify `calculate_confidence()` with conditional weights
  - [ ] Add embedding veto logic
  - [ ] Test shared word examples
  - [ ] Verify no regression on normal cases

- [ ] **Day 11**: Full regression testing
  - [ ] Run `test_and_analyze_performance.py`
  - [ ] Verify F1 ≥ 90%
  - [ ] Document results

### Phase 3 (Week 5-8) - Optional

- [ ] **Week 5**: Research and prototype HAC
  - [ ] Install scipy if not present: `pip install scipy`
  - [ ] Create prototype script to test HAC on small dataset
  - [ ] Validate transitive closure behavior

- [ ] **Week 6-7**: Implement HAC clustering
  - [ ] Add `_cluster_with_hac()` method
  - [ ] Modify `process_names()` to build full similarity matrix
  - [ ] Optimize memory usage (consider sparse matrices for large datasets)
  - [ ] Add progress logging for similarity matrix construction

- [ ] **Week 8**: Testing and optimization
  - [ ] Run `test_and_analyze_performance.py`
  - [ ] Profile performance, optimize bottlenecks
  - [ ] Verify F1 ≥ 92%
  - [ ] Document final results

---

## Testing Strategy

### Unit Tests

Create `backend/tests/test_improvements.py`:

```python
import pytest
from app.services.name_matcher import NameMatcher

class TestPhase1Improvements:

    def test_multi_token_requirement(self):
        """Test that short names require multiple token matches."""
        matcher = NameMatcher(embedding_mode='disabled')

        # Should reject: only 1 matching token
        score = matcher.calculate_confidence("Adobe", "Adobe Rent-A-Car", "adobe", "adobe rentacar")
        assert score < 75.0, "Should reject single-token match"

        # Should accept: 2+ matching tokens
        score = matcher.calculate_confidence("Adobe Inc", "Adobe Systems", "adobe", "adobe systems")
        assert score > 85.0, "Should accept multi-token match"

    def test_acronym_detection(self):
        """Test acronym detection boosts confidence."""
        matcher = NameMatcher(embedding_mode='disabled')

        # Should detect as acronym
        assert matcher._detect_acronym_match("IBM", "International Business Machines")
        assert matcher._detect_acronym_match("GE", "General Electric")

        # Should NOT detect as acronym
        assert not matcher._detect_acronym_match("AmEx", "American Express")  # Not initials
        assert not matcher._detect_acronym_match("Apple", "Apple Computer")  # Not abbreviation

    def test_shared_word_detection(self):
        """Test shared word detection for false positive prevention."""
        matcher = NameMatcher(embedding_mode='openai-small')

        # Should detect shared word problem
        assert matcher._detect_shared_common_words("american airlines", "american express")
        assert matcher._detect_shared_common_words("delta airlines", "delta dental")

        # Should NOT trigger on legitimate matches
        assert not matcher._detect_shared_common_words("american airlines", "american airlines inc")
```

### Integration Tests

```bash
cd backend

# Test Phase 1
python test_and_analyze_performance.py

# Expected results:
# - F1 Score: 88-90%
# - Precision: 90-92%
# - Recall: 84-88%
# - Processing Time: 60-80s (similar to current)

# Validate specific improvements
python -c "
from app.services.name_matcher import NameMatcher

matcher = NameMatcher(embedding_mode='openai-small')

# Test cases
test_pairs = [
    ('Adobe', 'Adobe Rent-A-Car', False),  # Should reject
    ('IBM', 'International Business Machines', True),  # Should match (acronym)
    ('American Airlines', 'American Express', False),  # Should reject (shared word)
]

for name1, name2, expected in test_pairs:
    result = matcher.process_names([name1, name2])
    num_groups = result['summary']['total_groups_created']
    actual_match = (num_groups == 1)
    status = '✓' if actual_match == expected else '✗'
    print(f'{status} {name1} vs {name2}: {num_groups} group(s)')
"
```

### Performance Regression Tests

Track these metrics after each phase:

| Metric | Baseline | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|----------|----------------|----------------|----------------|
| F1 Score | 82.4% | 88-90% | 90-91% | 92-93% |
| Precision | 93.0% | 90-92% | 94-96% | 94-96% |
| Recall | 74.0% | 84-88% | 85-90% | 87-93% |
| Processing Time | 67.5s | <80s | <80s | <90s |
| False Positives | 77 | 60-80 | 20-40 | 15-30 |
| False Negatives | 359 | 220-280 | 200-260 | 140-180 |

---

## Rollback Plan

If any phase causes issues:

1. **Threshold Change Rollback**:
   ```python
   # Revert backend/app/services/gmm_threshold_service.py line 45
   t_high = self._find_threshold_for_posterior(scores, weights, means, covariances, target_prob=0.98)
   ```

2. **Feature Toggle Pattern**:
   ```python
   # Add to backend/app/config/settings.py
   ENABLE_MULTI_TOKEN_REQUIREMENT: bool = True
   ENABLE_ACRONYM_DETECTION: bool = True
   ENABLE_DYNAMIC_WEIGHTING: bool = True
   ENABLE_HAC_CLUSTERING: bool = False

   # In name_matcher.py
   if settings.ENABLE_MULTI_TOKEN_REQUIREMENT:
       # Apply multi-token logic
   ```

3. **Git Branching Strategy**:
   ```bash
   # Work on feature branches
   git checkout -b feature/phase1-improvements
   # ... make changes ...
   git commit -m "Phase 1: Threshold adjustment"

   # If issues arise
   git checkout main  # Revert to stable version
   ```

---

## Success Criteria

### Phase 1 Success Criteria
- ✅ F1 Score ≥ 88%
- ✅ Precision ≥ 90%
- ✅ Recall ≥ 84%
- ✅ Processing time < 80s
- ✅ No critical bugs in production testing
- ✅ False positives ≤ 80

### Phase 2 Success Criteria
- ✅ F1 Score ≥ 90%
- ✅ Precision ≥ 94%
- ✅ Recall ≥ 85%
- ✅ Processing time < 80s
- ✅ Shared word false positives reduced by 70%

### Phase 3 Success Criteria
- ✅ F1 Score ≥ 92%
- ✅ Precision ≥ 94%
- ✅ Recall ≥ 87%
- ✅ Processing time < 90s
- ✅ Transitive closure working correctly

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Threshold too aggressive → more false positives | Medium | Medium | Start with 0.92, can tune to 0.94 if needed |
| Multi-token breaks legitimate 2-word matches | Low | Medium | Comprehensive testing, adjust penalty factor |
| Acronym detection has false positives | Low | Low | Strict matching rules, test with stock tickers |
| Dynamic weighting breaks edge cases | Medium | Medium | Feature toggle, extensive testing |
| HAC too slow for large datasets | Medium | High | Optimize with sparse matrices, parallel processing |

---

## Resource Requirements

### Development
- **Phase 1**: 1 developer, 2-3 days
- **Phase 2**: 1 developer, 2-3 days
- **Phase 3**: 1 developer, 2-3 weeks

### Infrastructure
- No additional infrastructure needed
- Uses existing OpenAI API (no cost increase)
- HAC requires scipy (add to requirements.txt)

### Monitoring
- Track false positive/negative rates weekly
- Monitor processing time for performance regression
- A/B test in production if possible

---

## Future Improvements (Beyond This Plan)

These require training and are out of scope:

1. **Fine-tuned Siamese Network** (F1: 94-96%)
   - Requires labeled training data
   - GPU infrastructure
   - 12+ weeks effort

2. **Industry Classifier with Supervised Learning** (Precision: +2-3%)
   - Requires industry-labeled dataset
   - Training pipeline
   - 8-10 weeks effort

3. **Active Learning System** (Continuous improvement)
   - Requires user feedback loop
   - Retraining infrastructure
   - 16+ weeks effort

---

## References

### Documentation
- Current algorithm: `CLAUDE.md` (lines 100-250)
- Performance analysis: `performance_detailed_reports.md`
- Ground truth data: `ground_truth.csv`

### Code Locations
- Main matcher: `backend/app/services/name_matcher.py`
- GMM service: `backend/app/services/gmm_threshold_service.py`
- Configuration: `backend/app/config/settings.py`
- Tests: `backend/tests/test_name_matcher.py`

### Analysis Scripts
- False negative analysis: Generated by Agent 1
- False positive analysis: Generated by Agent 2
- ML approaches research: `ML_APPROACHES_RESEARCH_REPORT.md`

---

## Approval & Sign-off

- [ ] **Phase 1 Approved**: _______________ (Date: _______)
- [ ] **Phase 2 Approved**: _______________ (Date: _______)
- [ ] **Phase 3 Approved**: _______________ (Date: _______)

**Notes**:
- Start with Phase 1 only
- Evaluate results before proceeding to Phase 2
- Phase 3 is optional unless 92%+ F1 is required

---

**Document Owner**: Development Team
**Last Updated**: 2025-10-24
**Next Review**: After Phase 1 completion
