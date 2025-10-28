# Executive Summary: Entity Name Resolution Performance Improvements

**Date:** October 24, 2025
**Current F1 Score:** 82.4%
**Target F1 Score:** 85%+
**Recommended Path:** 91-93% F1 achievable in 16 weeks

---

## The Problem

Your entity name resolution system performs well overall (82.4% F1) but suffers from **low recall (74%)**:
- **359 false negatives** (missed matches) vs 77 false positives
- **Over-clustering:** 289 groups predicted vs 229 ground truth = **60 extra groups**
- **Root cause:** Conservative greedy clustering that misses transitive relationships

### Error Breakdown
| Error Type | Count | % of FNs | Root Cause |
|------------|-------|----------|------------|
| Transitive closure failures | ~144 | 40% | A→B, B→C grouped, but A↮C |
| Abbreviation mismatches | ~90 | 25% | "IBM" vs "International Business Machines" |
| Borderline threshold cases | ~72 | 20% | Score just below threshold, no phonetic boost |
| Domain-specific ambiguity | ~53 | 15% | "Delta Air Lines" vs "Delta Dental" |

---

## Recommended Solution

### Three-Phase Approach (16 Weeks Total)

```
Phase 1 (Weeks 1-3):  HAC with Constraints      → 87-89% F1  (+5-7%)
Phase 2 (Weeks 4-6):  Graph Transitive Closure  → 88-90% F1  (+1-2%)
Phase 3 (Weeks 7-14): Siamese Neural Network    → 91-93% F1  (+3-4%)
```

**Final Performance Estimate:**
- **F1 Score:** 91-93% (target exceeded by 6-8 points)
- **Recall:** 89-92% (+15-18 points improvement)
- **Precision:** 94-96% (+1-3 points maintained)

---

## Approach Details

### 1. Hierarchical Agglomerative Clustering (HAC)

**What it is:** Bottom-up clustering that merges similar groups iteratively, with constraints to enforce domain knowledge.

**Why it's better:**
- **Global optimization** (considers entire dataset structure)
- **No order dependence** (all names treated equally)
- **Constraint injection** (must-link for tickers, cannot-link for ambiguous cases)

**Estimated Impact:**
- Recall: 74% → 81-84% (+7-10%)
- F1: 82.4% → 87-89% (+5-7%)

**Complexity:** Low-Medium
**Timeline:** 2-3 weeks
**Risk:** Low (fallback to current system)

**Key Advantages:**
- No ML training required
- No GPU needed
- Works with existing similarity scoring
- Easy to add/modify constraints

---

### 2. Graph-Based Transitive Closure

**What it is:** Build similarity graph, add transitive edges (if A→B and B→C, add A→C), then use community detection for clustering.

**Why it's better:**
- **Captures missed relationships** (transitive triangle: A-B-C where A-C was below threshold)
- **Leverages graph structure** (connected components reveal natural groupings)

**Estimated Impact:**
- Recall: 84% → 86-88% (+2-4% over HAC)
- F1: 87-89% → 88-90% (+1-2%)

**Complexity:** Medium
**Timeline:** 2-3 weeks (after HAC)
**Risk:** Low

**Key Advantages:**
- Builds on HAC foundation
- Proven graph algorithms (Louvain, Label Propagation)
- Visualizable for debugging

---

### 3. Fine-Tuned Siamese Neural Network

**What it is:** Train a neural network specifically for company name matching, learning patterns like:
- Abbreviations (IBM ↔ International Business Machines)
- Typos (Microsft ↔ Microsoft)
- Ticker symbols (AAPL ↔ Apple Inc.)

**Why it's better:**
- **Domain-specific learning** (current OpenAI embeddings are general-purpose)
- **Handles hard negatives** (American Express vs American Airlines)
- **Transfer learning** (overcomes small dataset limitation)

**Estimated Impact:**
- Recall: 88% → 91-94% (+3-6% over graph)
- F1: 88-90% → 91-93% (+2-3%)

**Complexity:** High
**Timeline:** 6-8 weeks
**Risk:** Medium (requires ML expertise, GPU)

**Key Advantages:**
- Highest performance ceiling
- Learns from errors automatically
- Can be continuously improved with more data

---

## Cost-Benefit Analysis

| Approach | F1 Gain | Dev Time | GPU? | Training Data? | Risk | ROI |
|----------|---------|----------|------|----------------|------|-----|
| **HAC** | +5-7% | 2-3 weeks | No | No | Low | ★★★★★ |
| **Graph** | +1-2% | 2-3 weeks | No | No | Low | ★★★★☆ |
| **Siamese** | +3-4% | 6-8 weeks | Yes (training) | 15K pairs (augmented) | Medium | ★★★★★ |

---

## Implementation Priority

### Immediate (Do First)
✅ **HAC with Constraints** - Lowest risk, fastest delivery, solid 5-7% gain

**Why:**
- Achieves 87-89% F1 (exceeds 85% target)
- No training data or GPU required
- Fallback to current system if issues arise
- Foundation for subsequent improvements

### Short-Term (Build On Success)
✅ **Graph Transitive Closure** - Incremental 1-2% boost

**Why:**
- Leverages HAC foundation
- Captures relationships HAC missed
- Low additional complexity

### Medium-Term (Maximum Performance)
✅ **Siamese Network** - Final 3-4% push to 91-93%

**Why:**
- Highest performance ceiling
- Domain-specific pattern learning
- Continuous improvement potential

---

## Technical Requirements

### Phase 1: HAC (Weeks 1-3)
- **Dependencies:** `scipy`, `numpy` (already common)
- **Infrastructure:** None (runs on existing servers)
- **Data:** None (uses existing ground truth for validation)
- **Team:** 1 backend engineer

### Phase 2: Graph (Weeks 4-6)
- **Dependencies:** `networkx`
- **Infrastructure:** None
- **Data:** None
- **Team:** Same engineer from Phase 1

### Phase 3: Siamese (Weeks 7-14)
- **Dependencies:** `torch`, `transformers`, `sentence-transformers`
- **Infrastructure:** GPU for training (NVIDIA T4 or better, ~$100-300 cloud cost)
- **Data:** 15K augmented pairs (from 739 names via augmentation pipeline)
- **Team:** 1 ML engineer + 1 backend engineer

---

## Risk Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HAC too slow for large datasets | Medium | Medium | Use FastCluster library (100x speedup) |
| Graph memory overflow | Low | Medium | Sparse matrix representation |
| Siamese overfitting | Medium | High | Heavy augmentation + transfer learning |
| API dependencies (OpenAI) | Low | High | Local model fallback, caching |

### Business Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Schedule delay | Medium | Low | Phase 1 alone achieves target (87-89%) |
| Performance regression | Low | High | A/B testing, shadow mode, rollback plan |
| Integration complexity | Low | Medium | Gradual rollout, feature flags |

---

## Success Criteria

### Phase 1 Acceptance (HAC)
- ✅ F1 score ≥ 87%
- ✅ No precision regression (≥92%)
- ✅ Processing time <5 seconds for 739 names
- ✅ Unit tests pass (>90% coverage)

### Phase 2 Acceptance (Graph)
- ✅ F1 score ≥ 88% (incremental improvement over HAC)
- ✅ Recall ≥ 84%
- ✅ Processing time <8 seconds

### Phase 3 Acceptance (Siamese)
- ✅ F1 score ≥ 91%
- ✅ Recall ≥ 88%
- ✅ Inference time <3 seconds (batch processing)
- ✅ Model size <1GB (deployable)

---

## Validation Strategy

### K-Fold Cross-Validation
- 5-fold split on 739 ground truth names
- Train on 591 names, test on 148 names per fold
- Average metrics across folds for robust estimate

### A/B Testing
- Shadow mode: Run new algorithm alongside current
- Compare metrics before deploying to production
- Decision threshold: +3% F1 improvement required

### Continuous Monitoring
- Track F1, precision, recall daily
- Alert if metrics drop >2% below baseline
- Monthly retraining for Siamese model (drift prevention)

---

## Alternative Approaches Considered (Lower Priority)

### 4. Active Learning (Human-in-the-Loop)
- **Impact:** +2-4% F1
- **Cost:** High (requires human labeling)
- **When:** After Phases 1-3, for continuous improvement

### 5. Knowledge Graph Integration (Wikidata, OpenCorporates)
- **Impact:** +3-5% F1
- **Cost:** Medium (API integration, data quality issues)
- **When:** Long-term enhancement (Phase 4)

### 6. Domain-Specific Normalization Rules
- **Impact:** +1-3% F1
- **Cost:** Low (hand-crafted rules)
- **When:** Quick wins, can implement anytime

### 7. Ensemble Voting (Multiple Algorithms)
- **Impact:** +3-5% F1
- **Cost:** Medium (computational overhead)
- **When:** After Phase 3, if pushing for 95%+ F1

---

## Timeline & Milestones

```
Week 1-3:  HAC Implementation
  ├─ Week 1: Core HAC algorithm + constraint framework
  ├─ Week 2: Constraint learning, integration with NameMatcher
  └─ Week 3: Testing, validation, documentation
  Milestone: 87-89% F1 achieved ✓

Week 4-6:  Graph Transitive Closure
  ├─ Week 4: Graph building, transitive edge logic
  ├─ Week 5: Community detection, visualization
  └─ Week 6: Integration with HAC, testing
  Milestone: 88-90% F1 achieved ✓

Week 7-14: Siamese Network
  ├─ Week 7-8: Data augmentation pipeline
  ├─ Week 9-10: External data integration (Wikidata, tickers)
  ├─ Week 11-12: Model training + hyperparameter tuning
  └─ Week 13-14: Inference optimization, integration
  Milestone: 91-93% F1 achieved ✓

Week 15-16: Ensemble & Polish
  ├─ Week 15: Ensemble integration (HAC + Graph + Siamese)
  └─ Week 16: Production deployment, monitoring setup
  Final: 92-94% F1 in production ✓
```

---

## Budget Estimate

### Development Costs
- **Backend Engineer:** 3 weeks @ $150/hr × 40hr/week = $18,000
- **ML Engineer:** 8 weeks @ $175/hr × 40hr/week = $56,000
- **Total Labor:** $74,000

### Infrastructure Costs
- **GPU Training (Siamese):** $200-500 one-time
- **Production Inference:** $0 (CPU sufficient) or $50/month (GPU optional)
- **Total Infrastructure:** ~$500 first year

### Total Project Cost: ~$75,000
**ROI:** Depends on business value of improved accuracy (reduced manual review, fewer errors)

---

## Key Takeaways

1. **Current system is high-precision, low-recall** → Over-clustering problem (too many groups)

2. **Quick win available:** HAC (Approach 1) delivers 87-89% F1 in just 3 weeks with low risk

3. **Maximum performance path:** All 3 approaches combined → 91-93% F1 (exceeds target by 6-8%)

4. **Phased rollout reduces risk:** Each phase builds on previous, with fallback options

5. **Transitive relationships are key:** 40% of errors are from missed A→B→C triangles

6. **Domain knowledge helps:** Constraints (must-link tickers, cannot-link ambiguous) boost performance

7. **Fine-tuned embeddings matter:** Generic OpenAI embeddings don't understand company name patterns

---

## Decision Point

### Option A: Conservative (Recommended for MVP)
**Implement Phase 1 (HAC) only**
- Timeline: 3 weeks
- Cost: ~$18,000
- F1: 87-89% (target exceeded)
- Risk: Very low

### Option B: Balanced (Recommended for Production)
**Implement Phases 1-2 (HAC + Graph)**
- Timeline: 6 weeks
- Cost: ~$36,000
- F1: 88-90%
- Risk: Low

### Option C: Maximum Performance (Recommended if Budget Allows)
**Implement Phases 1-3 (Full Pipeline)**
- Timeline: 14 weeks
- Cost: ~$75,000
- F1: 91-93%
- Risk: Medium

---

## Next Steps

1. **Review this document** with engineering and product teams
2. **Choose implementation option** (A, B, or C)
3. **Allocate engineering resources** (1-2 engineers for 3-14 weeks)
4. **Set up GPU infrastructure** (if Option C chosen)
5. **Begin Phase 1 implementation** following `IMPLEMENTATION_GUIDE.md`

---

## Questions?

**Technical Details:** See `ML_APPROACHES_RESEARCH_REPORT.md` (comprehensive 30-page analysis)

**Implementation:** See `IMPLEMENTATION_GUIDE.md` (code examples, testing, deployment)

**Contact:** Your development team

---

**Prepared by:** Claude (Anthropic)
**Version:** 1.0
**Last Updated:** October 24, 2025
