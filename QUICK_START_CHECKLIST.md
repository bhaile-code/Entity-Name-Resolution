# Quick Start Checklist: Improving Entity Name Resolution

**Goal:** Increase F1 score from 82.4% to 91-93% in 16 weeks

---

## Phase 1: HAC with Constraints (Weeks 1-3) → 87-89% F1

### Week 1: Core Implementation

- [ ] **Day 1-2: Setup**
  ```bash
  cd backend
  pip install scipy>=1.11.0 numpy>=1.24.0
  mkdir -p app/services/clustering
  touch app/services/clustering/__init__.py
  ```

- [ ] **Day 3-5: Implement HAC Matcher**
  - [ ] Create `app/services/clustering/hac_matcher.py`
  - [ ] Copy code from `IMPLEMENTATION_GUIDE.md` Section "Approach 1, Step 1"
  - [ ] Implement `ConstrainedHACMatcher` class:
    - [ ] `_build_distance_matrix()` method
    - [ ] `_enforce_constraints()` method
    - [ ] `cluster()` method
    - [ ] `auto_tune_threshold()` method

- [ ] **Day 6-7: Testing**
  - [ ] Create `tests/test_hac_matcher.py`
  - [ ] Test basic clustering (no constraints)
  - [ ] Test must-link constraints
  - [ ] Test cannot-link constraints
  - [ ] Test auto-tuning
  - [ ] Run: `pytest tests/test_hac_matcher.py -v`

### Week 2: Constraint Learning & Integration

- [ ] **Day 8-10: Constraint Learner**
  - [ ] Create `app/services/clustering/constraint_learner.py`
  - [ ] Implement `ConstraintLearner` class:
    - [ ] `_load_ticker_database()` method (hardcode 20-30 common tickers)
    - [ ] `generate_must_link()` method
    - [ ] `generate_cannot_link()` method
  - [ ] Test with sample names

- [ ] **Day 11-13: Integration**
  - [ ] Update `app/config/settings.py`:
    ```python
    HAC_LINKAGE_METHOD = "average"
    HAC_DISTANCE_THRESHOLD = 0.15
    HAC_AUTO_TUNE = True
    HAC_USE_CONSTRAINTS = True
    ```
  - [ ] Update `app/services/name_matcher.py`:
    - [ ] Add `use_hac` parameter to `__init__()`
    - [ ] Import HAC matcher and constraint learner
    - [ ] Modify `process_names()` to use HAC when `use_hac=True`
  - [ ] Update `app/api/routes.py`:
    - [ ] Add `use_hac` query parameter to `/api/process`

- [ ] **Day 14: End-to-End Test**
  - [ ] Test via API:
    ```bash
    curl -X POST "http://localhost:8000/api/process?use_hac=true" \
      -F "file=@ground_truth.csv"
    ```
  - [ ] Verify F1 ≥ 87%

### Week 3: Validation & Documentation

- [ ] **Day 15-17: Performance Testing**
  - [ ] Create `backend/test_clustering_approaches.py` (see `IMPLEMENTATION_GUIDE.md`)
  - [ ] Run comparison: Baseline vs HAC
  - [ ] Verify metrics:
    - [ ] F1 score ≥ 87%
    - [ ] Recall ≥ 81%
    - [ ] Precision ≥ 92%
    - [ ] Processing time <5s for 739 names

- [ ] **Day 18-19: Optimization**
  - [ ] Profile performance with `cProfile`
  - [ ] Optimize slow parts (distance matrix caching?)
  - [ ] Test with larger datasets (1000, 2000 names)

- [ ] **Day 20-21: Documentation**
  - [ ] Update `CLAUDE.md` with HAC usage
  - [ ] Write docstrings for all new functions
  - [ ] Create usage examples

**Phase 1 Checkpoint:** ✅ F1 score 87-89% achieved

---

## Phase 2: Graph Transitive Closure (Weeks 4-6) → 88-90% F1

### Week 4: Graph Building

- [ ] **Day 22-23: Setup**
  ```bash
  pip install networkx>=3.1 matplotlib>=3.7.0
  ```

- [ ] **Day 24-27: Implement Graph Matcher**
  - [ ] Create `app/services/clustering/graph_matcher.py`
  - [ ] Implement `GraphBasedMatcher` class:
    - [ ] `build_graph()` method
    - [ ] `add_transitive_edges()` method
    - [ ] `cluster_graph()` method
    - [ ] `process_names()` method

- [ ] **Day 28: Testing**
  - [ ] Create `tests/test_graph_matcher.py`
  - [ ] Test graph building
  - [ ] Test transitive closure (verify A→B, B→C adds A→C)
  - [ ] Test community detection

### Week 5: Community Detection & Visualization

- [ ] **Day 29-31: Clustering Algorithms**
  - [ ] Implement Louvain community detection
  - [ ] Implement Label Propagation
  - [ ] Compare performance on ground truth

- [ ] **Day 32-34: Visualization**
  - [ ] Implement `visualize_graph()` method
  - [ ] Generate graph viz for sample dataset
  - [ ] Identify visual patterns (isolated nodes, dense clusters, bridges)

- [ ] **Day 35: Integration with HAC**
  - [ ] Update `NameMatcher` to support `use_graph=True`
  - [ ] Test hybrid: HAC base clustering → Graph refinement

### Week 6: Testing & Validation

- [ ] **Day 36-38: Performance Testing**
  - [ ] Run comparison: HAC vs HAC+Graph
  - [ ] Verify F1 ≥ 88% (incremental +1-2% over HAC)
  - [ ] Check recall improvement (capturing transitive matches?)

- [ ] **Day 39-41: Optimization & Documentation**
  - [ ] Optimize transitive closure (limit iterations to 3)
  - [ ] Cache graph structures
  - [ ] Update documentation

- [ ] **Day 42: Final Validation**
  - [ ] K-fold cross-validation (5 folds)
  - [ ] A/B test: Compare with production baseline
  - [ ] Deploy to staging environment

**Phase 2 Checkpoint:** ✅ F1 score 88-90% achieved

---

## Phase 3: Siamese Neural Network (Weeks 7-14) → 91-93% F1

### Week 7-8: Data Preparation

- [ ] **Day 43-45: Setup ML Environment**
  ```bash
  pip install torch transformers sentence-transformers datasets
  mkdir -p backend/ml_models
  mkdir -p backend/training_data
  ```

- [ ] **Day 46-49: Data Augmentation**
  - [ ] Create `backend/ml_models/data_augmentation.py`
  - [ ] Implement typo generation (keyboard distance model)
  - [ ] Implement suffix substitution (Inc. ↔ Corp ↔ Ltd)
  - [ ] Implement acronym generation
  - [ ] Target: 15,000 training pairs (10x augmentation)

- [ ] **Day 50-52: External Data Integration**
  - [ ] Set up Wikidata API client (company aliases)
  - [ ] Integrate ticker symbol database (Alpha Vantage or Yahoo Finance)
  - [ ] Augment with external aliases

- [ ] **Day 53-56: Prepare Training Set**
  - [ ] Split ground truth into train/val/test (70/15/15)
  - [ ] Generate positive pairs from ground truth
  - [ ] Generate hard negative pairs (shared words, low semantic similarity)
  - [ ] Save to `training_data/train.csv`, `val.csv`, `test.csv`

### Week 9-10: Model Development

- [ ] **Day 57-60: Implement Siamese Architecture**
  - [ ] Create `backend/ml_models/siamese_matcher.py`
  - [ ] Implement `SiameseCompanyMatcher` class:
    - [ ] Encoder (Sentence-BERT base)
    - [ ] Feature extractor (linear layers)
    - [ ] Similarity head (learned combination)
  - [ ] Implement `ContrastiveLoss`

- [ ] **Day 61-64: Training Pipeline**
  - [ ] Create `backend/ml_models/train.py`
  - [ ] Implement data loading (PyTorch DataLoader)
  - [ ] Implement training loop (20 epochs)
  - [ ] Implement validation loop
  - [ ] Save best model checkpoint

### Week 11-12: Training & Hyperparameter Tuning

- [ ] **Day 65-70: Initial Training**
  - [ ] Train baseline model (20 epochs)
  - [ ] Monitor training metrics (loss, accuracy)
  - [ ] Evaluate on validation set
  - [ ] Target: Val F1 ≥ 90%

- [ ] **Day 71-76: Hyperparameter Tuning**
  - [ ] Tune learning rate (1e-5, 2e-5, 5e-5)
  - [ ] Tune batch size (16, 32, 64)
  - [ ] Tune margin in contrastive loss (0.2, 0.3, 0.5)
  - [ ] Tune embedding dimension (128, 256, 512)
  - [ ] Select best configuration

### Week 13-14: Inference & Integration

- [ ] **Day 77-79: Inference Optimization**
  - [ ] Create `backend/ml_models/inference.py`
  - [ ] Implement batch inference (process 100+ pairs/second)
  - [ ] Add caching layer (avoid recomputing embeddings)
  - [ ] Test inference speed

- [ ] **Day 80-83: Integration with NameMatcher**
  - [ ] Update `NameMatcher` to support `embedding_mode='siamese'`
  - [ ] Load trained model in `calculate_confidence()`
  - [ ] Test hybrid scoring:
    - [ ] 30% WRatio (fuzzy)
    - [ ] 10% Token Set
    - [ ] 25% OpenAI embeddings (general)
    - [ ] 35% Siamese embeddings (company-specific)

- [ ] **Day 84-87: End-to-End Testing**
  - [ ] Run full pipeline: HAC + Graph + Siamese
  - [ ] Verify F1 ≥ 91%
  - [ ] Stress test with large datasets (5000+ names)

- [ ] **Day 88-91: Production Prep**
  - [ ] Deploy model to production server
  - [ ] Set up monitoring (inference latency, error rate)
  - [ ] Create rollback plan (fallback to OpenAI embeddings)

- [ ] **Day 92-98: Documentation & Handoff**
  - [ ] Write model card (architecture, performance, limitations)
  - [ ] Create retraining guide (how to update model with new data)
  - [ ] Update `CLAUDE.md` with Siamese usage

**Phase 3 Checkpoint:** ✅ F1 score 91-93% achieved

---

## Phase 4: Ensemble & Polish (Weeks 15-16) → 92-94% F1

### Week 15: Ensemble Integration

- [ ] **Day 99-102: Combine All Approaches**
  - [ ] Create `EnsembleMatcher` class
  - [ ] Pipeline: HAC base → Graph refinement → Siamese re-scoring
  - [ ] Test combined system

- [ ] **Day 103-105: Optimization**
  - [ ] Profile end-to-end performance
  - [ ] Optimize bottlenecks (distance matrix, embeddings, graph building)
  - [ ] Target: <10 seconds for 739 names

### Week 16: Production Deployment

- [ ] **Day 106-108: Shadow Mode Testing**
  - [ ] Deploy to production (shadow mode)
  - [ ] Run new system alongside current
  - [ ] Compare metrics for 1 week

- [ ] **Day 109-111: Production Cutover**
  - [ ] Enable new system for 10% of traffic
  - [ ] Monitor metrics (F1, latency, errors)
  - [ ] Gradually increase to 100%

- [ ] **Day 112: Launch**
  - [ ] Full production deployment
  - [ ] Monitor for 24 hours
  - [ ] Celebrate! 🎉

**Final Checkpoint:** ✅ F1 score 92-94% in production

---

## Success Metrics Dashboard

Track these metrics daily during implementation:

| Metric | Baseline | Phase 1 Target | Phase 2 Target | Phase 3 Target | Final Target |
|--------|----------|----------------|----------------|----------------|--------------|
| **F1 Score** | 82.4% | 87-89% | 88-90% | 91-93% | 92-94% |
| **Precision** | 93.0% | ≥92% | ≥92% | ≥94% | ≥94% |
| **Recall** | 74.0% | 81-84% | 84-87% | 88-92% | 89-93% |
| **Processing Time** | 68s | <5s | <8s | <10s | <10s |
| **Groups Created** | 289 | 250-260 | 240-250 | 229-235 | 229-235 |

---

## Risk Management

### Red Flags 🚩
- F1 drops below baseline (82.4%)
- Processing time >15 seconds
- Precision drops below 90%
- System crashes or errors increase

### Mitigation Plan
1. **Revert to previous phase** (rollback feature flag)
2. **Debug in isolation** (test HAC/Graph/Siamese separately)
3. **Check data quality** (validate ground truth, check for corrupted inputs)
4. **Profile performance** (identify bottlenecks)
5. **Consult documentation** (see `ML_APPROACHES_RESEARCH_REPORT.md`)

---

## Daily Standup Template

Use this for daily progress tracking:

```
## Date: YYYY-MM-DD

### Completed Today
- [ ] Task 1
- [ ] Task 2

### In Progress
- [ ] Task 3 (70% complete)

### Blocked
- [ ] Task 4 (waiting on API access)

### Metrics
- F1 Score: XX.X%
- Processing Time: X.Xs
- Tests Passing: XX/XX

### Next Steps
- [ ] Task 5 (tomorrow)
- [ ] Task 6 (tomorrow)
```

---

## Resources

### Documentation
1. **ML_APPROACHES_RESEARCH_REPORT.md** - Comprehensive 30-page analysis
2. **IMPLEMENTATION_GUIDE.md** - Code examples and technical specs
3. **EXECUTIVE_SUMMARY.md** - 3-minute decision brief
4. **CLAUDE.md** - Project architecture and conventions

### External APIs & Data
- **Alpha Vantage** (ticker symbols): https://www.alphavantage.co/
- **Wikidata Query Service** (aliases): https://query.wikidata.org/
- **OpenCorporates API** (company data): https://api.opencorporates.com/

### Libraries
- **scipy** (HAC): https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
- **networkx** (graphs): https://networkx.org/documentation/stable/
- **PyTorch** (Siamese): https://pytorch.org/docs/stable/index.html
- **Sentence-Transformers**: https://www.sbert.net/

---

## Questions & Support

- **Technical implementation:** See `IMPLEMENTATION_GUIDE.md`
- **Algorithm theory:** See `ML_APPROACHES_RESEARCH_REPORT.md`
- **Business case:** See `EXECUTIVE_SUMMARY.md`
- **Project setup:** See `CLAUDE.md`

Good luck with implementation! 🚀
