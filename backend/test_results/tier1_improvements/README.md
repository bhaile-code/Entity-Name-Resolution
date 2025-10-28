# Tier 1 Improvements - Test Results

This folder contains performance test results **after** implementing Tier 1 improvements (no ML training required).

## Improvements Implemented

### 1. GMM Threshold Adjustment
- **Change**: Reduced posterior probability requirement from 0.98 → 0.92
- **File**: `backend/app/services/gmm_threshold_service.py` (line 171)
- **Expected Impact**: +8-10% recall

### 2. Acronym Detection
- **Change**: Added initials-matching logic to detect acronyms
- **File**: `backend/app/services/name_matcher.py`
- **Method**: New `_detect_acronym_match()` method
- **Expected Impact**: +5-8% recall, fixes 144 abbreviation pairs

### 3. Multi-Token Requirement
- **Change**: Apply 25% penalty for short names with <2 matching tokens
- **File**: `backend/app/services/name_matcher.py`
- **Method**: New `_count_matching_tokens()` helper
- **Expected Impact**: +1-2% precision, reduces subset false positives

## Expected Results

| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| F1 Score | 82.4% | 87-90% | [TBD] |
| Precision | 93.0% | 90-92% | [TBD] |
| Recall | 74.0% | 84-88% | [TBD] |
| Processing Time | 67.5s | <70s | [TBD] |

## Test Files

Results will be stored in timestamped folders (e.g., `2025-10-27_tier1/`)

- `performance_results.json` - Raw metrics for all configurations
- `performance_summary.csv` - Summary table
- `performance_detailed_reports.md` - Detailed report
- `comparison_summary.md` - Before/after comparison with analysis

## Implementation Date
October 27, 2025
