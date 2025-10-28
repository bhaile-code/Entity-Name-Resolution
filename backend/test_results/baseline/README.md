# Baseline Test Results

This folder contains performance test results from the system **before** implementing Tier 1 improvements.

## Configuration Tested

**Best Performing Configuration:**
- Embedding Mode: OpenAI text-embedding-3-large
- Threshold Mode: Adaptive GMM
- GMM T_HIGH: 0.98 (requires 98% posterior probability)

## Baseline Performance (OPENAI-LARGE + ADAPTIVE)

| Metric | Value |
|--------|-------|
| **F1 Score** | 82.4% |
| **Precision** | 93.0% |
| **Recall** | 74.0% |
| **Processing Time** | 67.5 seconds |
| **False Positives** | 77 pairs |
| **False Negatives** | 359 pairs |

## Key Issues Identified

1. **Conservative GMM Threshold**: 37.6% of false negatives (150 pairs) scored 80-100% but were rejected
2. **Abbreviation Gaps**: 36.1% of false negatives (144 pairs) are abbreviations (IBM ↔ International Business Machines)
3. **Subset Matching**: 12 false positives from subset matches (Adobe ↔ Adobe Rent-A-Car)

## Test Files

- `performance_results.json` - Raw metrics for all configurations
- `performance_summary.csv` - Summary table (Excel-friendly)
- `performance_detailed_reports.md` - Human-readable detailed report
- `performance_analysis_output.txt` - Full test output with examples
- `false_positive_analysis.json` - Detailed FP analysis
- `embedding_test_results.txt` - Embedding comparison results

## Date
October 27, 2025 (tests run October 24, 2025)
