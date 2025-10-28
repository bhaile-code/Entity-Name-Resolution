# Performance Analysis: Detailed Reports

**Analysis Date**: 2025-10-27 14:45:19

**Dataset**: 739 company names

**Ground Truth**: 229 groups

---

## Table of Contents

1. [OPENAI-LARGE + ADAPTIVE](#openai-large--adaptive)
2. [OPENAI-SMALL + ADAPTIVE](#openai-small--adaptive)
3. [OPENAI-LARGE + FIXED](#openai-large--fixed)
4. [OPENAI-SMALL + FIXED](#openai-small--fixed)

---

## Executive Summary

**WINNER: Best Overall Configuration**: OPENAI-LARGE + ADAPTIVE

| Metric | Value |
|--------|-------|
| F1 Score | 80.89% |
| Precision | 89.40% |
| Recall | 73.86% |
| Purity | 99.11% |
| Processing Time | 85.5s |

---

## OPENAI-LARGE + ADAPTIVE

### Overview

- **Groups Created**: 285
- **Reduction**: 61.4%
- **Processing Time**: 85.5s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.11% | How homogeneous are test groups? |
| **Completeness** | 93.38% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.82% | Overall correctness |
| **Precision** | 89.40% | Of pairs grouped together, how many should be? |
| **Recall** | 73.86% | Of pairs that should be together, how many are? |
| **F1 Score** | 80.89% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 1,020 | Correctly grouped together |
| True Negatives | 262,399 | Correctly kept apart |
| False Positives | 121 | Incorrectly grouped together |
| False Negatives | 361 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-SMALL + ADAPTIVE

### Overview

- **Groups Created**: 284
- **Reduction**: 61.6%
- **Processing Time**: 94.0s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.13% | How homogeneous are test groups? |
| **Completeness** | 93.38% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.82% | Overall correctness |
| **Precision** | 89.01% | Of pairs grouped together, how many should be? |
| **Recall** | 73.86% | Of pairs that should be together, how many are? |
| **F1 Score** | 80.73% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 1,020 | Correctly grouped together |
| True Negatives | 262,394 | Correctly kept apart |
| False Positives | 126 | Incorrectly grouped together |
| False Negatives | 361 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-LARGE + FIXED

### Overview

- **Groups Created**: 473
- **Reduction**: 36.0%
- **Processing Time**: 210.5s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 100.00% | How homogeneous are test groups? |
| **Completeness** | 77.95% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.64% | Overall correctness |
| **Precision** | 100.00% | Of pairs grouped together, how many should be? |
| **Recall** | 31.79% | Of pairs that should be together, how many are? |
| **F1 Score** | 48.24% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 439 | Correctly grouped together |
| True Negatives | 262,520 | Correctly kept apart |
| False Positives | 0 | Incorrectly grouped together |
| False Negatives | 942 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-SMALL + FIXED

### Overview

- **Groups Created**: 472
- **Reduction**: 36.1%
- **Processing Time**: 244.6s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.93% | How homogeneous are test groups? |
| **Completeness** | 77.88% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.64% | Overall correctness |
| **Precision** | 99.55% | Of pairs grouped together, how many should be? |
| **Recall** | 31.72% | Of pairs that should be together, how many are? |
| **F1 Score** | 48.11% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 438 | Correctly grouped together |
| True Negatives | 262,518 | Correctly kept apart |
| False Positives | 2 | Incorrectly grouped together |
| False Negatives | 943 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

