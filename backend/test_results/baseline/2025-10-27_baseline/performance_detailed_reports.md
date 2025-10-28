# Performance Analysis: Detailed Reports

**Analysis Date**: 2025-10-24 14:09:01

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
| F1 Score | 82.42% |
| Precision | 92.99% |
| Recall | 74.00% |
| Purity | 99.46% |
| Processing Time | 67.5s |

---

## OPENAI-LARGE + ADAPTIVE

### Overview

- **Groups Created**: 289
- **Reduction**: 60.9%
- **Processing Time**: 67.5s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.46% | How homogeneous are test groups? |
| **Completeness** | 93.49% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.83% | Overall correctness |
| **Precision** | 92.99% | Of pairs grouped together, how many should be? |
| **Recall** | 74.00% | Of pairs that should be together, how many are? |
| **F1 Score** | 82.42% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 1,022 | Correctly grouped together |
| True Negatives | 262,443 | Correctly kept apart |
| False Positives | 77 | Incorrectly grouped together |
| False Negatives | 359 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-SMALL + ADAPTIVE

### Overview

- **Groups Created**: 288
- **Reduction**: 61.0%
- **Processing Time**: 71.3s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.17% | How homogeneous are test groups? |
| **Completeness** | 93.21% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.82% | Overall correctness |
| **Precision** | 91.35% | Of pairs grouped together, how many should be? |
| **Recall** | 73.43% | Of pairs that should be together, how many are? |
| **F1 Score** | 81.41% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 1,014 | Correctly grouped together |
| True Negatives | 262,424 | Correctly kept apart |
| False Positives | 96 | Incorrectly grouped together |
| False Negatives | 367 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-LARGE + FIXED

### Overview

- **Groups Created**: 334
- **Reduction**: 54.8%
- **Processing Time**: 157.3s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.20% | How homogeneous are test groups? |
| **Completeness** | 88.72% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.79% | Overall correctness |
| **Precision** | 96.44% | Of pairs grouped together, how many should be? |
| **Recall** | 62.78% | Of pairs that should be together, how many are? |
| **F1 Score** | 76.05% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 867 | Correctly grouped together |
| True Negatives | 262,488 | Correctly kept apart |
| False Positives | 32 | Incorrectly grouped together |
| False Negatives | 514 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

## OPENAI-SMALL + FIXED

### Overview

- **Groups Created**: 348
- **Reduction**: 52.9%
- **Processing Time**: 196.9s

### Grouping Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Purity** | 99.22% | How homogeneous are test groups? |
| **Completeness** | 87.67% | How well are GT groups preserved? |

### Pair-Level Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.78% | Overall correctness |
| **Precision** | 97.27% | Of pairs grouped together, how many should be? |
| **Recall** | 59.23% | Of pairs that should be together, how many are? |
| **F1 Score** | 73.63% | Harmonic mean of precision and recall |

### Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| True Positives | 818 | Correctly grouped together |
| True Negatives | 262,497 | Correctly kept apart |
| False Positives | 23 | Incorrectly grouped together |
| False Negatives | 563 | Incorrectly kept apart |
| **Total Pairs** | 263,901 | |

---

