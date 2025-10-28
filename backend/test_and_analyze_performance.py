"""
Run comprehensive tests and analyze grouping performance against ground truth.
"""
import csv
import json
import pandas as pd
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set
from app.services.name_matcher import NameMatcher


def load_sample_data():
    """Load company names from sample CSV."""
    csv_path = Path(__file__).parent.parent / "sample_data_500.csv"
    names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row['company_name'])
    return names


def load_ground_truth():
    """Load ground truth mappings."""
    gt_path = Path(__file__).parent.parent / "ground_truth.csv"
    df = pd.read_csv(gt_path)
    return dict(zip(df['Original Name'], df['Canonical Name']))


def build_ground_truth_groups(gt_mapping: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build groups from ground truth."""
    groups = defaultdict(set)
    for original, canonical in gt_mapping.items():
        groups[canonical].add(original)
    return dict(groups)


def build_test_groups(mappings: list) -> Dict[str, Set[str]]:
    """Build groups from test results."""
    groups = defaultdict(set)
    for mapping in mappings:
        groups[mapping['canonical_name']].add(mapping['original_name'])
    return dict(groups)


def calculate_metrics(gt_groups, test_groups, gt_mapping):
    """Calculate comprehensive performance metrics."""

    # Build reverse mapping
    test_mapping = {}
    for canonical, members in test_groups.items():
        for member in members:
            test_mapping[member] = canonical

    # 1. PURITY: How pure are the test groups?
    purity_scores = []
    for test_canonical, test_members in test_groups.items():
        if len(test_members) == 1:
            purity_scores.append(1.0)
            continue

        gt_canonicals = [gt_mapping.get(m) for m in test_members if m in gt_mapping]
        if not gt_canonicals:
            continue

        most_common_gt = max(set(gt_canonicals), key=gt_canonicals.count)
        purity = gt_canonicals.count(most_common_gt) / len(gt_canonicals)
        purity_scores.append(purity)

    avg_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 0

    # 2. COMPLETENESS: How well are GT groups preserved?
    completeness_scores = []
    for gt_canonical, gt_members in gt_groups.items():
        if len(gt_members) == 1:
            completeness_scores.append(1.0)
            continue

        test_assignments = [test_mapping.get(m) for m in gt_members if m in test_mapping]
        if not test_assignments:
            continue

        most_common_test = max(set(test_assignments), key=test_assignments.count)
        completeness = test_assignments.count(most_common_test) / len(test_assignments)
        completeness_scores.append(completeness)

    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    # 3. PAIR-LEVEL METRICS
    common_names = set(gt_mapping.keys()) & set(test_mapping.keys())

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    names_list = list(common_names)
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            name1, name2 = names_list[i], names_list[j]

            gt_together = gt_mapping[name1] == gt_mapping[name2]
            test_together = test_mapping[name1] == test_mapping[name2]

            if gt_together and test_together:
                true_positives += 1
            elif not gt_together and not test_together:
                true_negatives += 1
            elif not gt_together and test_together:
                false_positives += 1
            else:
                false_negatives += 1

    total_pairs = true_positives + true_negatives + false_positives + false_negatives

    pair_accuracy = (true_positives + true_negatives) / total_pairs if total_pairs > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'purity': avg_purity,
        'completeness': avg_completeness,
        'pair_accuracy': pair_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_pairs': total_pairs,
        'num_test_groups': len(test_groups),
        'num_gt_groups': len(gt_groups),
    }


def print_report(test_name, metrics, processing_time):
    """Print detailed performance report."""
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 80}")

    print("\nGROUPING QUALITY:")
    print(f"  Purity:       {metrics['purity']:.1%}  (Homogeneity of test groups)")
    print(f"  Completeness: {metrics['completeness']:.1%}  (Preservation of GT groups)")

    print("\nPAIR-LEVEL PERFORMANCE:")
    print(f"  Accuracy:  {metrics['pair_accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}  (Grouped correctly)")
    print(f"  Recall:    {metrics['recall']:.1%}  (Found all matches)")
    print(f"  F1 Score:  {metrics['f1_score']:.1%}")

    print("\nCONFUSION MATRIX:")
    print(f"  True Positives:  {metrics['true_positives']:>6,}  (Correctly grouped together)")
    print(f"  True Negatives:  {metrics['true_negatives']:>6,}  (Correctly kept apart)")
    print(f"  False Positives: {metrics['false_positives']:>6,}  (Incorrectly grouped)")
    print(f"  False Negatives: {metrics['false_negatives']:>6,}  (Missed grouping)")
    print(f"  Total Pairs:     {metrics['total_pairs']:>6,}")

    print(f"\nProcessing Time: {processing_time:.1f}s")


def test_configuration(embedding_mode, use_adaptive, names, gt_groups, gt_mapping):
    """Test a specific configuration."""
    mode_label = f"{embedding_mode.upper()} + {'ADAPTIVE' if use_adaptive else 'FIXED'}"

    print(f"\n\nRunning: {mode_label}...")

    start_time = time.time()

    matcher = NameMatcher(
        use_adaptive_threshold=use_adaptive,
        embedding_mode=embedding_mode
    )

    result = matcher.process_names(names, filename="sample_data_500.csv")
    processing_time = time.time() - start_time

    # Build test groups and calculate metrics
    test_groups = build_test_groups(result['mappings'])
    metrics = calculate_metrics(gt_groups, test_groups, gt_mapping)

    return {
        'name': mode_label,
        'metrics': metrics,
        'processing_time': processing_time,
        'num_groups': result['summary']['total_groups_created'],
        'reduction_pct': result['summary']['reduction_percentage'],
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    names = load_sample_data()
    gt_mapping = load_ground_truth()
    gt_groups = build_ground_truth_groups(gt_mapping)

    print(f"Loaded {len(names)} company names")
    print(f"Loaded {len(gt_mapping)} ground truth mappings ({len(gt_groups)} groups)")

    # Run all test configurations
    configs = [
        ('openai-small', False),
        ('openai-small', True),
        ('openai-large', False),
        ('openai-large', True),
    ]

    results = []

    for embedding_mode, use_adaptive in configs:
        result = test_configuration(embedding_mode, use_adaptive, names, gt_groups, gt_mapping)
        results.append(result)

        # Print report
        print_report(result['name'], result['metrics'], result['processing_time'])

    # Summary comparison
    print(f"\n\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}\n")

    print(f"{'Configuration':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Purity':<8} {'Time':<8}")
    print("-" * 90)

    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x['metrics']['f1_score'], reverse=True)

    for result in results_sorted:
        m = result['metrics']
        print(f"{result['name']:<30} {m['f1_score']:.1%}   {m['precision']:.1%}     {m['recall']:.1%}   {m['purity']:.1%}   {result['processing_time']:>5.1f}s")

    print("\n" + "=" * 80)
    print(f"BEST OVERALL (by F1 Score): {results_sorted[0]['name']}")
    print(f"   F1 Score:  {results_sorted[0]['metrics']['f1_score']:.1%}")
    print(f"   Precision: {results_sorted[0]['metrics']['precision']:.1%}")
    print(f"   Recall:    {results_sorted[0]['metrics']['recall']:.1%}")
    print(f"   Time:      {results_sorted[0]['processing_time']:.1f}s")
    print("=" * 80)

    # Save results in multiple formats
    base_path = Path(__file__).parent

    # 1. CSV Summary of all metrics
    csv_file = base_path / 'performance_summary.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Configuration', 'F1_Score', 'Precision', 'Recall', 'Accuracy',
            'Purity', 'Completeness', 'True_Positives', 'True_Negatives',
            'False_Positives', 'False_Negatives', 'Total_Pairs',
            'Num_Test_Groups', 'Num_GT_Groups', 'Processing_Time_Sec'
        ])
        writer.writeheader()
        for r in results_sorted:
            m = r['metrics']
            writer.writerow({
                'Configuration': r['name'],
                'F1_Score': f"{m['f1_score']:.4f}",
                'Precision': f"{m['precision']:.4f}",
                'Recall': f"{m['recall']:.4f}",
                'Accuracy': f"{m['pair_accuracy']:.4f}",
                'Purity': f"{m['purity']:.4f}",
                'Completeness': f"{m['completeness']:.4f}",
                'True_Positives': m['true_positives'],
                'True_Negatives': m['true_negatives'],
                'False_Positives': m['false_positives'],
                'False_Negatives': m['false_negatives'],
                'Total_Pairs': m['total_pairs'],
                'Num_Test_Groups': m['num_test_groups'],
                'Num_GT_Groups': m['num_gt_groups'],
                'Processing_Time_Sec': f"{r['processing_time']:.2f}",
            })
    print(f"\nCSV summary saved to: {csv_file}")

    # 2. Markdown detailed reports
    md_file = base_path / 'performance_detailed_reports.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Performance Analysis: Detailed Reports\n\n")
        f.write(f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset**: {len(names)} company names\n\n")
        f.write(f"**Ground Truth**: {len(gt_groups)} groups\n\n")
        f.write("---\n\n")

        # Table of contents
        f.write("## Table of Contents\n\n")
        for i, r in enumerate(results_sorted, 1):
            f.write(f"{i}. [{r['name']}](#{r['name'].lower().replace(' ', '-').replace('+', '')})\n")
        f.write("\n---\n\n")

        # Best performer summary
        best = results_sorted[0]
        f.write("## Executive Summary\n\n")
        f.write(f"**WINNER: Best Overall Configuration**: {best['name']}\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| F1 Score | {best['metrics']['f1_score']:.2%} |\n")
        f.write(f"| Precision | {best['metrics']['precision']:.2%} |\n")
        f.write(f"| Recall | {best['metrics']['recall']:.2%} |\n")
        f.write(f"| Purity | {best['metrics']['purity']:.2%} |\n")
        f.write(f"| Processing Time | {best['processing_time']:.1f}s |\n")
        f.write("\n---\n\n")

        # Detailed reports for each configuration
        for r in results_sorted:
            m = r['metrics']
            f.write(f"## {r['name']}\n\n")

            f.write("### Overview\n\n")
            f.write(f"- **Groups Created**: {r['num_groups']}\n")
            f.write(f"- **Reduction**: {r['reduction_pct']:.1f}%\n")
            f.write(f"- **Processing Time**: {r['processing_time']:.1f}s\n\n")

            f.write("### Grouping Quality Metrics\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|--------|-------|-------------|\n")
            f.write(f"| **Purity** | {m['purity']:.2%} | How homogeneous are test groups? |\n")
            f.write(f"| **Completeness** | {m['completeness']:.2%} | How well are GT groups preserved? |\n\n")

            f.write("### Pair-Level Performance\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|--------|-------|-------------|\n")
            f.write(f"| **Accuracy** | {m['pair_accuracy']:.2%} | Overall correctness |\n")
            f.write(f"| **Precision** | {m['precision']:.2%} | Of pairs grouped together, how many should be? |\n")
            f.write(f"| **Recall** | {m['recall']:.2%} | Of pairs that should be together, how many are? |\n")
            f.write(f"| **F1 Score** | {m['f1_score']:.2%} | Harmonic mean of precision and recall |\n\n")

            f.write("### Confusion Matrix\n\n")
            f.write("| Category | Count | Description |\n")
            f.write("|----------|-------|-------------|\n")
            f.write(f"| True Positives | {m['true_positives']:,} | Correctly grouped together |\n")
            f.write(f"| True Negatives | {m['true_negatives']:,} | Correctly kept apart |\n")
            f.write(f"| False Positives | {m['false_positives']:,} | Incorrectly grouped together |\n")
            f.write(f"| False Negatives | {m['false_negatives']:,} | Incorrectly kept apart |\n")
            f.write(f"| **Total Pairs** | {m['total_pairs']:,} | |\n\n")

            f.write("---\n\n")

    print(f"Markdown reports saved to: {md_file}")

    # 3. JSON detailed results
    json_file = base_path / 'performance_results.json'
    with open(json_file, 'w') as f:
        json.dump({
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(names),
            'ground_truth_groups': len(gt_groups),
            'best_configuration': results_sorted[0]['name'],
            'results': results_sorted,
        }, f, indent=2)
    print(f"JSON results saved to: {json_file}")

    print(f"\n{'=' * 80}")
    print("All output files saved successfully!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
