"""
Analyze grouping performance against ground truth.

This script evaluates how well each test configuration grouped companies together
by comparing with the ground truth mappings.
"""

import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def load_ground_truth(filepath: str) -> Dict[str, str]:
    """Load ground truth mappings from CSV."""
    df = pd.read_csv(filepath)
    # Create mapping of original name to canonical name
    return dict(zip(df['Original Name'], df['Canonical Name']))


def load_test_results(filepath: str) -> Dict[str, Dict]:
    """Parse test results file and extract groupings for each test."""
    results = {}
    current_test = None
    current_groups = {}
    in_group_section = False
    current_canonical = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # Detect test configuration headers
        if 'TESTING:' in line:
            # Save previous test if exists
            if current_test and current_groups:
                results[current_test] = current_groups

            # Parse test name
            parts = line.split('TESTING:')[1].strip()
            current_test = parts
            current_groups = {}
            in_group_section = False
            current_canonical = None

        # Detect start of sample groups section
        elif 'Sample Groups with Confidence Variation' in line:
            in_group_section = True

        # Detect group headers like "Group 1 (8 members, variation: 13.7%):"
        elif in_group_section and line.strip().startswith('Group ') and '(' in line:
            # Reset for new group
            current_canonical = None

        # Detect canonical name like "  Canonical: Microsoft"
        elif in_group_section and '  Canonical:' in line:
            canonical = line.split('Canonical:')[1].strip()
            current_canonical = canonical
            if canonical not in current_groups:
                current_groups[canonical] = set()

        # Detect member lines like "    - Microsoft Corporation                         100.0%"
        elif in_group_section and line.strip().startswith('- ') and current_canonical:
            # Extract member name (before the percentage)
            parts = line.split('- ')[1]
            # The name is everything before the percentage (which has multiple spaces before it)
            # Split by multiple spaces
            name_parts = parts.split('  ')
            if name_parts:
                member_name = name_parts[0].strip()
                current_groups[current_canonical].add(member_name)

        # Stop at next major section
        elif in_group_section and ('=' * 40 in line or line.strip().startswith('TESTING:')):
            in_group_section = False

    # Save last test
    if current_test and current_groups:
        results[current_test] = current_groups

    return results


def build_ground_truth_groups(gt_mapping: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build groups from ground truth (canonical -> set of originals)."""
    groups = defaultdict(set)
    for original, canonical in gt_mapping.items():
        groups[canonical].add(original)
    return dict(groups)


def build_test_groups(mappings: List[Dict]) -> Dict[str, Set[str]]:
    """Build groups from test results (canonical -> set of originals)."""
    groups = defaultdict(set)
    for mapping in mappings:
        canonical = mapping['canonical_name']
        original = mapping['original_name']
        groups[canonical].add(original)
    return dict(groups)


def calculate_group_metrics(
    gt_groups: Dict[str, Set[str]],
    test_groups: Dict[str, Set[str]],
    gt_mapping: Dict[str, str]
) -> Dict:
    """
    Calculate grouping metrics.

    Metrics:
    - Purity: For each test group, what % of members belong to the same GT group?
    - Completeness: For each GT group, what % of members ended up in the same test group?
    - Pair-level accuracy: Of all possible pairs, how many are correctly grouped/separated?
    """

    # Build reverse mapping: original -> test canonical
    test_mapping = {}
    for canonical, members in test_groups.items():
        for member in members:
            test_mapping[member] = canonical

    # 1. PURITY: How pure are the test groups?
    purity_scores = []
    for test_canonical, test_members in test_groups.items():
        if len(test_members) == 1:
            purity_scores.append(1.0)  # Single-member groups are pure
            continue

        # Find most common GT canonical in this test group
        gt_canonicals = [gt_mapping.get(member) for member in test_members if member in gt_mapping]
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

        # Find where these members ended up in test groups
        test_assignments = [test_mapping.get(member) for member in gt_members if member in test_mapping]
        if not test_assignments:
            continue

        # Most common test group
        most_common_test = max(set(test_assignments), key=test_assignments.count)
        completeness = test_assignments.count(most_common_test) / len(test_assignments)
        completeness_scores.append(completeness)

    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    # 3. PAIR-LEVEL ACCURACY
    # Generate all pairs from names that appear in both GT and test
    common_names = set(gt_mapping.keys()) & set(test_mapping.keys())

    true_positives = 0  # Should be together, are together
    true_negatives = 0  # Should be apart, are apart
    false_positives = 0  # Should be apart, are together
    false_negatives = 0  # Should be together, are apart

    names_list = list(common_names)
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            name1, name2 = names_list[i], names_list[j]

            # Ground truth: should they be together?
            gt_together = gt_mapping[name1] == gt_mapping[name2]

            # Test result: are they together?
            test_together = test_mapping[name1] == test_mapping[name2]

            if gt_together and test_together:
                true_positives += 1
            elif not gt_together and not test_together:
                true_negatives += 1
            elif not gt_together and test_together:
                false_positives += 1
            else:  # gt_together and not test_together
                false_negatives += 1

    total_pairs = true_positives + true_negatives + false_positives + false_negatives

    pair_accuracy = (true_positives + true_negatives) / total_pairs if total_pairs > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 4. GROUP-LEVEL STATISTICS
    # How many groups were created vs GT?
    num_test_groups = len(test_groups)
    num_gt_groups = len(gt_groups)

    # Average group sizes
    avg_test_group_size = sum(len(members) for members in test_groups.values()) / num_test_groups if num_test_groups > 0 else 0
    avg_gt_group_size = sum(len(members) for members in gt_groups.values()) / num_gt_groups if num_gt_groups > 0 else 0

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
        'num_test_groups': num_test_groups,
        'num_gt_groups': num_gt_groups,
        'avg_test_group_size': avg_test_group_size,
        'avg_gt_group_size': avg_gt_group_size,
    }


def print_report(test_name: str, metrics: Dict):
    """Print a formatted report for a test configuration."""
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 80}")

    print("\n📊 GROUPING QUALITY METRICS:")
    print(f"  Purity:       {metrics['purity']:.1%}  (How homogeneous are test groups?)")
    print(f"  Completeness: {metrics['completeness']:.1%}  (How well are GT groups preserved?)")

    print("\n🎯 PAIR-LEVEL METRICS:")
    print(f"  Accuracy:  {metrics['pair_accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}  (Of pairs grouped together, how many should be?)")
    print(f"  Recall:    {metrics['recall']:.1%}  (Of pairs that should be together, how many are?)")
    print(f"  F1 Score:  {metrics['f1_score']:.1%}")

    print("\n📈 CONFUSION MATRIX (Pairs):")
    print(f"  True Positives:  {metrics['true_positives']:,}  (Correctly grouped together)")
    print(f"  True Negatives:  {metrics['true_negatives']:,}  (Correctly kept apart)")
    print(f"  False Positives: {metrics['false_positives']:,}  (Incorrectly grouped together)")
    print(f"  False Negatives: {metrics['false_negatives']:,}  (Incorrectly kept apart)")
    print(f"  Total Pairs:     {metrics['total_pairs']:,}")

    print("\n📦 GROUP STATISTICS:")
    print(f"  Test Groups Created: {metrics['num_test_groups']}")
    print(f"  Ground Truth Groups: {metrics['num_gt_groups']}")
    print(f"  Avg Test Group Size: {metrics['avg_test_group_size']:.2f}")
    print(f"  Avg GT Group Size:   {metrics['avg_gt_group_size']:.2f}")


def main():
    # Load ground truth
    print("Loading ground truth mappings...")
    gt_mapping = load_ground_truth('../ground_truth.csv')
    gt_groups = build_ground_truth_groups(gt_mapping)

    print(f"Ground truth: {len(gt_mapping)} names in {len(gt_groups)} groups")

    # Load test results
    print("\nLoading test results...")
    test_results = load_test_results('embedding_test_results.txt')

    print(f"Found {len(test_results)} test configurations")

    # Analyze each test
    all_metrics = {}
    for test_name, mappings in test_results.items():
        test_groups = build_test_groups(mappings)
        metrics = calculate_group_metrics(gt_groups, test_groups, gt_mapping)
        all_metrics[test_name] = metrics
        print_report(test_name, metrics)

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n{'Configuration':<40} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Purity':<8}")
    print("-" * 80)

    # Sort by F1 score
    sorted_tests = sorted(all_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)

    for test_name, metrics in sorted_tests:
        print(f"{test_name:<40} {metrics['f1_score']:.1%}   {metrics['precision']:.1%}     {metrics['recall']:.1%}   {metrics['purity']:.1%}")

    print("\n🏆 WINNER: " + sorted_tests[0][0])
    print(f"   Best F1 Score: {sorted_tests[0][1]['f1_score']:.1%}")


if __name__ == "__main__":
    main()
