"""
Quick baseline verification test - Tests only OPENAI-LARGE + ADAPTIVE configuration.
"""
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.name_matcher import NameMatcher
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data():
    """Load test data and ground truth."""
    # Load sample data (files are in parent directory)
    df = pd.read_csv('../sample_data_500.csv')
    names = df.iloc[:, 0].dropna().astype(str).tolist()

    # Load ground truth - format is "Original Name" -> "Canonical Name"
    gt_df = pd.read_csv('../ground_truth.csv')
    gt_mapping = dict(zip(gt_df['Original Name'], gt_df['Canonical Name']))

    # Build groups from ground truth
    from collections import defaultdict
    groups = defaultdict(set)
    for original, canonical in gt_mapping.items():
        groups[canonical].add(original)
        groups[canonical].add(canonical)

    # Convert to name -> group_id mapping
    ground_truth_groups = {}
    for group_id, (canonical, names_in_group) in enumerate(groups.items()):
        for name in names_in_group:
            ground_truth_groups[name] = group_id

    return names, ground_truth_groups

def calculate_metrics(predicted_groups, ground_truth_groups):
    """Calculate clustering metrics."""
    # Build pairs
    gt_pairs = set()
    for name1 in ground_truth_groups:
        for name2 in ground_truth_groups:
            if name1 < name2 and ground_truth_groups[name1] == ground_truth_groups[name2]:
                gt_pairs.add((name1, name2))

    pred_pairs = set()
    for group in predicted_groups.values():
        names_in_group = group
        for i, name1 in enumerate(names_in_group):
            for name2 in names_in_group[i+1:]:
                if name1 < name2:
                    pred_pairs.add((name1, name2))
                else:
                    pred_pairs.add((name2, name1))

    # Calculate metrics
    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def main():
    print("="*80)
    print("QUICK BASELINE VERIFICATION TEST")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    names, ground_truth_groups = load_data()
    print(f"Loaded {len(names)} company names")
    print(f"Ground truth has {len(set(ground_truth_groups.values()))} groups")
    print()

    # Test OPENAI-LARGE + ADAPTIVE
    print("Testing: OPENAI-LARGE + ADAPTIVE")
    print("-" * 80)

    matcher = NameMatcher(
        use_adaptive_threshold=True,
        embedding_mode='openai-large'
    )

    start_time = time.time()
    result = matcher.process_names(names, filename="sample_data_500.csv")
    elapsed = time.time() - start_time

    # Extract groups from result
    predicted_groups = {}
    for mapping in result['mappings']:
        canonical = mapping['canonical_name']
        if canonical not in predicted_groups:
            predicted_groups[canonical] = []
        predicted_groups[canonical].append(mapping['original_name'])

    # Calculate metrics
    metrics = calculate_metrics(predicted_groups, ground_truth_groups)

    # Print results
    print()
    print("RESULTS:")
    print(f"  F1 Score:       {metrics['f1']*100:.2f}%")
    print(f"  Precision:      {metrics['precision']*100:.2f}%")
    print(f"  Recall:         {metrics['recall']*100:.2f}%")
    print(f"  Processing Time: {elapsed:.1f}s")
    print()
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print()
    print(f"  Groups Created:  {len(predicted_groups)}")
    print()

    # Get threshold info
    if 'threshold_info' in result['summary']:
        thresh = result['summary']['threshold_info']
        print("ADAPTIVE THRESHOLDS:")
        print(f"  Method:  {thresh.get('method', 'N/A')}")
        print(f"  T_LOW:   {thresh.get('t_low', 0)*100:.1f}%")
        print(f"  S_90:    {thresh.get('s_90', 0)*100:.1f}%")
        print(f"  T_HIGH:  {thresh.get('t_high', 0)*100:.1f}%")
        print()

    # Compare with baseline
    print("="*80)
    print("COMPARISON WITH ORIGINAL BASELINE:")
    print("="*80)
    baseline = {
        'f1': 0.8242,
        'precision': 0.9299,
        'recall': 0.7400,
        'fp': 77,
        'fn': 359,
        'time': 67.54
    }

    print(f"  F1 Score:       {metrics['f1']*100:.2f}% (baseline: {baseline['f1']*100:.2f}%, diff: {(metrics['f1']-baseline['f1'])*100:+.2f}%)")
    print(f"  Precision:      {metrics['precision']*100:.2f}% (baseline: {baseline['precision']*100:.2f}%, diff: {(metrics['precision']-baseline['precision'])*100:+.2f}%)")
    print(f"  Recall:         {metrics['recall']*100:.2f}% (baseline: {baseline['recall']*100:.2f}%, diff: {(metrics['recall']-baseline['recall'])*100:+.2f}%)")
    print(f"  Processing Time: {elapsed:.1f}s (baseline: {baseline['time']:.1f}s, diff: {elapsed-baseline['time']:+.1f}s)")
    print()
    print(f"  False Positives: {metrics['fp']} (baseline: {baseline['fp']}, diff: {metrics['fp']-baseline['fp']:+d})")
    print(f"  False Negatives: {metrics['fn']} (baseline: {baseline['fn']}, diff: {metrics['fn']-baseline['fn']:+d})")
    print()

    # Status
    if abs(metrics['f1'] - baseline['f1']) < 0.01:
        print("✅ STATUS: Baseline performance CONFIRMED (within 1% of original)")
    elif metrics['f1'] >= baseline['f1']:
        print("✅ STATUS: Performance IMPROVED or MAINTAINED")
    else:
        print("⚠️  STATUS: Performance regression detected")

    print("="*80)

if __name__ == '__main__':
    main()
