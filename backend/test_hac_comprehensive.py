"""
Comprehensive test for HAC mode with OPENAI-LARGE embeddings.
Tests against ground truth and saves results to test_results folder.
"""
import time
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import first so settings loads .env
from app.services.name_matcher import NameMatcher
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data():
    """Load test data and ground truth."""
    # Load sample data
    df = pd.read_csv('../sample_data_500.csv')
    names = df.iloc[:, 0].dropna().astype(str).tolist()

    # Load ground truth
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

    print(f"Loaded {len(names)} company names")
    print(f"Ground truth has {len(groups)} groups")

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
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def test_hac_configuration(names, ground_truth_groups, threshold, embedding_mode='openai-large'):
    """Test HAC with specific configuration."""

    print(f"\n{'='*80}")
    print(f"Testing: {embedding_mode.upper()} + HAC (threshold={threshold})")
    print(f"{'='*80}")

    start_time = time.time()

    # Initialize matcher
    matcher = NameMatcher(
        clustering_mode='hac',
        hac_threshold=threshold,
        hac_linkage='average',
        embedding_mode=embedding_mode
    )

    # Process names
    result = matcher.process_names(names, filename='sample_data_500.csv')

    elapsed = time.time() - start_time

    # Extract predicted groups
    predicted_groups = {}
    for mapping in result['mappings']:
        group_id = mapping['group_id']
        if group_id not in predicted_groups:
            predicted_groups[group_id] = []
        predicted_groups[group_id].append(mapping['original_name'])

    # Calculate metrics
    metrics = calculate_metrics(predicted_groups, ground_truth_groups)

    # Print results
    print(f"\nResults:")
    print(f"  F1 Score:       {metrics['f1']:.2%}")
    print(f"  Precision:      {metrics['precision']:.2%}")
    print(f"  Recall:         {metrics['recall']:.2%}")
    print(f"  Processing Time: {elapsed:.1f}s")
    print(f"\n  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"\n  Groups Created:  {result['summary']['total_groups_created']}")

    # HAC metadata
    if 'hac_metadata' in result:
        meta = result['hac_metadata']
        print(f"\nHAC Metadata:")
        print(f"  Cophenetic Correlation: {meta['cophenetic_distance']:.4f}")
        print(f"  Avg Cluster Size: {meta['avg_cluster_size']:.2f}")
        print(f"  Singleton Clusters: {meta['singleton_clusters']}")
        print(f"  Avg Distance (all pairs): {meta['avg_distance_all_pairs']:.4f}")
        print(f"  Avg Distance (within clusters): {meta['avg_distance_within_clusters']:.4f}")

    # Threshold info
    threshold_info = result['summary']['threshold_info']
    print(f"\nThreshold Info:")
    print(f"  Method: {threshold_info['method']}")
    print(f"  HAC Threshold: {threshold_info.get('hac_threshold', 'N/A')}")
    print(f"  HAC Linkage: {threshold_info.get('hac_linkage', 'N/A')}")

    return {
        'configuration': {
            'embedding_mode': embedding_mode,
            'clustering_mode': 'hac',
            'hac_threshold': threshold,
            'hac_linkage': 'average'
        },
        'metrics': metrics,
        'summary': result['summary'],
        'hac_metadata': result.get('hac_metadata', {}),
        'threshold_info': threshold_info,
        'processing_time': elapsed
    }


def save_results(results, output_dir):
    """Save test results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_dir / f'hac_test_results_{timestamp}.json'

    # Convert to serializable format
    serializable_results = {}
    for config_name, result in results.items():
        serializable_results[config_name] = {
            'configuration': result['configuration'],
            'metrics': {
                'f1_score': result['metrics']['f1'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'true_positives': result['metrics']['true_positives'],
                'false_positives': result['metrics']['false_positives'],
                'false_negatives': result['metrics']['false_negatives']
            },
            'processing_time_seconds': result['processing_time'],
            'groups_created': result['summary']['total_groups_created'],
            'reduction_percentage': result['summary']['reduction_percentage'],
            'hac_metadata': result['hac_metadata'],
            'threshold_info': result['threshold_info']
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    return output_file


def create_summary_markdown(results, output_dir):
    """Create a markdown summary of the test results."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = output_dir / f'HAC_TEST_SUMMARY_{timestamp}.md'

    with open(output_file, 'w') as f:
        f.write("# HAC Clustering Test Results\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Configuration**: OPENAI-LARGE + HAC\n")
        f.write(f"**Dataset**: sample_data_500.csv (739 company names)\n\n")
        f.write("---\n\n")

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Configuration | F1 Score | Precision | Recall | Processing Time | Groups |\n")
        f.write("|---------------|----------|-----------|--------|-----------------|--------|\n")

        for config_name, result in results.items():
            m = result['metrics']
            f.write(f"| {config_name} | {m['f1']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | "
                   f"{result['processing_time']:.1f}s | {result['summary']['total_groups_created']} |\n")

        f.write("\n---\n\n")

        # Detailed results for each configuration
        for config_name, result in results.items():
            f.write(f"## {config_name}\n\n")

            # Configuration
            f.write("### Configuration\n")
            cfg = result['configuration']
            f.write(f"- **Embedding Mode**: {cfg['embedding_mode']}\n")
            f.write(f"- **Clustering Mode**: {cfg['clustering_mode']}\n")
            f.write(f"- **HAC Threshold**: {cfg['hac_threshold']} ")
            f.write(f"(Similarity: {(1 - cfg['hac_threshold']) * 100:.0f}%)\n")
            f.write(f"- **HAC Linkage**: {cfg['hac_linkage']}\n\n")

            # Metrics
            f.write("### Performance Metrics\n")
            m = result['metrics']
            f.write(f"- **F1 Score**: {m['f1']:.2%}\n")
            f.write(f"- **Precision**: {m['precision']:.2%}\n")
            f.write(f"- **Recall**: {m['recall']:.2%}\n")
            f.write(f"- **Processing Time**: {result['processing_time']:.1f} seconds\n\n")

            # Error Analysis
            f.write("### Error Analysis\n")
            f.write(f"- **True Positives**: {m['true_positives']:,} pairs correctly grouped\n")
            f.write(f"- **False Positives**: {m['false_positives']:,} pairs incorrectly grouped\n")
            f.write(f"- **False Negatives**: {m['false_negatives']:,} pairs missed\n\n")

            # Grouping Statistics
            f.write("### Grouping Statistics\n")
            f.write(f"- **Total Input Names**: {result['summary']['total_input_names']}\n")
            f.write(f"- **Groups Created**: {result['summary']['total_groups_created']}\n")
            f.write(f"- **Reduction**: {result['summary']['reduction_percentage']:.1f}%\n")
            f.write(f"- **Average Group Size**: {result['summary']['average_group_size']:.2f}\n\n")

            # HAC Metadata
            if result['hac_metadata']:
                meta = result['hac_metadata']
                f.write("### HAC Quality Metrics\n")
                f.write(f"- **Cophenetic Correlation**: {meta['cophenetic_distance']:.4f} ")
                f.write("(measures how well dendrogram preserves distances)\n")
                f.write(f"- **Singleton Clusters**: {meta['singleton_clusters']} ")
                f.write("(names that didn't match anything)\n")
                f.write(f"- **Max Cluster Size**: {meta['max_cluster_size']} names\n")
                f.write(f"- **Avg Distance (all pairs)**: {meta['avg_distance_all_pairs']:.4f}\n")
                f.write(f"- **Avg Distance (within clusters)**: {meta['avg_distance_within_clusters']:.4f}\n\n")

            f.write("---\n\n")

        # Comparison with baseline
        f.write("## Comparison with Baseline (from previous tests)\n\n")
        f.write("| Method | F1 Score | Precision | Recall | Time | Notes |\n")
        f.write("|--------|----------|-----------|--------|------|-------|\n")
        f.write("| **Fixed Threshold** | 82.4% | 93.0% | 74.0% | <1s | Original baseline |\n")
        f.write("| **Adaptive GMM** | 80.1% | 90.3% | 71.9% | 78s | Unstable, varies |\n")

        for config_name, result in results.items():
            m = result['metrics']
            f.write(f"| **HAC ({result['configuration']['hac_threshold']})** | "
                   f"{m['f1']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | "
                   f"{result['processing_time']:.0f}s | Deterministic |\n")

        f.write("\n---\n\n")

        # Key Findings
        f.write("## Key Findings\n\n")

        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
        best_name = best_config[0]
        best_result = best_config[1]

        f.write(f"### Best Configuration: {best_name}\n\n")
        f.write(f"- **F1 Score**: {best_result['metrics']['f1']:.2%}\n")
        f.write(f"- **Threshold**: {best_result['configuration']['hac_threshold']} ")
        f.write(f"({(1 - best_result['configuration']['hac_threshold']) * 100:.0f}% similarity)\n")
        f.write(f"- **Processing Time**: {best_result['processing_time']:.1f}s\n\n")

        f.write("### Advantages of HAC\n\n")
        f.write("1. ✅ **Deterministic**: Same input always produces same output\n")
        f.write("2. ✅ **Fast**: Processing time comparable to fixed threshold\n")
        f.write("3. ✅ **Configurable**: Users can adjust threshold via UI\n")
        f.write("4. ✅ **Quality Metrics**: Cophenetic correlation provides clustering quality assessment\n")
        f.write("5. ✅ **Transparent**: Clear threshold with intuitive similarity percentage\n\n")

        f.write("### Recommendations\n\n")
        f.write("- **Production Use**: HAC with threshold 0.42 (58% similarity) provides balanced results\n")
        f.write("- **Conservative Mode**: Use threshold 0.15-0.30 (70-85% similarity) for fewer false positives\n")
        f.write("- **Aggressive Mode**: Use threshold 0.50-0.60 (40-50% similarity) for maximum grouping\n")
        f.write("- **Embedding Mode**: openai-large provides best accuracy, openai-small is cost-effective\n\n")

    print(f"Summary saved to: {output_file}")
    return output_file


def main():
    """Run comprehensive HAC tests."""
    print("="*80)
    print("COMPREHENSIVE HAC CLUSTERING TEST")
    print("="*80)
    print()

    # Load data
    names, ground_truth_groups = load_data()

    # Test configurations - just test default for now
    test_configs = [
        {'threshold': 0.42, 'name': 'HAC-0.42 (Default - 58% similarity)'},
    ]

    results = {}

    # Run tests
    for config in test_configs:
        try:
            result = test_hac_configuration(
                names,
                ground_truth_groups,
                threshold=config['threshold'],
                embedding_mode='openai-large'
            )
            results[config['name']] = result
        except Exception as e:
            logger.error(f"Test failed for {config['name']}: {e}", exc_info=True)
            print(f"\n❌ ERROR: {e}\n")

    if not results:
        print("\n❌ All tests failed!")
        return

    # Save results
    output_dir = Path(__file__).parent / 'test_results' / 'hac_comprehensive'
    json_file = save_results(results, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    for config_name, result in results.items():
        m = result['metrics']
        print(f"\n{config_name}:")
        print(f"  F1 Score: {m['f1']:.2%}")
        print(f"  Precision: {m['precision']:.2%}")
        print(f"  Recall: {m['recall']:.2%}")
        print(f"  Processing Time: {result['processing_time']:.1f}s")
        print(f"  Groups Created: {result['summary']['total_groups_created']}")
        print(f"  Cophenetic Correlation: {result['hac_metadata']['cophenetic_distance']:.4f}")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {json_file}")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {e}")
        exit(1)
