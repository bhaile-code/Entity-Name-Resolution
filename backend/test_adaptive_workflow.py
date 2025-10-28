"""
Test script to verify adaptive GMM threshold workflow.
Compares fixed vs adaptive threshold modes on sample data.
"""
from app.services.name_matcher import NameMatcher
from app.config import settings


def test_fixed_threshold_mode():
    """Test traditional fixed threshold mode."""
    print("=" * 70)
    print("TESTING FIXED THRESHOLD MODE")
    print("=" * 70)

    matcher = NameMatcher(use_adaptive_threshold=False)

    test_names = [
        "Apple Inc.",
        "Apple Computer",
        "Apple",
        "Microsoft Corporation",
        "Microsoft Corp",
        "Microsoft",
        "Google LLC",
        "Google Inc.",
        "Amazon.com Inc.",
        "Amazon",
        "Meta Platforms Inc.",
        "Facebook Inc.",
        "Tesla Inc.",
        "Tesla Motors"
    ]

    result = matcher.process_names(test_names, filename="test_fixed.csv")

    print(f"\nInput names: {result['summary']['total_input_names']}")
    print(f"Groups created: {result['summary']['total_groups_created']}")
    print(f"Reduction: {result['summary']['reduction_percentage']:.1f}%")
    print(f"Processing time: {result['summary']['processing_time_seconds']:.3f}s")

    print(f"\nThreshold Info:")
    print(f"  Method: {result['summary']['threshold_info']['method']}")
    print(f"  Fixed Threshold: {result['summary']['threshold_info']['fixed_threshold']}%")

    print(f"\nSample Mappings:")
    for mapping in result['mappings'][:5]:
        print(f"  {mapping['original_name']:30} -> {mapping['canonical_name']:20} ({mapping['confidence_score']:.2%})")

    return result


def test_adaptive_threshold_mode():
    """Test adaptive GMM-based threshold mode."""
    print("\n" + "=" * 70)
    print("TESTING ADAPTIVE GMM THRESHOLD MODE")
    print("=" * 70)

    matcher = NameMatcher(use_adaptive_threshold=True)

    test_names = [
        "Apple Inc.",
        "Apple Computer",
        "Apple",
        "Microsoft Corporation",
        "Microsoft Corp",
        "Microsoft",
        "Google LLC",
        "Google Inc.",
        "Amazon.com Inc.",
        "Amazon",
        "Meta Platforms Inc.",
        "Facebook Inc.",
        "Tesla Inc.",
        "Tesla Motors"
    ]

    result = matcher.process_names(test_names, filename="test_adaptive.csv")

    print(f"\nInput names: {result['summary']['total_input_names']}")
    print(f"Groups created: {result['summary']['total_groups_created']}")
    print(f"Reduction: {result['summary']['reduction_percentage']:.1f}%")
    print(f"Processing time: {result['summary']['processing_time_seconds']:.3f}s")

    threshold_info = result['summary']['threshold_info']
    print(f"\nThreshold Info:")
    print(f"  Method: {threshold_info['method']}")

    if threshold_info['method'] == 'adaptive_gmm':
        print(f"  T_LOW: {threshold_info['t_low']:.3f}")
        print(f"  S_90: {threshold_info['s_90']:.3f}")
        print(f"  T_HIGH: {threshold_info['t_high']:.3f}")

        if 'gmm_metadata' in result:
            gmm = result['gmm_metadata']
            print(f"\nGMM Cluster Statistics:")
            print(f"  Cluster Means: {', '.join([f'{m:.3f}' for m in gmm['cluster_means']])}")
            print(f"  Cluster Weights: {', '.join([f'{w:.3f}' for w in gmm['cluster_weights']])}")
            print(f"  Pairs Analyzed: {gmm['total_pairs_analyzed']}")
    elif threshold_info['fallback_reason']:
        print(f"  Fallback Reason: {threshold_info['fallback_reason']}")

    print(f"\nSample Mappings:")
    for mapping in result['mappings'][:5]:
        print(f"  {mapping['original_name']:30} -> {mapping['canonical_name']:20} ({mapping['confidence_score']:.2%})")

    return result


def compare_modes():
    """Compare results between fixed and adaptive modes."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    fixed_result = test_fixed_threshold_mode()
    adaptive_result = test_adaptive_threshold_mode()

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Fixed':<15} {'Adaptive':<15}")
    print("-" * 70)
    print(f"{'Groups Created':<30} {fixed_result['summary']['total_groups_created']:<15} {adaptive_result['summary']['total_groups_created']:<15}")
    print(f"{'Reduction %':<30} {fixed_result['summary']['reduction_percentage']:<15.1f} {adaptive_result['summary']['reduction_percentage']:<15.1f}")
    print(f"{'Processing Time (s)':<30} {fixed_result['summary']['processing_time_seconds']:<15.3f} {adaptive_result['summary']['processing_time_seconds']:<15.3f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compare_modes()
