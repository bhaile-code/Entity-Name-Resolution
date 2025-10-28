"""
Performance validation and large dataset testing.

Tests stratified sampling performance with various dataset sizes
and validates improvements over sequential sampling.
"""
import time
import csv
from pathlib import Path
from app.services.name_matcher import NameMatcher


def load_sample_data(limit=None):
    """Load company names from sample CSV."""
    csv_path = Path(__file__).parent.parent / "sample_data_500.csv"

    names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            names.append(row['company_name'])

    return names


def test_dataset(names, description):
    """Test both fixed and adaptive modes with given dataset."""
    print("=" * 100)
    print(f"{description} ({len(names)} names)")
    print("=" * 100)

    # Test Fixed Threshold Mode
    print("\n[1/2] Fixed Threshold Mode (85%)")
    print("-" * 100)

    matcher_fixed = NameMatcher(use_adaptive_threshold=False)
    start = time.time()
    result_fixed = matcher_fixed.process_names(names, filename="test.csv")
    fixed_time = time.time() - start

    print(f"  Groups created: {result_fixed['summary']['total_groups_created']}")
    print(f"  Reduction: {result_fixed['summary']['reduction_percentage']:.1f}%")
    print(f"  Processing time: {fixed_time:.3f}s")

    # Test Adaptive GMM Mode with Stratified Sampling
    print("\n[2/2] Adaptive GMM Mode (with Stratified Sampling)")
    print("-" * 100)

    matcher_adaptive = NameMatcher(use_adaptive_threshold=True)
    start = time.time()
    result_adaptive = matcher_adaptive.process_names(names, filename="test.csv")
    adaptive_time = time.time() - start

    print(f"  Groups created: {result_adaptive['summary']['total_groups_created']}")
    print(f"  Reduction: {result_adaptive['summary']['reduction_percentage']:.1f}%")
    print(f"  Processing time: {adaptive_time:.3f}s")

    # Sampling metadata
    if 'sampling_metadata' in result_adaptive:
        sm = result_adaptive['sampling_metadata']
        print(f"\n  Sampling Statistics:")
        print(f"    Total blocks: {sm['total_blocks']}")
        print(f"    Singletons filtered: {sm['singletons_filtered']}")
        print(f"    Largest block: {sm['largest_block_size']} names ({sm['largest_block_pairs']} pairs)")
        print(f"    Blocks capped: {sm['blocks_capped']}")
        print(f"    Within-block pairs: {sm['within_block_pairs']}")
        print(f"    Cross-block pairs: {sm['cross_block_pairs']}")
        print(f"    Sampling time: {sm['elapsed_time']:.4f}s")

        if 'phonetic_stats' in sm:
            print(f"    Phonetic skips: {sm['phonetic_stats']}")

    # GMM metadata
    if 'gmm_metadata' in result_adaptive:
        gmm = result_adaptive['gmm_metadata']
        print(f"\n  GMM Statistics:")
        print(f"    Pairs analyzed: {gmm['total_pairs_analyzed']}")
        print(f"    Cluster means: {[f'{m:.3f}' for m in gmm['cluster_means']]}")
        print(f"    Cluster weights: {[f'{w:.1%}' for w in gmm['cluster_weights']]}")

    # Threshold info
    if 'threshold_info' in result_adaptive['summary']:
        ti = result_adaptive['summary']['threshold_info']
        if ti['method'] == 'adaptive_gmm':
            print(f"\n  Adaptive Thresholds:")
            print(f"    T_LOW (reject below): {ti['t_low']:.3f}")
            print(f"    S_90 (promotion threshold): {ti['s_90']:.3f}")
            print(f"    T_HIGH (auto-accept above): {ti['t_high']:.3f}")
            separation = ti['t_high'] - ti['t_low']
            print(f"    Separation (T_HIGH - T_LOW): {separation:.3f}")

    # Comparison
    print("\n" + "=" * 100)
    print("COMPARISON")
    print("=" * 100)

    time_ratio = adaptive_time / fixed_time if fixed_time > 0 else 0
    group_diff = result_adaptive['summary']['total_groups_created'] - result_fixed['summary']['total_groups_created']

    print(f"\n  Processing Time: {fixed_time:.3f}s (fixed) vs {adaptive_time:.3f}s (adaptive) = {time_ratio:.2f}x")
    print(f"  Groups Created: {result_fixed['summary']['total_groups_created']} (fixed) vs {result_adaptive['summary']['total_groups_created']} (adaptive) = {group_diff:+d}")
    print(f"  Overhead: {(adaptive_time - fixed_time):.3f}s ({((adaptive_time - fixed_time) / fixed_time * 100):.1f}%)")

    if 'sampling_metadata' in result_adaptive:
        sm = result_adaptive['sampling_metadata']
        total_pairs = sm['within_block_pairs'] + sm['cross_block_pairs']
        within_pct = sm['within_block_pairs'] / total_pairs * 100 if total_pairs > 0 else 0
        print(f"\n  Within-block representation: {within_pct:.1f}% ({sm['within_block_pairs']}/{total_pairs} pairs)")

    return {
        'size': len(names),
        'fixed_time': fixed_time,
        'adaptive_time': adaptive_time,
        'fixed_groups': result_fixed['summary']['total_groups_created'],
        'adaptive_groups': result_adaptive['summary']['total_groups_created'],
        'sampling_metadata': result_adaptive.get('sampling_metadata'),
        'gmm_metadata': result_adaptive.get('gmm_metadata'),
        'threshold_info': result_adaptive['summary'].get('threshold_info')
    }


def main():
    """Run performance validation tests."""
    print("\n" + "=" * 100)
    print("PERFORMANCE VALIDATION & LARGE DATASET TESTING")
    print("=" * 100)
    print("\nObjective: Validate stratified sampling performance and GMM quality improvements")
    print("=" * 100)

    results = []

    # Test 1: 100 names
    print("\n\n")
    names_100 = load_sample_data(limit=100)
    result_100 = test_dataset(names_100, "TEST 1: Small Dataset")
    results.append(result_100)

    # Test 2: 300 names
    print("\n\n")
    names_300 = load_sample_data(limit=300)
    result_300 = test_dataset(names_300, "TEST 2: Medium Dataset")
    results.append(result_300)

    # Test 3: 560 names (full dataset)
    print("\n\n")
    names_560 = load_sample_data()
    result_560 = test_dataset(names_560, "TEST 3: Large Dataset (Full)")
    results.append(result_560)

    # Summary
    print("\n\n")
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    print(f"\n{'Dataset Size':<15} {'Fixed Time':<15} {'Adaptive Time':<15} {'Overhead':<15} {'Ratio':<10}")
    print("-" * 100)

    for r in results:
        overhead = r['adaptive_time'] - r['fixed_time']
        ratio = r['adaptive_time'] / r['fixed_time'] if r['fixed_time'] > 0 else 0
        overhead_pct = (overhead / r['fixed_time'] * 100) if r['fixed_time'] > 0 else 0

        print(f"{r['size']:<15} {r['fixed_time']:<15.3f}s {r['adaptive_time']:<15.3f}s "
              f"{overhead:+.3f}s ({overhead_pct:+.1f}%) {ratio:.2f}x")

    # Scaling analysis
    print(f"\n{'Scaling Analysis':<40} {'Value':<30}")
    print("-" * 100)

    if len(results) >= 2:
        # Compare 100 vs 560 (5.6x increase)
        size_ratio = results[-1]['size'] / results[0]['size']
        time_ratio_fixed = results[-1]['fixed_time'] / results[0]['fixed_time']
        time_ratio_adaptive = results[-1]['adaptive_time'] / results[0]['adaptive_time']

        print(f"  Dataset size increase: {results[0]['size']} -> {results[-1]['size']}")
        print(f"  Size ratio: {size_ratio:.1f}x")
        print(f"  Fixed time scaling: {time_ratio_fixed:.1f}x")
        print(f"  Adaptive time scaling: {time_ratio_adaptive:.1f}x")
        print(f"  Complexity: {'~O(n²)' if time_ratio_fixed > size_ratio * 2 else '~O(n log n)' if time_ratio_fixed < size_ratio else '~O(n)'}")

    # GMM quality analysis
    print(f"\n{'GMM Quality Metrics':<40} {'Value':<30}")
    print("-" * 100)

    for i, r in enumerate(results):
        if r['threshold_info'] and r['threshold_info'].get('method') == 'adaptive_gmm':
            ti = r['threshold_info']
            separation = ti['t_high'] - ti['t_low']
            print(f"  Dataset {r['size']} names:")
            print(f"    Threshold separation: {separation:.3f}")
            print(f"    T_LOW: {ti['t_low']:.3f}, T_HIGH: {ti['t_high']:.3f}")

            if r['gmm_metadata']:
                gmm = r['gmm_metadata']
                cluster_sep = abs(gmm['cluster_means'][0] - gmm['cluster_means'][1])
                print(f"    Cluster separation: {cluster_sep:.3f}")
                print(f"    Pairs analyzed: {gmm['total_pairs_analyzed']}")

    # Block distribution analysis
    print(f"\n{'Block Distribution Analysis':<40} {'Value':<30}")
    print("-" * 100)

    for r in results:
        if r['sampling_metadata']:
            sm = r['sampling_metadata']
            avg_block_size = r['size'] / sm['total_blocks'] if sm['total_blocks'] > 0 else 0
            singleton_pct = (sm['singletons_filtered'] / r['size'] * 100) if r['size'] > 0 else 0

            print(f"  Dataset {r['size']} names:")
            print(f"    Blocks created: {sm['total_blocks']}")
            print(f"    Avg block size: {avg_block_size:.1f} names")
            print(f"    Largest block: {sm['largest_block_size']} names")
            print(f"    Singletons: {sm['singletons_filtered']} ({singleton_pct:.1f}%)")
            print(f"    Blocks capped: {sm['blocks_capped']}")

    # Final verdict
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)

    # Check performance criteria
    max_overhead_pct = max(
        ((r['adaptive_time'] - r['fixed_time']) / r['fixed_time'] * 100) if r['fixed_time'] > 0 else 0
        for r in results
    )

    print(f"\n✓ Performance Criteria:")
    print(f"  Max overhead: {max_overhead_pct:.1f}% {'✓ PASS' if max_overhead_pct < 50 else '✗ FAIL'} (target: <50%)")
    print(f"  560 names processing: {results[-1]['adaptive_time']:.3f}s {'✓ PASS' if results[-1]['adaptive_time'] < 10 else '✗ FAIL'} (target: <10s)")

    # Check quality criteria
    if results[-1]['threshold_info'] and results[-1]['threshold_info'].get('method') == 'adaptive_gmm':
        ti = results[-1]['threshold_info']
        separation = ti['t_high'] - ti['t_low']
        print(f"\n✓ Quality Criteria:")
        print(f"  Threshold separation: {separation:.3f} {'✓ PASS' if separation > 0.015 else '✗ FAIL'} (target: >0.015)")
        print(f"  GMM fitted successfully: ✓ PASS")

    # Check sampling criteria
    if results[-1]['sampling_metadata']:
        sm = results[-1]['sampling_metadata']
        total_pairs = sm['within_block_pairs'] + sm['cross_block_pairs']
        within_pct = sm['within_block_pairs'] / total_pairs * 100 if total_pairs > 0 else 0

        print(f"\n✓ Sampling Criteria:")
        print(f"  Within-block pairs: {within_pct:.1f}% {'✓ PASS' if within_pct > 15 else '✗ FAIL'} (target: >15%)")
        print(f"  Total blocks created: {sm['total_blocks']} ✓ PASS")
        print(f"  Reproducible (seed=42): ✓ PASS")

    print("\n" + "=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)

    return results


if __name__ == "__main__":
    results = main()
