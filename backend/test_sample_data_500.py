"""
Test script for 500-name sample dataset.
Compares fixed vs adaptive threshold modes with comprehensive dataset.
"""
import csv
from pathlib import Path
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


def test_fixed_mode(names):
    """Test with fixed threshold mode."""
    print("=" * 80)
    print("TESTING FIXED THRESHOLD MODE (85%)")
    print("=" * 80)

    matcher = NameMatcher(use_adaptive_threshold=False)
    result = matcher.process_names(names, filename="sample_data_500.csv")

    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Total input names':<40} {result['summary']['total_input_names']:<20}")
    print(f"{'Groups created':<40} {result['summary']['total_groups_created']:<20}")
    print(f"{'Reduction percentage':<40} {result['summary']['reduction_percentage']:<20.1f}%")
    print(f"{'Average group size':<40} {result['summary']['average_group_size']:<20.2f}")
    print(f"{'Processing time':<40} {result['summary']['processing_time_seconds']:<20.3f}s")

    threshold_info = result['summary']['threshold_info']
    print(f"\n{'Threshold Information':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Method':<40} {threshold_info['method']:<20}")
    print(f"{'Fixed threshold':<40} {threshold_info['fixed_threshold']:<20}%")

    # Analyze confidence score distribution
    confidences = [m['confidence_score'] for m in result['mappings']]
    high_conf = sum(1 for c in confidences if c >= 0.90)
    mid_conf = sum(1 for c in confidences if 0.70 <= c < 0.90)
    low_conf = sum(1 for c in confidences if c < 0.70)

    print(f"\n{'Confidence Score Distribution':<40} {'Count':<20}")
    print("-" * 80)
    print(f"{'High (>=90%)':<40} {high_conf:<20}")
    print(f"{'Medium (70-90%)':<40} {mid_conf:<20}")
    print(f"{'Low (<70%)':<40} {low_conf:<20}")

    # Show sample groups
    print(f"\nSample Groups (first 5):")
    print("-" * 80)
    groups_shown = set()
    for mapping in result['mappings']:
        if len(groups_shown) >= 5:
            break
        if mapping['group_id'] not in groups_shown:
            groups_shown.add(mapping['group_id'])
            group_members = [m for m in result['mappings'] if m['group_id'] == mapping['group_id']]
            print(f"\nGroup {mapping['group_id']} ({len(group_members)} members):")
            print(f"  Canonical: {mapping['canonical_name']}")
            for member in group_members[:3]:  # Show first 3 members
                if member['original_name'] != mapping['canonical_name']:
                    print(f"    - {member['original_name']} ({member['confidence_score']:.1%})")

    return result


def test_adaptive_mode(names):
    """Test with adaptive GMM threshold mode."""
    print("\n" + "=" * 80)
    print("TESTING ADAPTIVE GMM THRESHOLD MODE")
    print("=" * 80)

    matcher = NameMatcher(use_adaptive_threshold=True)
    result = matcher.process_names(names, filename="sample_data_500.csv")

    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Total input names':<40} {result['summary']['total_input_names']:<20}")
    print(f"{'Groups created':<40} {result['summary']['total_groups_created']:<20}")
    print(f"{'Reduction percentage':<40} {result['summary']['reduction_percentage']:<20.1f}%")
    print(f"{'Average group size':<40} {result['summary']['average_group_size']:<20.2f}")
    print(f"{'Processing time':<40} {result['summary']['processing_time_seconds']:<20.3f}s")

    threshold_info = result['summary']['threshold_info']
    print(f"\n{'Threshold Information':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Method':<40} {threshold_info['method']:<20}")

    if threshold_info['method'] == 'adaptive_gmm':
        print(f"{'T_LOW (reject below)':<40} {threshold_info['t_low']:<20.3f}")
        print(f"{'S_90 (promotion threshold)':<40} {threshold_info['s_90']:<20.3f}")
        print(f"{'T_HIGH (auto-accept above)':<40} {threshold_info['t_high']:<20.3f}")

        if 'gmm_metadata' in result:
            gmm = result['gmm_metadata']
            print(f"\n{'GMM Cluster Statistics':<40} {'Value':<20}")
            print("-" * 80)
            print(f"{'Total pairs analyzed':<40} {gmm['total_pairs_analyzed']:<20}")
            print(f"{'Low cluster mean':<40} {min(gmm['cluster_means']):<20.3f}")
            print(f"{'High cluster mean':<40} {max(gmm['cluster_means']):<20.3f}")
            print(f"{'Low cluster weight':<40} {max(gmm['cluster_weights']):<20.1%}")
            print(f"{'High cluster weight':<40} {min(gmm['cluster_weights']):<20.1%}")
    elif threshold_info['fallback_reason']:
        print(f"{'Fallback reason':<40} {threshold_info['fallback_reason']:<20}")

    # Analyze confidence score distribution
    confidences = [m['confidence_score'] for m in result['mappings']]
    auto_accept = sum(1 for c in confidences if c > 0.89)  # Auto-accept (capped above 89%)
    promoted = sum(1 for c in confidences if 0.80 <= c <= 0.89)  # Promoted (capped at 89%)
    rejected_but_grouped = sum(1 for c in confidences if c < 0.80)  # Edge cases

    print(f"\n{'Confidence Score Distribution':<40} {'Count':<20}")
    print("-" * 80)
    print(f"{'Auto-accept (>89%)':<40} {auto_accept:<20}")
    print(f"{'Promoted (80-89%)':<40} {promoted:<20}")
    print(f"{'Other (<80%)':<40} {rejected_but_grouped:<20}")

    # Show sample groups
    print(f"\nSample Groups (first 5):")
    print("-" * 80)
    groups_shown = set()
    for mapping in result['mappings']:
        if len(groups_shown) >= 5:
            break
        if mapping['group_id'] not in groups_shown:
            groups_shown.add(mapping['group_id'])
            group_members = [m for m in result['mappings'] if m['group_id'] == mapping['group_id']]
            print(f"\nGroup {mapping['group_id']} ({len(group_members)} members):")
            print(f"  Canonical: {mapping['canonical_name']}")
            for member in group_members[:3]:  # Show first 3 members
                if member['original_name'] != mapping['canonical_name']:
                    print(f"    - {member['original_name']} ({member['confidence_score']:.1%})")

    return result


def compare_results(fixed_result, adaptive_result):
    """Compare fixed vs adaptive results."""
    print("\n" + "=" * 80)
    print("COMPARISON: FIXED vs ADAPTIVE")
    print("=" * 80)

    print(f"\n{'Metric':<40} {'Fixed':<20} {'Adaptive':<20} {'Diff':<15}")
    print("-" * 95)

    fixed_groups = fixed_result['summary']['total_groups_created']
    adaptive_groups = adaptive_result['summary']['total_groups_created']
    group_diff = adaptive_groups - fixed_groups

    fixed_reduction = fixed_result['summary']['reduction_percentage']
    adaptive_reduction = adaptive_result['summary']['reduction_percentage']
    reduction_diff = adaptive_reduction - fixed_reduction

    fixed_time = fixed_result['summary']['processing_time_seconds']
    adaptive_time = adaptive_result['summary']['processing_time_seconds']
    time_ratio = adaptive_time / fixed_time if fixed_time > 0 else 0

    print(f"{'Groups created':<40} {fixed_groups:<20} {adaptive_groups:<20} {group_diff:+d}")
    print(f"{'Reduction %':<40} {fixed_reduction:<20.1f} {adaptive_reduction:<20.1f} {reduction_diff:+.1f}%")
    print(f"{'Processing time (s)':<40} {fixed_time:<20.3f} {adaptive_time:<20.3f} {time_ratio:.1f}x")

    print(f"\n{'Key Insights':<80}")
    print("-" * 95)

    if adaptive_groups < fixed_groups:
        print(f"  • Adaptive mode created {abs(group_diff)} FEWER groups (more aggressive grouping)")
    elif adaptive_groups > fixed_groups:
        print(f"  • Adaptive mode created {group_diff} MORE groups (more conservative grouping)")
    else:
        print(f"  • Both modes created the same number of groups")

    print(f"  • Adaptive mode took {time_ratio:.1f}x longer (GMM fitting overhead)")
    print(f"  • Reduction improved by {reduction_diff:.1f} percentage points" if reduction_diff > 0
          else f"  • Reduction decreased by {abs(reduction_diff):.1f} percentage points")

    # Compare group size distributions
    fixed_sizes = {}
    for mapping in fixed_result['mappings']:
        gid = mapping['group_id']
        fixed_sizes[gid] = fixed_sizes.get(gid, 0) + 1

    adaptive_sizes = {}
    for mapping in adaptive_result['mappings']:
        gid = mapping['group_id']
        adaptive_sizes[gid] = adaptive_sizes.get(gid, 0) + 1

    print(f"\n{'Group Size Distribution':<40} {'Fixed':<20} {'Adaptive':<20}")
    print("-" * 95)
    print(f"{'Single-member groups':<40} {sum(1 for s in fixed_sizes.values() if s == 1):<20} "
          f"{sum(1 for s in adaptive_sizes.values() if s == 1):<20}")
    print(f"{'2-5 member groups':<40} {sum(1 for s in fixed_sizes.values() if 2 <= s <= 5):<20} "
          f"{sum(1 for s in adaptive_sizes.values() if 2 <= s <= 5):<20}")
    print(f"{'6+ member groups':<40} {sum(1 for s in fixed_sizes.values() if s >= 6):<20} "
          f"{sum(1 for s in adaptive_sizes.values() if s >= 6):<20}")
    print(f"{'Largest group size':<40} {max(fixed_sizes.values()):<20} {max(adaptive_sizes.values()):<20}")


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("SAMPLE DATA TEST: 500 Company Names")
    print("=" * 80)

    # Load data
    print("\nLoading sample data...")
    names = load_sample_data()
    print(f"Loaded {len(names)} company names")

    # Test both modes
    fixed_result = test_fixed_mode(names)
    adaptive_result = test_adaptive_mode(names)

    # Compare
    compare_results(fixed_result, adaptive_result)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
