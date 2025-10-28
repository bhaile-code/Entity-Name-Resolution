"""
Comprehensive embedding test script.
Tests both openai-small and openai-large models in fixed and adaptive modes.
"""
import csv
import time
from pathlib import Path
from app.services.name_matcher import NameMatcher
from app.services.embedding_service import create_embedding_service


def load_sample_data():
    """Load company names from sample CSV."""
    csv_path = Path(__file__).parent.parent / "sample_data_500.csv"

    names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row['company_name'])

    return names


def analyze_confidence_distribution(result):
    """Analyze and return confidence score distribution."""
    confidences = [m['confidence_score'] for m in result['mappings']]

    return {
        'total': len(confidences),
        'perfect_100': sum(1 for c in confidences if c >= 0.999),
        'high_95_99': sum(1 for c in confidences if 0.95 <= c < 0.999),
        'good_90_95': sum(1 for c in confidences if 0.90 <= c < 0.95),
        'medium_85_90': sum(1 for c in confidences if 0.85 <= c < 0.90),
        'low_80_85': sum(1 for c in confidences if 0.80 <= c < 0.85),
        'borderline_70_80': sum(1 for c in confidences if 0.70 <= c < 0.80),
        'weak_60_70': sum(1 for c in confidences if 0.60 <= c < 0.70),
        'poor_below_60': sum(1 for c in confidences if c < 0.60),
        'min': min(confidences) if confidences else 0,
        'max': max(confidences) if confidences else 0,
        'avg': sum(confidences) / len(confidences) if confidences else 0,
    }


def print_distribution(dist, label):
    """Print confidence distribution in a formatted table."""
    print(f"\n{label} - Confidence Distribution:")
    print("=" * 80)
    print(f"{'Range':<20} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    total = dist['total']
    print(f"{'100%':<20} {dist['perfect_100']:<10} {dist['perfect_100']/total*100:>6.1f}%")
    print(f"{'95-99%':<20} {dist['high_95_99']:<10} {dist['high_95_99']/total*100:>6.1f}%")
    print(f"{'90-95%':<20} {dist['good_90_95']:<10} {dist['good_90_95']/total*100:>6.1f}%")
    print(f"{'85-90%':<20} {dist['medium_85_90']:<10} {dist['medium_85_90']/total*100:>6.1f}%")
    print(f"{'80-85%':<20} {dist['low_80_85']:<10} {dist['low_80_85']/total*100:>6.1f}%")
    print(f"{'70-80%':<20} {dist['borderline_70_80']:<10} {dist['borderline_70_80']/total*100:>6.1f}%")
    print(f"{'60-70%':<20} {dist['weak_60_70']:<10} {dist['weak_60_70']/total*100:>6.1f}%")
    print(f"{'<60%':<20} {dist['poor_below_60']:<10} {dist['poor_below_60']/total*100:>6.1f}%")
    print("-" * 80)
    print(f"{'Min/Max/Avg':<20} {dist['min']:.2%} / {dist['max']:.2%} / {dist['avg']:.2%}")


def show_sample_groups(result, n=10):
    """Show sample groups with varied confidence scores."""
    print(f"\n\nSample Groups with Confidence Variation (showing up to {n}):")
    print("=" * 80)

    # Find groups with multiple members and varied confidence
    interesting_groups = []

    groups_dict = {}
    for mapping in result['mappings']:
        gid = mapping['group_id']
        if gid not in groups_dict:
            groups_dict[gid] = []
        groups_dict[gid].append(mapping)

    # Sort by group size (larger first) and confidence variation
    for gid, members in groups_dict.items():
        if len(members) > 1:
            confidences = [m['confidence_score'] for m in members]
            variation = max(confidences) - min(confidences)
            interesting_groups.append((gid, members, len(members), variation))

    # Sort by size then variation
    interesting_groups.sort(key=lambda x: (x[2], x[3]), reverse=True)

    shown = 0
    for gid, members, size, variation in interesting_groups[:n]:
        print(f"\nGroup {gid} ({size} members, variation: {variation:.1%}):")
        print(f"  Canonical: {members[0]['canonical_name']}")
        for member in members[:8]:  # Show max 8 per group
            if member['original_name'] != members[0]['canonical_name']:
                print(f"    - {member['original_name']:<45} {member['confidence_score']:>6.1%}")
        if len(members) > 8:
            print(f"    ... and {len(members) - 8} more")
        shown += 1


def test_mode(embedding_mode, use_adaptive, names):
    """Test a specific embedding mode and threshold mode."""
    mode_label = f"{embedding_mode.upper()} + {'ADAPTIVE' if use_adaptive else 'FIXED'}"
    print("\n" + "=" * 80)
    print(f"TESTING: {mode_label}")
    print("=" * 80)

    start_time = time.time()

    # Create matcher with specific embedding mode
    matcher = NameMatcher(
        use_adaptive_threshold=use_adaptive,
        embedding_mode=embedding_mode
    )

    # Process names
    result = matcher.process_names(names, filename="sample_data_500.csv")

    elapsed = time.time() - start_time

    # Print summary
    summary = result['summary']
    print(f"\n{'Metric':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Total input names':<40} {summary['total_input_names']:<20}")
    print(f"{'Groups created':<40} {summary['total_groups_created']:<20}")
    print(f"{'Reduction percentage':<40} {summary['reduction_percentage']:<20.1f}%")
    print(f"{'Average group size':<40} {summary['average_group_size']:<20.2f}")
    print(f"{'Processing time':<40} {elapsed:<20.2f}s")

    # Threshold info
    threshold_info = summary['threshold_info']
    print(f"\n{'Threshold Configuration':<40} {'Value':<20}")
    print("-" * 80)
    print(f"{'Method':<40} {threshold_info['method']:<20}")
    if threshold_info['method'] == 'adaptive_gmm':
        print(f"{'T_LOW (reject below)':<40} {threshold_info['t_low']:<20.3f}")
        print(f"{'S_90 (promotion threshold)':<40} {threshold_info['s_90']:<20.3f}")
        print(f"{'T_HIGH (auto-accept above)':<40} {threshold_info['t_high']:<20.3f}")
    else:
        print(f"{'Fixed threshold':<40} {threshold_info['fixed_threshold']:<20.1f}%")

    # Analyze and print confidence distribution
    dist = analyze_confidence_distribution(result)
    print_distribution(dist, mode_label)

    # Show interesting groups
    show_sample_groups(result, n=10)

    return result, dist


def compare_results(results):
    """Compare results across all test modes."""
    print("\n" + "=" * 80)
    print("COMPARISON ACROSS ALL MODES")
    print("=" * 80)

    print(f"\n{'Mode':<35} {'Groups':<12} {'Reduction':<12} {'Time (s)':<12}")
    print("-" * 80)

    for mode_label, result, dist in results:
        summary = result['summary']
        groups = summary['total_groups_created']
        reduction = summary['reduction_percentage']
        time_sec = summary['processing_time_seconds']
        print(f"{mode_label:<35} {groups:<12} {reduction:<12.1f} {time_sec:<12.2f}")

    print(f"\n{'Mode':<35} {'100%':<8} {'95-99%':<8} {'90-95%':<8} {'85-90%':<8} {'<85%':<8}")
    print("-" * 80)

    for mode_label, result, dist in results:
        below_85 = dist['low_80_85'] + dist['borderline_70_80'] + dist['weak_60_70'] + dist['poor_below_60']
        print(f"{mode_label:<35} {dist['perfect_100']:<8} {dist['high_95_99']:<8} "
              f"{dist['good_90_95']:<8} {dist['medium_85_90']:<8} {below_85:<8}")


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EMBEDDING TEST")
    print("Testing: openai-small and openai-large in fixed and adaptive modes")
    print("=" * 80)

    # Load data
    print("\nLoading sample data...")
    names = load_sample_data()
    print(f"Loaded {len(names)} company names")

    # Test all combinations
    results = []

    # Test 1: openai-small + fixed
    result, dist = test_mode('openai-small', False, names)
    results.append(("OpenAI Small + Fixed", result, dist))

    # Test 2: openai-small + adaptive
    result, dist = test_mode('openai-small', True, names)
    results.append(("OpenAI Small + Adaptive", result, dist))

    # Test 3: openai-large + fixed
    result, dist = test_mode('openai-large', False, names)
    results.append(("OpenAI Large + Fixed", result, dist))

    # Test 4: openai-large + adaptive
    result, dist = test_mode('openai-large', True, names)
    results.append(("OpenAI Large + Adaptive", result, dist))

    # Compare all results
    compare_results(results)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
