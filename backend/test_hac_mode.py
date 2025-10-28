"""
Quick test script to verify HAC clustering mode works correctly.
"""
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.name_matcher import NameMatcher
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_hac_mode():
    """Test HAC clustering mode with sample company names."""

    # Sample company names with clear groupings
    test_names = [
        # Apple group
        "Apple Inc.",
        "Apple",
        "Apple Computer",

        # Microsoft group
        "Microsoft Corporation",
        "Microsoft Corp",
        "Microsoft",

        # Google group
        "Google LLC",
        "Google",
        "Alphabet Inc",

        # Amazon group
        "Amazon.com Inc",
        "Amazon",
        "Amazon.com",

        # Single names (no group)
        "IBM",
        "Tesla",
        "Oracle"
    ]

    print("=" * 80)
    print("HAC MODE TEST")
    print("=" * 80)
    print(f"\nTesting with {len(test_names)} company names\n")

    # Test with HAC mode
    print("Testing HAC mode with threshold=0.42 (58% similarity required)")
    print("-" * 80)

    start_time = time.time()

    matcher = NameMatcher(
        clustering_mode='hac',
        hac_threshold=0.42,
        embedding_mode='openai-small'
    )

    result = matcher.process_names(test_names, filename='test_hac.csv')

    elapsed = time.time() - start_time

    # Display results
    print(f"\nProcessing completed in {elapsed:.2f} seconds")
    print(f"\nTotal groups created: {result['summary']['total_groups_created']}")
    print(f"Reduction: {result['summary']['reduction_percentage']:.1f}%")

    # Display threshold info
    threshold_info = result['summary']['threshold_info']
    print(f"\nClustering Method: {threshold_info['method']}")
    print(f"HAC Threshold: {threshold_info.get('hac_threshold', 'N/A')}")
    print(f"HAC Linkage: {threshold_info.get('hac_linkage', 'N/A')}")

    # Display HAC metadata if available
    if 'hac_metadata' in result:
        meta = result['hac_metadata']
        print(f"\nHAC Metadata:")
        print(f"  Total clusters: {meta['total_clusters']}")
        print(f"  Avg cluster size: {meta['avg_cluster_size']}")
        print(f"  Singleton clusters: {meta['singleton_clusters']}")
        print(f"  Cophenetic correlation: {meta['cophenetic_distance']:.4f}")

    # Display groupings
    print("\nGroupings:")
    print("-" * 80)

    groups = {}
    for mapping in result['mappings']:
        canonical = mapping['canonical_name']
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append({
            'original': mapping['original_name'],
            'confidence': mapping['confidence_score']
        })

    for canonical, members in groups.items():
        print(f"\n{canonical} ({len(members)} members):")
        for member in members:
            if member['original'] == canonical:
                print(f"  - {member['original']} (CANONICAL)")
            else:
                print(f"  - {member['original']} ({member['confidence']:.1%} confidence)")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return result


def test_different_thresholds():
    """Test HAC with different thresholds to show behavior."""

    test_names = [
        "Apple Inc.", "Apple", "Microsoft Corp", "Microsoft",
        "Google", "Alphabet Inc", "Amazon.com", "Amazon"
    ]

    thresholds = [0.15, 0.30, 0.42, 0.60]

    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON TEST")
    print("=" * 80)

    for threshold in thresholds:
        similarity_pct = (1 - threshold) * 100
        print(f"\nThreshold: {threshold:.2f} (Similarity: {similarity_pct:.0f}%)")
        print("-" * 40)

        matcher = NameMatcher(
            clustering_mode='hac',
            hac_threshold=threshold,
            embedding_mode='openai-small'
        )

        result = matcher.process_names(test_names, filename=f'test_hac_{threshold}.csv')

        groups_count = result['summary']['total_groups_created']
        reduction_pct = result['summary']['reduction_percentage']

        print(f"  Groups: {groups_count}/{len(test_names)}")
        print(f"  Reduction: {reduction_pct:.1f}%")


if __name__ == '__main__':
    try:
        # Run main HAC test
        result = test_hac_mode()

        # Run threshold comparison test
        test_different_thresholds()

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n ERROR: {e}")
        exit(1)
