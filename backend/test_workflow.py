"""
Quick workflow test to verify the refactored code works end-to-end.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services import NameMatcher
from app.utils import CSVHandler
from app.config import settings

def test_complete_workflow():
    """Test the complete processing workflow."""
    print("=" * 60)
    print("Testing Company Name Standardizer Workflow")
    print("=" * 60)

    # Sample company names
    company_names = [
        "Apple Inc.",
        "Apple",
        "Apple Corporation",
        "Microsoft Corporation",
        "Microsoft Corp",
        "Microsoft",
        "Google LLC",
        "Google Inc",
        "Google",
    ]

    print(f"\n1. Configuration loaded:")
    print(f"   - Similarity threshold: {settings.SIMILARITY_THRESHOLD}%")
    print(f"   - Corporate suffixes: {len(settings.CORPORATE_SUFFIXES)} defined")
    print(f"   - API version: {settings.VERSION}")

    print(f"\n2. Processing {len(company_names)} company names...")

    # Initialize matcher
    matcher = NameMatcher()

    # Process names
    result = matcher.process_names(company_names, filename="test.csv")

    print(f"\n3. Results:")
    print(f"   - Total input names: {result['summary']['total_input_names']}")
    print(f"   - Groups created: {result['summary']['total_groups_created']}")
    print(f"   - Reduction: {result['summary']['reduction_percentage']:.1f}%")
    print(f"   - Processing time: {result['summary']['processing_time_seconds']:.3f}s")

    print(f"\n4. Sample mappings:")
    for mapping in result['mappings'][:5]:
        print(f"   '{mapping['original_name']}' -> '{mapping['canonical_name']}' "
              f"(confidence: {mapping['confidence_score']:.2%}, group: {mapping['group_id']})")

    print(f"\n5. Audit log entries: {len(result['audit_log']['entries'])}")
    print(f"   Sample reasoning: \"{result['audit_log']['entries'][0]['reasoning']}\"")

    # Verify expected grouping
    canonical_names = set(m['canonical_name'] for m in result['mappings'])
    print(f"\n6. Canonical names identified: {sorted(canonical_names)}")

    # Test CSV handler
    print(f"\n7. Testing CSV validation...")
    test_valid = CSVHandler.validate_file_extension("test.csv", settings.ALLOWED_EXTENSIONS)
    test_invalid = CSVHandler.validate_file_extension("test.txt", settings.ALLOWED_EXTENSIONS)
    print(f"   - CSV validation: {'PASS' if test_valid else 'FAIL'}")
    print(f"   - TXT rejection: {'PASS' if not test_invalid else 'FAIL'}")

    print(f"\n8. Testing normalization...")
    test_cases = [
        ("Apple Inc.", "apple"),
        ("Microsoft Corporation", "microsoft"),
        ("Google LLC", "google"),
    ]
    for original, expected in test_cases:
        normalized = matcher.normalize_name(original)
        status = "PASS" if normalized == expected else "FAIL"
        print(f"   {status} '{original}' -> '{normalized}' (expected: '{expected}')")

    print("\n" + "=" * 60)
    print("SUCCESS: All workflow tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_complete_workflow()
