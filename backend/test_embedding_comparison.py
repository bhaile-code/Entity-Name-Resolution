"""
Test script to compare original algorithm vs embeddings-enhanced algorithm
on the problematic "American" companies dataset.
"""
import sys
import os
import io
from pathlib import Path

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.name_matcher import NameMatcher

# Problematic test data
test_names = [
    "American",
    "American Airlines",
    "American Airlines Group Inc.",
    "American Airlines Inc.",
    "American Express",
    "American Express Company",
    "American Express Corporation",
    "American Telephone and Telegraph",
    "Amex",
    "AmEx"
]

print("=" * 80)
print("TESTING: Original Algorithm (Embeddings DISABLED)")
print("=" * 80)

# Test 1: Original algorithm (embeddings disabled)
matcher_old = NameMatcher(embedding_mode='disabled')
result_old = matcher_old.process_names(test_names, filename="test.csv")

print(f"\nTotal groups created: {result_old['summary']['total_groups_created']}")
print(f"Reduction: {result_old['summary']['reduction_percentage']:.1f}%\n")

print("Mappings (sorted by canonical name):")
print("-" * 80)
mappings_old = sorted(result_old['mappings'], key=lambda x: (x['canonical_name'], x['original_name']))
for mapping in mappings_old:
    confidence_pct = mapping['confidence_score'] * 100
    print(f"{mapping['original_name']:40} → {mapping['canonical_name']:30} {confidence_pct:5.1f}% (Group {mapping['group_id']})")

print("\n" + "=" * 80)
print("TESTING: NEW Algorithm (Embeddings ENABLED - OpenAI)")
print("=" * 80)

# Test 2: New algorithm with embeddings
matcher_new = NameMatcher(embedding_mode='openai-small')
result_new = matcher_new.process_names(test_names, filename="test.csv")

print(f"\nTotal groups created: {result_new['summary']['total_groups_created']}")
print(f"Reduction: {result_new['summary']['reduction_percentage']:.1f}%\n")

print("Mappings (sorted by canonical name):")
print("-" * 80)
mappings_new = sorted(result_new['mappings'], key=lambda x: (x['canonical_name'], x['original_name']))
for mapping in mappings_new:
    confidence_pct = mapping['confidence_score'] * 100
    print(f"{mapping['original_name']:40} → {mapping['canonical_name']:30} {confidence_pct:5.1f}% (Group {mapping['group_id']})")

print("\n" + "=" * 80)
print("COMPARISON: Key Differences")
print("=" * 80)

# Analyze differences
def get_group_mapping(mappings):
    """Create dict of original_name -> canonical_name"""
    return {m['original_name']: m['canonical_name'] for m in mappings}

old_groups = get_group_mapping(mappings_old)
new_groups = get_group_mapping(mappings_new)

print("\nCritical Test Cases:")
print("-" * 80)

test_cases = [
    ("American Express", "American"),
    ("American Express Company", "American"),
    ("American Airlines", "American"),
    ("American Airlines Inc.", "American Airlines"),
]

for original, expected_reject in test_cases:
    old_canonical = old_groups.get(original, "NOT FOUND")
    new_canonical = new_groups.get(original, "NOT FOUND")

    old_match = "✓ CORRECT" if old_canonical != expected_reject else "✗ WRONG"
    new_match = "✓ CORRECT" if new_canonical != expected_reject else "✗ WRONG"

    print(f"\n{original}:")
    print(f"  OLD: → {old_canonical:30} {old_match}")
    print(f"  NEW: → {new_canonical:30} {new_match}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

old_correct = sum(1 for orig, exp_rej in test_cases if old_groups.get(orig) != exp_rej)
new_correct = sum(1 for orig, exp_rej in test_cases if new_groups.get(orig) != exp_rej)

print(f"\nOLD Algorithm: {old_correct}/{len(test_cases)} test cases correct")
print(f"NEW Algorithm: {new_correct}/{len(test_cases)} test cases correct")
print(f"\nImprovement: {((new_correct - old_correct) / len(test_cases)) * 100:+.0f}%")

if new_correct == len(test_cases):
    print("\n🎉 SUCCESS! All problematic cases now handled correctly!")
else:
    print(f"\n⚠️  Still {len(test_cases) - new_correct} cases incorrect")

print("\n" + "=" * 80)
