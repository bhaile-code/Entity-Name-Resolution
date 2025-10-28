"""
Test with ADAPTIVE GMM mode - this is where the problem likely occurred
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

# Test data - replicate your original problem
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
    "AmEx",
]

print("=" * 100)
print("PROBLEM REPRODUCTION: Adaptive GMM Mode WITHOUT Embeddings")
print("=" * 100)
print("\nThis should show the bug where everything matches 'American' at 100%\n")

# Test with adaptive mode, embeddings disabled
matcher_broken = NameMatcher(use_adaptive_threshold=True, embedding_mode='disabled')
result_broken = matcher_broken.process_names(test_names, filename="test.csv")

print(f"Total groups: {result_broken['summary']['total_groups_created']}")
print(f"Threshold info: {result_broken['summary']['threshold_info']}\n")

mappings = sorted(result_broken['mappings'], key=lambda x: x['original_name'])
for m in mappings:
    conf_pct = m['confidence_score'] * 100
    canonical = m['canonical_name']
    problem = "❌ BUG!" if "American" in m['original_name'] and "American" in canonical and m['original_name'] != canonical and len(canonical) < 15 else ""
    print(f"{m['original_name']:40} → {canonical:30} {conf_pct:5.1f}%  {problem}")

print("\n" + "=" * 100)
print("SOLUTION: Adaptive GMM Mode WITH Embeddings")
print("=" * 100)
print("\nThis should correctly separate different companies\n")

# Test with adaptive mode + embeddings
matcher_fixed = NameMatcher(use_adaptive_threshold=True, embedding_mode='openai-small')
result_fixed = matcher_fixed.process_names(test_names, filename="test.csv")

print(f"Total groups: {result_fixed['summary']['total_groups_created']}")
print(f"Threshold info: {result_fixed['summary']['threshold_info']}\n")

mappings_fixed = sorted(result_fixed['mappings'], key=lambda x: x['original_name'])
for m in mappings_fixed:
    conf_pct = m['confidence_score'] * 100
    canonical = m['canonical_name']
    status = "✓ Fixed" if "American" in m['original_name'] and canonical in m['original_name'] else ""
    print(f"{m['original_name']:40} → {canonical:30} {conf_pct:5.1f}%  {status}")

print("\n" + "=" * 100)
