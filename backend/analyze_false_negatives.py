"""
Analyze false negatives from OPENAI-LARGE + ADAPTIVE configuration.

This script:
1. Loads ground truth mappings
2. Runs the best-performing configuration on the test data
3. Identifies all false negative pairs (should be grouped but weren't)
4. Analyzes patterns and categorizes the false negatives
5. Provides actionable improvement recommendations
"""

import sys
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.services.name_matcher import NameMatcher

def load_ground_truth(filepath: str) -> Dict[str, str]:
    """Load ground truth mappings from CSV."""
    df = pd.read_csv(filepath)
    # Create mapping of original name -> canonical name
    return dict(zip(df['Original Name'], df['Canonical Name']))

def load_test_data(filepath: str) -> List[str]:
    """Load test company names."""
    df = pd.read_csv(filepath)
    return df['company_name'].dropna().tolist()

def get_ground_truth_pairs(mappings: Dict[str, str]) -> Set[Tuple[str, str]]:
    """Get all pairs that should be grouped together according to ground truth."""
    # Group names by their canonical
    groups = defaultdict(list)
    for original, canonical in mappings.items():
        groups[canonical].append(original)

    # Generate all pairs within each group
    pairs = set()
    for canonical, names in groups.items():
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                # Store as sorted tuple to avoid duplicates
                pair = tuple(sorted([name1, name2]))
                pairs.add(pair)

    return pairs

def get_predicted_pairs(result: dict) -> Set[Tuple[str, str]]:
    """Get all pairs grouped together by the algorithm."""
    pairs = set()

    # Group mappings by group_id
    groups = defaultdict(list)
    for mapping in result['mappings']:
        groups[mapping['group_id']].append(mapping['original_name'])

    # Generate all pairs within each predicted group
    for group_id, names in groups.items():
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                pair = tuple(sorted([name1, name2]))
                pairs.add(pair)

    return pairs

def categorize_false_negative(name1: str, name2: str, matcher: NameMatcher) -> Dict:
    """Analyze why a pair was missed and categorize it."""
    norm1 = matcher.normalize_name(name1)
    norm2 = matcher.normalize_name(name2)

    # Calculate component scores
    from rapidfuzz import fuzz
    wratio = fuzz.WRatio(norm1, norm2)
    token_set = fuzz.token_set_ratio(norm1, norm2)

    # Get embedding score if available
    embedding_score = 0.0
    if matcher.embedding_service:
        try:
            embedding_sim = matcher.embedding_service.similarity(norm1, norm2)
            embedding_score = embedding_sim * 100
        except:
            pass

    # Calculate overall confidence
    confidence = matcher.calculate_confidence(name1, name2) * 100

    # Categorize based on characteristics
    categories = []

    # 1. Abbreviation vs full name
    len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
    if len_ratio < 0.5:
        categories.append("ABBREVIATION")

    # 2. Acronym vs full name
    if (len(norm1) <= 5 and len(norm1.split()) == 1 and norm1.isupper()) or \
       (len(norm2) <= 5 and len(norm2.split()) == 1 and norm2.isupper()):
        categories.append("ACRONYM")

    # 3. Typo/misspelling
    if wratio > 70 and confidence < 50:
        categories.append("TYPO")

    # 4. Word order difference
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    jaccard = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
    if jaccard > 0.6 and token_set > 70:
        categories.append("WORD_ORDER")

    # 5. Partial name (subset)
    if words1.issubset(words2) or words2.issubset(words1):
        categories.append("PARTIAL_NAME")

    # 6. Historical name / merger / acquisition
    if any(term in norm1.lower() or term in norm2.lower()
           for term in ['old', 'former', 'merged', 'acquired']):
        categories.append("HISTORICAL")

    # 7. Low semantic similarity despite being same entity
    if embedding_score < 40:
        categories.append("LOW_SEMANTIC")

    # 8. Nickname / informal name
    informal_indicators = ['co', 'corp', 'group', 'inc', 'ltd']
    if len(norm1.split()) == 1 or len(norm2.split()) == 1:
        if any(ind in norm1 or ind in norm2 for ind in informal_indicators):
            categories.append("NICKNAME")

    # 9. International/transliteration issues
    if any(char.isascii() for name in [name1, name2] for char in name):
        if name1 != name1.encode('ascii', 'ignore').decode() or \
           name2 != name2.encode('ascii', 'ignore').decode():
            categories.append("TRANSLITERATION")

    # 10. Common word collision (e.g., "American" in many company names)
    common_words = ['american', 'united', 'general', 'national', 'international',
                    'global', 'world', 'first', 'new', 'group']
    shared_common = [w for w in words1 & words2 if w in common_words]
    if shared_common and len(words1 - words2) > 0 and len(words2 - words1) > 0:
        categories.append("COMMON_WORD_COLLISION")

    if not categories:
        categories.append("OTHER")

    return {
        'name1': name1,
        'name2': name2,
        'norm1': norm1,
        'norm2': norm2,
        'confidence': confidence,
        'wratio': wratio,
        'token_set': token_set,
        'embedding_score': embedding_score,
        'categories': categories
    }

def main():
    print("=" * 80)
    print("FALSE NEGATIVE ANALYSIS - OPENAI-LARGE + ADAPTIVE")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    ground_truth_path = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\ground_truth.csv"
    test_data_path = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\sample_data_500.csv"

    ground_truth = load_ground_truth(ground_truth_path)
    test_names = load_test_data(test_data_path)

    print(f"Loaded {len(ground_truth)} ground truth mappings")
    print(f"Loaded {len(test_names)} test names")
    print()

    # Run best-performing configuration
    print("Running OPENAI-LARGE + ADAPTIVE configuration...")
    matcher = NameMatcher(
        use_adaptive_threshold=True,
        embedding_mode='openai-large'
    )
    result = matcher.process_names(test_names)
    print(f"Processing complete: {result['summary']['total_groups_created']} groups created")
    print()

    # Get true pairs and predicted pairs
    print("Analyzing pairs...")
    gt_pairs = get_ground_truth_pairs(ground_truth)
    pred_pairs = get_predicted_pairs(result)

    # Calculate metrics
    true_positives = gt_pairs & pred_pairs
    false_negatives_pairs = gt_pairs - pred_pairs
    false_positives_pairs = pred_pairs - gt_pairs

    print(f"Ground truth pairs: {len(gt_pairs)}")
    print(f"Predicted pairs: {len(pred_pairs)}")
    print(f"True positives: {len(true_positives)}")
    print(f"False negatives: {len(false_negatives_pairs)}")
    print(f"False positives: {len(false_positives_pairs)}")
    print()

    # Analyze false negatives
    print("Categorizing false negatives...")
    print("-" * 80)

    fn_analyses = []
    for i, (name1, name2) in enumerate(sorted(false_negatives_pairs)):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(false_negatives_pairs)} pairs...")

        analysis = categorize_false_negative(name1, name2, matcher)
        fn_analyses.append(analysis)

    print(f"Analyzed {len(fn_analyses)} false negative pairs")
    print()

    # Aggregate statistics
    print("=" * 80)
    print("FALSE NEGATIVE PATTERNS")
    print("=" * 80)
    print()

    # Category frequency
    category_counts = defaultdict(int)
    for analysis in fn_analyses:
        for cat in analysis['categories']:
            category_counts[cat] += 1

    print("CATEGORY DISTRIBUTION:")
    print("-" * 80)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(fn_analyses)) * 100
        print(f"{cat:30s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Confidence distribution
    confidences = [a['confidence'] for a in fn_analyses]
    print("CONFIDENCE SCORE DISTRIBUTION:")
    print("-" * 80)
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for low, high in bins:
        count = sum(1 for c in confidences if low <= c < high)
        pct = (count / len(confidences)) * 100
        print(f"{low:3d}-{high:3d}%: {count:4d} ({pct:5.1f}%)")
    print()

    # Component score analysis
    wratios = [a['wratio'] for a in fn_analyses]
    token_sets = [a['token_set'] for a in fn_analyses]
    embeddings = [a['embedding_score'] for a in fn_analyses]

    print("COMPONENT SCORE STATISTICS:")
    print("-" * 80)
    print(f"WRatio       - Mean: {sum(wratios)/len(wratios):5.1f}  Min: {min(wratios):5.1f}  Max: {max(wratios):5.1f}")
    print(f"Token Set    - Mean: {sum(token_sets)/len(token_sets):5.1f}  Min: {min(token_sets):5.1f}  Max: {max(token_sets):5.1f}")
    print(f"Embedding    - Mean: {sum(embeddings)/len(embeddings):5.1f}  Min: {min(embeddings):5.1f}  Max: {max(embeddings):5.1f}")
    print()

    # Show examples from each major category
    print("=" * 80)
    print("EXAMPLE FALSE NEGATIVES BY CATEGORY")
    print("=" * 80)
    print()

    for cat in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True)[:5]:
        examples = [a for a in fn_analyses if cat in a['categories']][:3]
        print(f"{cat}:")
        print("-" * 80)
        for ex in examples:
            print(f"  '{ex['name1']}' ↔ '{ex['name2']}'")
            print(f"    Confidence: {ex['confidence']:.1f}% | WRatio: {ex['wratio']:.1f} | Token: {ex['token_set']:.1f} | Embed: {ex['embedding_score']:.1f}")
        print()

    # Save detailed results
    output_file = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\backend\false_negative_analysis.json"
    print(f"Saving detailed analysis to {output_file}...")

    output_data = {
        'summary': {
            'total_fn_pairs': len(false_negatives_pairs),
            'category_counts': dict(category_counts),
            'confidence_stats': {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'component_stats': {
                'wratio': {'mean': sum(wratios)/len(wratios), 'min': min(wratios), 'max': max(wratios)},
                'token_set': {'mean': sum(token_sets)/len(token_sets), 'min': min(token_sets), 'max': max(token_sets)},
                'embedding': {'mean': sum(embeddings)/len(embeddings), 'min': min(embeddings), 'max': max(embeddings)}
            }
        },
        'false_negatives': fn_analyses
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Analysis complete!")
    print()

    # Generate recommendations
    print("=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Based on the false negative analysis, key opportunities for improvement:")
    print()

    # Identify top 3 categories
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (cat, count) in enumerate(top_categories, 1):
        pct = (count / len(fn_analyses)) * 100
        print(f"{i}. {cat}: {count} occurrences ({pct:.1f}% of false negatives)")
    print()

if __name__ == "__main__":
    main()
