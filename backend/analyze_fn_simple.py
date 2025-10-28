"""
Quick false negative analysis - outputs key stats.
"""

import sys
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.services.name_matcher import NameMatcher

def load_ground_truth(filepath: str) -> Dict[str, str]:
    df = pd.read_csv(filepath)
    return dict(zip(df['Original Name'], df['Canonical Name']))

def load_test_data(filepath: str) -> List[str]:
    df = pd.read_csv(filepath)
    return df['company_name'].dropna().tolist()

def get_ground_truth_pairs(mappings: Dict[str, str]) -> Set[Tuple[str, str]]:
    groups = defaultdict(list)
    for original, canonical in mappings.items():
        groups[canonical].append(original)

    pairs = set()
    for canonical, names in groups.items():
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                pair = tuple(sorted([name1, name2]))
                pairs.add(pair)
    return pairs

def get_predicted_pairs(result: dict) -> Set[Tuple[str, str]]:
    pairs = set()
    groups = defaultdict(list)
    for mapping in result['mappings']:
        groups[mapping['group_id']].append(mapping['original_name'])

    for group_id, names in groups.items():
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                pair = tuple(sorted([name1, name2]))
                pairs.add(pair)
    return pairs

def categorize_fn(name1: str, name2: str, matcher: NameMatcher) -> Dict:
    norm1 = matcher.normalize_name(name1)
    norm2 = matcher.normalize_name(name2)

    from rapidfuzz import fuzz
    wratio = fuzz.WRatio(norm1, norm2)
    token_set = fuzz.token_set_ratio(norm1, norm2)

    embedding_score = 0.0
    if matcher.embedding_service:
        try:
            embedding_sim = matcher.embedding_service.similarity(norm1, norm2)
            embedding_score = embedding_sim * 100
        except:
            pass

    confidence = matcher.calculate_confidence(name1, name2) * 100

    categories = []

    # Categorization logic
    len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
    if len_ratio < 0.5:
        categories.append("ABBREVIATION")

    if (len(norm1) <= 5 and len(norm1.split()) == 1 and norm1.isupper()) or \
       (len(norm2) <= 5 and len(norm2.split()) == 1 and norm2.isupper()):
        categories.append("ACRONYM")

    if wratio > 70 and confidence < 50:
        categories.append("TYPO")

    words1 = set(norm1.split())
    words2 = set(norm2.split())
    jaccard = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
    if jaccard > 0.6 and token_set > 70:
        categories.append("WORD_ORDER")

    if words1.issubset(words2) or words2.issubset(words1):
        categories.append("PARTIAL_NAME")

    if embedding_score < 40:
        categories.append("LOW_SEMANTIC")

    if len(norm1.split()) == 1 or len(norm2.split()) == 1:
        categories.append("NICKNAME")

    if not categories:
        categories.append("OTHER")

    return {
        'name1': name1,
        'name2': name2,
        'confidence': confidence,
        'wratio': wratio,
        'token_set': token_set,
        'embedding_score': embedding_score,
        'categories': categories
    }

# Load and process
gt_path = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\ground_truth.csv"
test_path = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\sample_data_500.csv"

print("Loading data...")
ground_truth = load_ground_truth(gt_path)
test_names = load_test_data(test_path)

print("Running matcher...")
matcher = NameMatcher(use_adaptive_threshold=True, embedding_mode='openai-large')
result = matcher.process_names(test_names)

print("Analyzing pairs...")
gt_pairs = get_ground_truth_pairs(ground_truth)
pred_pairs = get_predicted_pairs(result)
fn_pairs = gt_pairs - pred_pairs

print(f"\nFound {len(fn_pairs)} false negative pairs")
print("Analyzing...")

fn_analyses = []
for i, (name1, name2) in enumerate(sorted(fn_pairs)):
    if i % 100 == 0:
        print(f"  {i}/{len(fn_pairs)}...")
    analysis = categorize_fn(name1, name2, matcher)
    fn_analyses.append(analysis)

# Stats
category_counts = defaultdict(int)
for analysis in fn_analyses:
    for cat in analysis['categories']:
        category_counts[cat] += 1

confidences = [a['confidence'] for a in fn_analyses]
wratios = [a['wratio'] for a in fn_analyses]
token_sets = [a['token_set'] for a in fn_analyses]
embeddings = [a['embedding_score'] for a in fn_analyses]

# Save JSON
output = {
    'summary': {
        'total_fn_pairs': len(fn_pairs),
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
    'false_negatives': fn_analyses[:50]  # Sample only
}

output_file = r"c:\Users\Beemnet\Documents\Prototypes\Entity Name Resolution v2\backend\fn_analysis.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_file}")
print("\nCategory distribution:")
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(fn_analyses)) * 100
    print(f"  {cat:25s}: {count:3d} ({pct:5.1f}%)")
