"""
Find actual false positives from the performance test results.

This script identifies the 77 false positive pairs that were incorrectly merged.
"""
import sys
sys.path.append('.')

from app.services.name_matcher import NameMatcher
import pandas as pd
from collections import defaultdict

def load_ground_truth():
    """Load ground truth mappings."""
    gt = pd.read_csv('../ground_truth.csv')
    # Create mapping from original name to canonical name
    gt_mapping = dict(zip(gt['Original Name'], gt['Canonical Name']))
    return gt, gt_mapping

def test_system_and_find_errors():
    """Run the system and find false positives."""

    # Load data
    sample_data = pd.read_csv('../sample_data_500.csv')
    gt_df, gt_mapping = load_ground_truth()

    # Get all names
    names = sample_data['company_name'].dropna().tolist()
    print(f"Processing {len(names)} company names...")

    # Run the matcher
    matcher = NameMatcher(use_adaptive_threshold=True, embedding_mode='openai-large')
    result = matcher.process_names(names, filename='sample_data_500.csv')

    # Build predicted mapping from results
    predicted_mapping = {}
    for mapping in result['mappings']:
        predicted_mapping[mapping['original_name']] = mapping['canonical_name']

    print(f"\nSystem created {result['summary']['total_groups_created']} groups")

    # Find false positives: pairs grouped together by system but should be separate
    false_positives = []
    false_positive_pairs = []

    # Build groups from predictions
    pred_groups = defaultdict(list)
    for orig, canon in predicted_mapping.items():
        pred_groups[canon].append(orig)

    # Check each predicted group
    for canon, members in pred_groups.items():
        if len(members) > 1:
            # Get ground truth canonical names for all members
            gt_canonicals = set()
            for member in members:
                if member in gt_mapping:
                    gt_canonicals.add(gt_mapping[member])

            # If members belong to different ground truth groups, we have false positives
            if len(gt_canonicals) > 1:
                # This group contains names from multiple ground truth groups
                print(f"\nFALSE POSITIVE GROUP: {canon}")
                print(f"  Members ({len(members)}):")

                # Organize by ground truth canonical
                by_gt_canon = defaultdict(list)
                for member in members:
                    if member in gt_mapping:
                        by_gt_canon[gt_mapping[member]].append(member)
                    else:
                        by_gt_canon['UNKNOWN'].append(member)

                for gt_canon, gt_members in by_gt_canon.items():
                    print(f"    GT: {gt_canon}")
                    for m in gt_members:
                        print(f"      - {m}")

                # Count pairwise false positives
                # Each pair of names from different GT groups is a false positive
                members_with_gt = [(m, gt_mapping[m]) for m in members if m in gt_mapping]
                for i in range(len(members_with_gt)):
                    for j in range(i+1, len(members_with_gt)):
                        name1, gt1 = members_with_gt[i]
                        name2, gt2 = members_with_gt[j]
                        if gt1 != gt2:
                            false_positives.append((name1, name2, gt1, gt2, canon))

                            # Calculate scores for this pair
                            score = matcher.calculate_confidence(name1, name2)
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

                            phonetic = matcher._calculate_phonetic_bonus(norm1, norm2)

                            false_positive_pairs.append({
                                'name1': name1,
                                'name2': name2,
                                'gt1': gt1,
                                'gt2': gt2,
                                'predicted_canon': canon,
                                'score': score * 100,
                                'wratio': wratio,
                                'token_set': token_set,
                                'embedding': embedding_score,
                                'phonetic': phonetic,
                                'norm1': norm1,
                                'norm2': norm2
                            })

    print(f"\n{'='*100}")
    print(f"FALSE POSITIVE ANALYSIS")
    print(f"{'='*100}\n")

    print(f"Total false positive pairs found: {len(false_positive_pairs)}")

    if false_positive_pairs:
        # Sort by score (highest first)
        false_positive_pairs.sort(key=lambda x: x['score'], reverse=True)

        print(f"\nTop 20 False Positives (by score):\n")
        print(f"{'Name 1':<35} {'Name 2':<35} {'GT1':<20} {'GT2':<20} {'Score':<7} {'WR':<6} {'TK':<6} {'EM':<6} {'PH':<5}")
        print(f"{'-'*35} {'-'*35} {'-'*20} {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

        for fp in false_positive_pairs[:20]:
            print(f"{fp['name1'][:33]:<35} {fp['name2'][:33]:<35} "
                  f"{fp['gt1'][:18]:<20} {fp['gt2'][:18]:<20} "
                  f"{fp['score']:5.1f}%  {fp['wratio']:4.0f}%  {fp['token_set']:4.0f}%  "
                  f"{fp['embedding']:4.0f}%  {fp['phonetic']:+4.1f}")

        # Analyze patterns
        print(f"\n{'='*100}")
        print(f"PATTERN ANALYSIS")
        print(f"{'='*100}\n")

        # Group by problem type
        shared_word_fps = []
        subset_fps = []
        typo_fps = []
        abbreviation_fps = []

        for fp in false_positive_pairs:
            norm1_words = set(fp['norm1'].split())
            norm2_words = set(fp['norm2'].split())
            shared_words = norm1_words & norm2_words

            if len(shared_words) > 0:
                shared_word_fps.append((fp, list(shared_words)))

            if norm1_words.issubset(norm2_words) or norm2_words.issubset(norm1_words):
                subset_fps.append(fp)

        print(f"1. SHARED WORD PROBLEM: {len(shared_word_fps)} pairs ({len(shared_word_fps)/len(false_positive_pairs)*100:.1f}%)")
        print(f"   Examples of shared words causing problems:")
        word_freq = defaultdict(int)
        for fp, words in shared_word_fps:
            for word in words:
                if len(word) > 2:  # Skip short words
                    word_freq[word] += 1

        top_problem_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for word, count in top_problem_words:
            print(f"     - '{word}': {count} false positives")

        print(f"\n2. SUBSET PROBLEM: {len(subset_fps)} pairs ({len(subset_fps)/len(false_positive_pairs)*100:.1f}%)")
        if subset_fps:
            print(f"   Examples:")
            for fp in subset_fps[:5]:
                print(f"     - '{fp['name1']}' vs '{fp['name2']}'")
                print(f"       Normalized: '{fp['norm1']}' vs '{fp['norm2']}'")

        # Component analysis
        print(f"\n3. COMPONENT SCORE ANALYSIS:")
        avg_score = sum(fp['score'] for fp in false_positive_pairs) / len(false_positive_pairs)
        avg_wratio = sum(fp['wratio'] for fp in false_positive_pairs) / len(false_positive_pairs)
        avg_token = sum(fp['token_set'] for fp in false_positive_pairs) / len(false_positive_pairs)
        avg_embed = sum(fp['embedding'] for fp in false_positive_pairs) / len(false_positive_pairs)
        avg_phonetic = sum(fp['phonetic'] for fp in false_positive_pairs) / len(false_positive_pairs)

        print(f"   Average final score: {avg_score:.1f}%")
        print(f"   Average WRatio: {avg_wratio:.1f}%")
        print(f"   Average Token Set: {avg_token:.1f}%")
        print(f"   Average Embedding: {avg_embed:.1f}%")
        print(f"   Average Phonetic: {avg_phonetic:+.1f}")

        print(f"\n   Dominan component:")
        if avg_token > avg_wratio and avg_token > avg_embed:
            print(f"     TOKEN_SET is the main driver of false positives (avg {avg_token:.1f}%)")
        elif avg_wratio > avg_embed:
            print(f"     WRATIO is the main driver of false positives (avg {avg_wratio:.1f}%)")
        else:
            print(f"     EMBEDDING is the main driver of false positives (avg {avg_embed:.1f}%)")

        # Score distribution
        print(f"\n4. SCORE DISTRIBUTION:")
        score_ranges = [(85, 100), (80, 85), (75, 80), (70, 75), (0, 70)]
        for low, high in score_ranges:
            count = len([fp for fp in false_positive_pairs if low <= fp['score'] < high])
            if count > 0:
                print(f"   {low}-{high}%: {count} pairs ({count/len(false_positive_pairs)*100:.1f}%)")

        # Save detailed results
        import json
        with open('false_positive_analysis.json', 'w') as f:
            json.dump(false_positive_pairs, f, indent=2)
        print(f"\n   Detailed results saved to: false_positive_analysis.json")

if __name__ == "__main__":
    test_system_and_find_errors()
