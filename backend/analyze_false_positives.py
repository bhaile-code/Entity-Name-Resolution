"""
Analyze potential false positive patterns in the name matching system.

This script identifies pairs that SHOULD NOT match but might score high similarity.
"""
import sys
sys.path.append('.')

from app.services.name_matcher import NameMatcher
from app.config import settings
import pandas as pd

def load_ground_truth():
    """Load ground truth mappings."""
    gt = pd.read_csv('../ground_truth.csv')
    return gt

def identify_risky_pairs(gt_df):
    """
    Identify pairs of companies that should NOT match but have shared characteristics.

    Returns a list of (name1, name2, reason) tuples.
    """
    risky_pairs = []

    # Get all unique canonical names
    canonical_names = gt_df['Canonical Name'].unique()

    # Pattern 1: Shared common words (American, Delta, United, General, etc.)
    common_words = ['American', 'Delta', 'United', 'General', 'Target', 'Oracle',
                    'Meta', 'Continental', 'Adobe', 'Apple']

    for word in common_words:
        matching_canonicals = [c for c in canonical_names if word.lower() in c.lower()]
        if len(matching_canonicals) > 1:
            # Add all pairs
            for i in range(len(matching_canonicals)):
                for j in range(i+1, len(matching_canonicals)):
                    # Get representative names from each canonical group
                    name1_variants = gt_df[gt_df['Canonical Name'] == matching_canonicals[i]]['Original Name'].tolist()
                    name2_variants = gt_df[gt_df['Canonical Name'] == matching_canonicals[j]]['Original Name'].tolist()

                    # Test several combinations
                    risky_pairs.append((name1_variants[0], name2_variants[0],
                                      f"Shared word '{word}' but different companies"))

                    # Also test longer variants
                    if len(name1_variants) > 1 and len(name2_variants) > 1:
                        risky_pairs.append((name1_variants[-1], name2_variants[-1],
                                          f"Shared word '{word}' but different companies (long variants)"))

    # Pattern 2: Abbreviations that could match unrelated companies
    # Example: "American Airlines" abbreviated as "AA" might match "American Express" abbreviated
    abbreviation_pairs = [
        ("American Airlines Inc.", "American Express Company", "Both contain 'American'"),
        ("American Airlines", "American", "Subset matching risk"),
        ("Delta Air Lines", "Delta Dental", "Shared 'Delta' but different industries"),
        ("Delta Airlines", "Delta Faucet Company", "Shared 'Delta' but different industries"),
        ("Delta Air Lines", "Delta Community Credit Union", "Shared 'Delta' but different industries"),
        ("United Airlines", "United Healthcare", "Shared 'United' but different industries"),
        ("United Airlines Inc.", "United Rentals Inc", "Shared 'United' but different companies"),
        ("Target Corporation", "Target Marketing Group", "Shared 'Target' but different companies"),
        ("Target Stores", "Target Media", "Shared 'Target' but different companies"),
        ("Oracle Corporation", "Oracle Financial Services", "Parent company vs service division?"),
        ("Meta Platforms Inc.", "Meta Financial Group", "Shared 'Meta' but completely different"),
        ("Apple Inc.", "Apple Records Limited", "Shared 'Apple' but different (tech vs music)"),
        ("Amazon.com Inc.", "Amazon Logistics LLC", "Parent vs subsidiary - should these match?"),
        ("Continental Airlines", "Continental Tire", "Shared 'Continental' but different industries"),
        ("Adobe Inc.", "Adobe Rent-A-Car", "Shared 'Adobe' but completely different"),
        ("General Electric Company", "General Motors Company", "Shared 'General' but different"),
        ("General Electric", "General Dynamics Corp", "Shared 'General' but different"),
        ("General Electric", "General Mills Inc", "Shared 'General' but different"),
        ("American Standard", "American Airlines", "Shared 'American' but different"),
        ("American Family Insurance", "American Airlines", "Shared 'American' but different"),
        ("Zoom Video Communications Inc.", "Zoom Telephonics Inc", "Shared 'Zoom' but different"),
        ("Domino Printing Sciences", "Dominos Pizza Inc", "Similar name but different"),
        ("J&J", "J.P. Morgan", "Both abbreviations, different companies"),
    ]

    risky_pairs.extend(abbreviation_pairs)

    return risky_pairs

def analyze_matcher_performance(risky_pairs, use_adaptive=False):
    """
    Test the name matcher on risky pairs and report scores.
    """
    # Initialize matcher with best configuration
    matcher = NameMatcher(
        use_adaptive_threshold=use_adaptive,
        embedding_mode='openai-large'
    )

    print(f"\n{'='*100}")
    print(f"TESTING WITH: {'ADAPTIVE GMM' if use_adaptive else 'FIXED THRESHOLD'}")
    print(f"{'='*100}\n")

    results = []

    threshold = matcher.similarity_threshold / 100.0

    for name1, name2, reason in risky_pairs:
        score = matcher.calculate_confidence(name1, name2)

        # Check if they would be grouped (false positive)
        would_match = score >= threshold

        # Get component scores for debugging
        norm1 = matcher.normalize_name(name1)
        norm2 = matcher.normalize_name(name2)

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

        phonetic_bonus = matcher._calculate_phonetic_bonus(norm1, norm2)

        results.append({
            'name1': name1,
            'name2': name2,
            'reason': reason,
            'score': score,
            'would_match': would_match,
            'wratio': wratio,
            'token_set': token_set,
            'embedding': embedding_score,
            'phonetic': phonetic_bonus
        })

    # Sort by score (highest first - most likely false positives)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print results
    print(f"\n{'='*100}")
    print(f"POTENTIAL FALSE POSITIVES (Should NOT match but score high)")
    print(f"{'='*100}\n")

    false_positives = [r for r in results if r['would_match']]
    high_risk = [r for r in results if r['score'] >= 0.75 and not r['would_match']]

    print(f"\nACTUAL FALSE POSITIVES (would be incorrectly merged): {len(false_positives)}\n")
    print(f"{'Name 1':<40} {'Name 2':<40} {'Score':<8} {'WRatio':<8} {'Token':<8} {'Embed':<8} {'Phone':<6}")
    print(f"{'-'*40} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in false_positives:
        print(f"{r['name1'][:38]:<40} {r['name2'][:38]:<40} "
              f"{r['score']*100:6.1f}%  {r['wratio']:6.1f}%  {r['token_set']:6.1f}%  "
              f"{r['embedding']:6.1f}%  {r['phonetic']:+5.1f}")
        print(f"  -> Reason: {r['reason']}")
        print()

    print(f"\nHIGH RISK (score >= 75% but below threshold): {len(high_risk)}\n")
    print(f"{'Name 1':<40} {'Name 2':<40} {'Score':<8} {'WRatio':<8} {'Token':<8} {'Embed':<8} {'Phone':<6}")
    print(f"{'-'*40} {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in high_risk[:10]:  # Show top 10
        print(f"{r['name1'][:38]:<40} {r['name2'][:38]:<40} "
              f"{r['score']*100:6.1f}%  {r['wratio']:6.1f}%  {r['token_set']:6.1f}%  "
              f"{r['embedding']:6.1f}%  {r['phonetic']:+5.1f}")
        print(f"  -> Reason: {r['reason']}")
        print()

    # Statistics
    print(f"\n{'='*100}")
    print(f"STATISTICS")
    print(f"{'='*100}\n")

    print(f"Total risky pairs tested: {len(results)}")
    print(f"False positives (would merge incorrectly): {len(false_positives)} ({len(false_positives)/len(results)*100:.1f}%)")
    print(f"High risk (score >= 75%): {len(high_risk)} ({len(high_risk)/len(results)*100:.1f}%)")
    print(f"Low risk (score < 75%): {len([r for r in results if r['score'] < 0.75])} "
          f"({len([r for r in results if r['score'] < 0.75])/len(results)*100:.1f}%)")

    # Component analysis for false positives
    if false_positives:
        print(f"\nFALSE POSITIVE COMPONENT ANALYSIS:")
        avg_wratio = sum(r['wratio'] for r in false_positives) / len(false_positives)
        avg_token = sum(r['token_set'] for r in false_positives) / len(false_positives)
        avg_embed = sum(r['embedding'] for r in false_positives) / len(false_positives)
        avg_phonetic = sum(r['phonetic'] for r in false_positives) / len(false_positives)

        print(f"  Average WRatio: {avg_wratio:.1f}%")
        print(f"  Average Token Set: {avg_token:.1f}%")
        print(f"  Average Embedding: {avg_embed:.1f}%")
        print(f"  Average Phonetic: {avg_phonetic:+.1f}")

        # Find dominant component
        print(f"\n  Component causing false positives:")
        if avg_token > avg_wratio and avg_token > avg_embed:
            print(f"    WARNING: TOKEN_SET is the main culprit (avg {avg_token:.1f}%)")
        elif avg_wratio > avg_embed:
            print(f"    WARNING: WRATIO is the main culprit (avg {avg_wratio:.1f}%)")
        else:
            print(f"    WARNING: EMBEDDING is the main culprit (avg {avg_embed:.1f}%)")

    return results

def main():
    """Main analysis function."""
    print("Loading ground truth...")
    gt = load_ground_truth()

    print("Identifying risky pairs...")
    risky_pairs = identify_risky_pairs(gt)
    print(f"Found {len(risky_pairs)} risky pairs to test")

    # Remove duplicates
    seen = set()
    unique_pairs = []
    for name1, name2, reason in risky_pairs:
        pair_key = tuple(sorted([name1.lower(), name2.lower()]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append((name1, name2, reason))

    print(f"Testing {len(unique_pairs)} unique risky pairs")

    # Test with current best configuration
    print("\n" + "="*100)
    print("ANALYSIS: FALSE POSITIVE RISKS IN NAME MATCHING SYSTEM")
    print("="*100)

    results = analyze_matcher_performance(unique_pairs, use_adaptive=True)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main()
