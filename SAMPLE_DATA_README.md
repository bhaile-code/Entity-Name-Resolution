# Sample Dataset Documentation

## Overview

This dataset contains **500 company names** designed to comprehensively test the GMM-based adaptive thresholding system and all fuzzy matching features.

## Dataset Composition

### 1. Clear Match Groups (Auto-Accept Zone)
**~80 major companies with 3-6 variations each**

Examples:
- **Apple**: "Apple Inc.", "Apple Computer Inc.", "Apple Computer", "Apple", "Apple Incorporated"
- **Microsoft**: "Microsoft Corporation", "Microsoft Corp", "Microsoft Corp.", "Microsoft", "Microsoft Company"
- **Google/Alphabet**: "Google LLC", "Google Inc.", "Google", "Alphabet Inc.", "Alphabet"

These should score **≥ T_HIGH** (typically >92%) and be auto-accepted.

### 2. Phonetic Variation Tests
**Companies with sound-alike names**

Examples:
- **Nestlé/Nestle**: Tests accent folding (é → e)
- **L'Oréal/L'Oreal/Loreal**: Tests apostrophe handling + accents
- **Hermès/Hermes**: Tests accent sensitivity
- **Mondelez/Mondelēz**: Tests special characters (ē)

These test:
- `_prepare_for_phonetics()` accent folding via unidecode
- `_calculate_phonetic_bonus()` agreement detection

### 3. Borderline Cases (Promotion Zone)
**Similar names that need phonetic confirmation**

Examples:
- **Twitter/X Corp**: "Twitter Inc.", "Twitter", "X Corp", "X Corporation" (rebrand)
- **Facebook/Meta**: "Facebook Inc.", "Facebook", "Meta Platforms Inc.", "Meta" (rebrand)
- **Daimler/Mercedes-Benz**: "Mercedes-Benz Group AG", "Daimler AG" (corporate restructure)

These should fall in **[S_90, T_HIGH)** range:
- **With phonetic agreement** → promoted and grouped
- **Without phonetic agreement** → rejected

### 4. Acronym and Abbreviation Tests
**Companies with common abbreviations**

Examples:
- **IBM**: "IBM Corporation", "IBM Corp", "IBM", "International Business Machines"
- **3M**: "3M Company", "3M", "Three M Company", "3M Corporation"
- **H&M**: "H&M Hennes & Mauritz AB", "H&M", "H and M", "Hennes & Mauritz"
- **AT&T**: "AT&T Inc.", "AT&T", "American Telephone and Telegraph", "ATT"

These test:
- `_should_use_phonetics()` skip logic for short acronyms
- Phonetic skip for tokens with numbers

### 5. Corporate Suffix Variations
**Same company with different legal suffixes**

Examples:
- **Inc.** vs **Incorporated** vs **Corporation**
- **Corp** vs **Corp.** vs **Corporation**
- **LLC** vs **Limited Liability Company**
- **AG** (German) vs **S.A.** (French) vs **PLC** (British)

These test:
- `normalize_name()` suffix removal from `CORPORATE_SUFFIXES` list

### 6. Punctuation and Special Character Tests
**Names with various punctuation marks**

Examples:
- **Johnson & Johnson**: "Johnson & Johnson", "Johnson and Johnson", "J&J"
- **Procter & Gamble**: "Procter & Gamble", "Procter and Gamble", "P&G"
- **Moët Hennessy**: "LVMH Moët Hennessy Louis Vuitton", "LVMH"
- **McDonald's**: "McDonald's Corporation", "McDonalds Corporation", "McDonald's"

These test:
- Punctuation normalization (& → and)
- Apostrophe handling
- Special character removal

### 7. Different Companies (Reject Zone)
**Similar names that are actually different entities**

Examples:
- **Amazon** vs **Amazon Web Services** (subsidiary, but distinct in some contexts)
- **Morgan Stanley** vs **J.P. Morgan** (different banks)
- **Exxon** vs **Mobil** (merged, but historically separate)
- **Kraft** vs **Heinz** (merged, but historically separate)

These should score **≤ T_LOW** and be rejected (not grouped).

### 8. International Character Tests
**Non-ASCII characters and accents**

Examples:
- **Nestlé** (é)
- **L'Oréal** (é)
- **Hermès** (è)
- **Mondelēz** (ē)
- **Volkswagen** (German)
- **São Paulo Airlines** (ã, if present)

These test the unidecode accent folding pipeline.

### 9. Word Order Variations
**Same company, different word order**

Examples:
- **Walt Disney Company** vs **Disney Company** vs **Disney**
- **Royal Dutch Shell** vs **Shell plc** vs **Shell**
- **Berkshire Hathaway** vs **Berkshire-Hathaway** vs **Berkshire**

These test:
- `token_set_ratio` in fuzzy matching (handles word reordering)

### 10. Numeric Company Names
**Companies with numbers**

Examples:
- **3M**: "3M Company", "3M", "Three M Company"
- **7-Eleven** (if included)

These test:
- Phonetic skip for numeric tokens
- Number-to-word matching ("3M" vs "Three M")

## Expected Behavior

### Fixed Threshold Mode (85%)
- **Expected groups**: ~100-120
- **Reduction**: ~75-80%
- **Processing time**: <1 second
- Uses single threshold: 85%

### Adaptive GMM Mode
- **Expected groups**: ~90-110 (more precise)
- **Reduction**: ~78-82%
- **Processing time**: ~5-10 seconds (GMM fitting overhead)
- Thresholds calculated from data:
  - **T_LOW**: ~0.80-0.85 (reject below)
  - **S_90**: ~0.88-0.92 (promotion eligibility)
  - **T_HIGH**: ~0.93-0.96 (auto-accept above)

### GMM Cluster Characteristics
With this dataset, expect:
- **Low cluster** (different companies): Mean ~0.25-0.35, Weight ~85-90%
- **High cluster** (same companies): Mean ~0.95-0.99, Weight ~10-15%

Most pairs are different companies (low similarity), but the same-company pairs are highly similar.

## Testing Scenarios

### 1. Phonetic Promotion Test
**Look for**: Companies in [S_90, T_HIGH) range with phonetic agreement

Example:
- "Nestlé" ↔ "Nestle": Should be promoted and grouped (phonetics agree)
- "Twitter" ↔ "X Corp": May be rejected (phonetics disagree)

### 2. Confidence Score Validation
**Verify**:
- Auto-accepts: Confidence > 89%
- Promoted pairs: Confidence ≤ 89% (penalty applied)
- Rejected pairs: Not grouped

### 3. Fallback Behavior
**Test with small subset** (<50 pairs):
- Should fallback to fixed threshold
- `threshold_info.fallback_reason` should explain why

### 4. Suffix Handling
**Verify**:
- "Apple Inc." ↔ "Apple Corporation" grouped despite different suffixes
- Suffixes removed during normalization

### 5. Accent Folding
**Verify**:
- "Nestlé" ↔ "Nestle" treated as similar
- "L'Oréal" ↔ "Loreal" grouped

## Dataset Statistics

- **Total names**: 500
- **Unique canonical entities**: ~80-90
- **Average variations per entity**: ~5-6
- **Name types**:
  - US corporations: ~50%
  - International companies: ~30%
  - Banks/Financial: ~10%
  - Tech companies: ~20%
  - Retail/Consumer: ~15%
  - Automotive: ~10%
  - Other: ~15%

## How to Use

### CLI Test
```bash
cd backend
python test_adaptive_workflow.py
```

### Full Stack Test
1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Upload `sample_data_500.csv`
4. Compare Fixed vs Adaptive results

### Expected Output
- **Fixed mode**: Fast, consistent, ~100-120 groups
- **Adaptive mode**: Slower, data-driven, ~90-110 groups (more precise)
- **GMM metadata**: 2 clear clusters visible in results

## Dataset Features Summary

✅ **Clear matches** (high confidence)
✅ **Phonetic variations** (accent folding)
✅ **Borderline cases** (promotion zone)
✅ **Acronyms** (skip phonetics)
✅ **Corporate suffixes** (normalization)
✅ **Punctuation** (& vs and)
✅ **International** (non-ASCII)
✅ **Word order** (token_set_ratio)
✅ **Numbers** (3M, phonetic skip)
✅ **Different entities** (reject zone)

This dataset provides comprehensive coverage of all implemented features and edge cases.
