"""
Tests for the NameMatcher service.
"""
import pytest
from app.services import NameMatcher


def test_normalize_name():
    """Test name normalization."""
    matcher = NameMatcher()

    assert matcher.normalize_name("Apple Inc.") == "apple"
    assert matcher.normalize_name("Microsoft Corporation") == "microsoft"
    assert matcher.normalize_name("Google LLC") == "google"
    assert matcher.normalize_name("Amazon.com, Inc.") == "amazon com"


def test_select_canonical_name():
    """Test canonical name selection."""
    matcher = NameMatcher()

    # Should select shortest name
    names = ["Apple Inc.", "Apple", "Apple Corporation"]
    assert matcher.select_canonical_name(names) == "Apple"

    # Single name
    assert matcher.select_canonical_name(["Single"]) == "Single"

    # Empty list
    assert matcher.select_canonical_name([]) == ""


def test_calculate_confidence():
    """Test confidence score calculation."""
    matcher = NameMatcher()

    # Identical normalized names
    score = matcher.calculate_confidence("Apple Inc.", "Apple Incorporated")
    assert score > 0.9

    # Similar names
    score = matcher.calculate_confidence("Microsoft", "Microsoft Corp")
    assert score > 0.8

    # Different names
    score = matcher.calculate_confidence("Apple", "Google")
    assert score < 0.5


def test_group_similar_names():
    """Test grouping of similar names."""
    matcher = NameMatcher()

    names = [
        "Apple Inc.",
        "Apple",
        "Apple Corporation",
        "Google LLC",
        "Google",
        "Microsoft"
    ]

    groups = matcher.group_similar_names(names)

    # Should create approximately 3 groups (Apple, Google, Microsoft)
    assert 2 <= len(groups) <= 4

    # Total names should be preserved
    total_names = sum(len(group) for group in groups)
    assert total_names == len(names)


def test_process_names():
    """Test complete name processing pipeline."""
    matcher = NameMatcher()

    names = [
        "Apple Inc.",
        "Apple",
        "Google LLC",
        "Google"
    ]

    result = matcher.process_names(names, filename="test.csv")

    # Check structure
    assert "mappings" in result
    assert "audit_log" in result
    assert "summary" in result

    # Check mappings
    assert len(result["mappings"]) == len(names)

    # Check audit log
    assert result["audit_log"]["filename"] == "test.csv"
    assert result["audit_log"]["total_names"] == len(names)
    assert len(result["audit_log"]["entries"]) == len(names)

    # Check summary
    assert result["summary"]["total_input_names"] == len(names)
    assert result["summary"]["total_groups_created"] >= 1


def test_phonetic_agreement():
    """Test that phonetically similar names get a bonus."""
    matcher = NameMatcher()

    # "Smith" and "Smyth" sound the same - should get +4% bonus
    score_smith = matcher.calculate_confidence("Smith Inc.", "Smyth Inc.")
    score_smith_no_phonetic = matcher.calculate_confidence("Smith Inc.", "Smith Inc.")

    # Should be high confidence due to phonetic agreement
    # Base fuzzy score ~80% + 4% phonetic bonus = 84%
    assert score_smith >= 0.83

    # Test with common phonetic variations
    score_steven = matcher.calculate_confidence("Steven Corp", "Stephen Corp")
    assert score_steven > 0.75  # Different spelling, but phonetically similar


def test_phonetic_disagreement():
    """Test that phonetically different names get a penalty."""
    matcher = NameMatcher()

    # "Apple" and "Orange" are completely different
    score = matcher.calculate_confidence("Apple", "Orange")
    # Base fuzzy score ~36% - 2% phonetic penalty = 34%
    assert score < 0.35  # Should be low due to phonetic disagreement


def test_phonetic_skip_numbers():
    """Test that tokens with numbers skip phonetic processing."""
    matcher = NameMatcher()

    # "3M" contains a number - phonetics should be skipped
    score = matcher.calculate_confidence("3M Company", "3M Corp")
    assert score > 0.85  # Should still match well via fuzzy matching

    # "7-Eleven" contains numbers
    score = matcher.calculate_confidence("7Eleven Store", "7Eleven Shop")
    assert score > 0.70  # Match via fuzzy, not phonetics


def test_phonetic_skip_acronyms():
    """Test that short acronyms skip phonetic processing."""
    matcher = NameMatcher()

    # "IBM" is a short acronym - should skip phonetics
    score = matcher.calculate_confidence("IBM", "IBM Corp")
    assert score > 0.8

    # "GE" is another short acronym
    score = matcher.calculate_confidence("GE Company", "GE Corp")
    assert score > 0.85


def test_phonetic_accent_folding():
    """Test that accented characters are handled properly."""
    matcher = NameMatcher()

    # "São Paulo" with accent should match "Sao Paulo" without
    # Accent folding helps phonetics recognize them as the same
    score = matcher.calculate_confidence("São Paulo Bank", "Sao Paulo Bank")
    assert score > 0.96  # High score with phonetic agreement

    # Test with other diacritics - "café" vs "cafe"
    # Base fuzzy matching sees them as different, but phonetics helps
    score = matcher.calculate_confidence("Café Corp", "Cafe Corp")
    assert score > 0.75  # Phonetic bonus helps bridge the gap


def test_should_use_phonetics():
    """Test the _should_use_phonetics helper method."""
    matcher = NameMatcher()

    # Valid tokens for phonetics
    assert matcher._should_use_phonetics("apple") == True
    assert matcher._should_use_phonetics("microsoft") == True

    # Invalid tokens
    assert matcher._should_use_phonetics("") == False  # Empty
    assert matcher._should_use_phonetics("a") == False  # Single char
    assert matcher._should_use_phonetics("3m") == False  # Contains number
    assert matcher._should_use_phonetics("7eleven") == False  # Contains number

    # Note: Normalized names are lowercase, so all-caps check won't trigger
    # But we can test the logic works for mixed case
    assert matcher._should_use_phonetics("ibm") == True  # Lowercase ok


def test_calculate_phonetic_bonus():
    """Test the phonetic bonus calculation directly."""
    matcher = NameMatcher()

    # Phonetically similar - should get +4
    bonus = matcher._calculate_phonetic_bonus("smith", "smyth")
    assert bonus == 4.0

    # Phonetically different - should get -2
    bonus = matcher._calculate_phonetic_bonus("apple", "orange")
    assert bonus == -2.0

    # No valid phonetic comparison (numbers) - should get 0
    bonus = matcher._calculate_phonetic_bonus("3m", "3m")
    assert bonus == 0.0
