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
