"""
Core name matching and standardization service.
Uses fuzzy string matching to group similar company names.
"""
import re
from typing import List, Dict, Tuple, Set
from datetime import datetime
from rapidfuzz import fuzz
from metaphone import doublemetaphone
from unidecode import unidecode

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class NameMatcher:
    """
    Handles company name normalization and grouping.

    Algorithm:
    1. Normalize all names (remove punctuation, standardize case, etc.)
    2. Use fuzzy matching to identify similar names
    3. Group names into clusters based on similarity threshold
    4. Select the simplest/shortest name as canonical for each group
    """

    def __init__(self, similarity_threshold: float = None):
        """
        Initialize the name matcher.

        Args:
            similarity_threshold: Minimum similarity score (0-100) to group names
                                 Defaults to value from settings if not provided
        """
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.common_suffixes = settings.CORPORATE_SUFFIXES
        logger.info(f"Initialized NameMatcher with threshold: {self.similarity_threshold}%")

    def normalize_name(self, name: str) -> str:
        """
        Normalize a company name for comparison.

        Steps:
        - Convert to lowercase
        - Remove punctuation and extra whitespace
        - Remove common suffixes

        Args:
            name: Company name to normalize

        Returns:
            Normalized name string
        """
        if not name or not isinstance(name, str):
            return ""

        # Convert to lowercase and remove special characters
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Remove common corporate suffixes for better matching
        words = normalized.split()
        filtered_words = [w for w in words if w not in self.common_suffixes]

        return ' '.join(filtered_words) if filtered_words else normalized

    def select_canonical_name(self, names: List[str]) -> str:
        """
        Select the best canonical name from a group.

        Selection criteria (in order of priority):
        1. Shortest name (fewer characters)
        2. Fewest words
        3. Most recognizable (original capitalization preserved)

        Args:
            names: List of names to choose from

        Returns:
            Selected canonical name
        """
        if not names:
            return ""

        if len(names) == 1:
            return names[0]

        # Score each name (lower score is better)
        def score_name(name: str) -> Tuple[int, int, int]:
            return (
                len(name),  # Prefer shorter
                len(name.split()),  # Prefer fewer words
                -sum(1 for c in name if c.isupper())  # Prefer proper capitalization
            )

        canonical = min(names, key=score_name)
        logger.debug(f"Selected '{canonical}' as canonical from {len(names)} alternatives")
        return canonical

    def _should_use_phonetics(self, token: str) -> bool:
        """
        Determine if a token should be processed by metaphone.

        Skips phonetics for:
        - Empty strings
        - Single-character tokens
        - Tokens containing digits (e.g., "3M", "7eleven")
        - All-caps acronyms <= 2 characters (e.g., "IBM", "GE")

        Args:
            token: Normalized token to check

        Returns:
            True if token should use phonetics, False otherwise
        """
        if not token or len(token) <= 1:
            return False

        # Skip tokens with numbers
        if any(char.isdigit() for char in token):
            return False

        # Skip short all-caps acronyms (likely acronyms like IBM, GE)
        if len(token) <= 2 and token.isupper():
            return False

        return True

    def _prepare_for_phonetics(self, name: str) -> str:
        """
        Prepare a normalized name for phonetic comparison.

        Applies accent folding to handle non-ASCII characters.

        Args:
            name: Normalized name string

        Returns:
            Accent-folded name ready for phonetic processing
        """
        return unidecode(name)

    def _calculate_phonetic_bonus(self, norm1: str, norm2: str) -> float:
        """
        Calculate phonetic bonus/penalty for name matching.

        Uses hybrid word-by-word comparison:
        - Compares phonetic codes token-by-token
        - Skips inappropriate tokens (numbers, acronyms, etc.)
        - Returns +4 if majority agree, -2 if majority disagree, 0 if no comparison

        Args:
            norm1: First normalized name
            norm2: Second normalized name

        Returns:
            Phonetic adjustment value (+4, -2, or 0)
        """
        # Prepare names for phonetic comparison
        prep1 = self._prepare_for_phonetics(norm1)
        prep2 = self._prepare_for_phonetics(norm2)

        tokens1 = prep1.split()
        tokens2 = prep2.split()

        # If different number of tokens, do best effort word-by-word
        agreements = 0
        disagreements = 0
        comparisons = 0

        # Compare tokens pairwise
        for i in range(min(len(tokens1), len(tokens2))):
            token1 = tokens1[i]
            token2 = tokens2[i]

            # Check if both tokens should use phonetics
            if not self._should_use_phonetics(token1) or not self._should_use_phonetics(token2):
                continue

            # Get phonetic codes (primary only)
            phonetic1 = doublemetaphone(token1)[0]
            phonetic2 = doublemetaphone(token2)[0]

            # Skip if phonetic encoding failed
            if not phonetic1 or not phonetic2:
                continue

            comparisons += 1
            if phonetic1 == phonetic2:
                agreements += 1
            else:
                disagreements += 1

        # If no valid comparisons, return neutral
        if comparisons == 0:
            return 0.0

        # Determine bonus/penalty based on majority
        if agreements > disagreements:
            logger.debug(f"Phonetic agreement: {agreements}/{comparisons} matches (+4 bonus)")
            return 4.0
        else:
            logger.debug(f"Phonetic disagreement: {disagreements}/{comparisons} mismatches (-2 penalty)")
            return -2.0

    def calculate_confidence(self, name1: str, name2: str) -> float:
        """
        Calculate confidence score for a name match.

        Uses WRatio (weighted ratio) and token_set_ratio for less correlated,
        more comprehensive matching coverage. Applies phonetic matching bonus/penalty:
        - +4 percentage points if phonetics agree
        - -2 percentage points if phonetics disagree
        - Final score clamped to [0.0, 1.0]

        Args:
            name1: First company name
            name2: Second company name

        Returns:
            Confidence score between 0.0 and 1.0
        """
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)

        # Perfect match after normalization
        if norm1 == norm2:
            return 1.0

        # Use WRatio (adaptive) and token_set (word-based) for robust matching
        wratio = fuzz.WRatio(norm1, norm2)
        token_set = fuzz.token_set_ratio(norm1, norm2)

        # Weighted average: WRatio 60%, token_set 40%
        base_score = (wratio * 0.6 + token_set * 0.4)

        # Apply phonetic bonus/penalty
        phonetic_bonus = self._calculate_phonetic_bonus(norm1, norm2)
        adjusted_score = base_score + phonetic_bonus

        # Clamp to [0, 100] range
        final_score = max(0, min(100, adjusted_score))

        return final_score / 100.0

    def group_similar_names(self, names: List[str]) -> List[List[str]]:
        """
        Group similar company names together using greedy clustering.

        Algorithm:
        - Iterate through names
        - For each unprocessed name, start a new group
        - Add all similar names (above threshold) to the group
        - Mark all grouped names as processed

        Args:
            names: List of company names to group

        Returns:
            List of groups, where each group is a list of similar names
        """
        if not names:
            return []

        groups: List[List[str]] = []
        processed: Set[str] = set()

        for name in names:
            if name in processed:
                continue

            # Start a new group with this name
            current_group = [name]
            processed.add(name)

            # Find all similar names
            for other_name in names:
                if other_name in processed:
                    continue

                # Calculate similarity
                confidence = self.calculate_confidence(name, other_name)

                if confidence * 100 >= self.similarity_threshold:
                    current_group.append(other_name)
                    processed.add(other_name)

            groups.append(current_group)

        logger.info(f"Grouped {len(names)} names into {len(groups)} clusters "
                   f"({(1 - len(groups)/len(names))*100:.1f}% reduction)")
        return groups

    def process_names(self, names: List[str], filename: str = "unknown") -> Dict:
        """
        Process a list of company names and return complete results.

        This is the main entry point for the matching service.

        Args:
            names: List of company names to process
            filename: Name of the source file (for audit log)

        Returns:
            Dictionary containing:
            - mappings: List of original -> canonical mappings
            - audit_log: Detailed processing log
            - summary: Statistics about the processing
        """
        start_time = datetime.now()
        logger.info(f"Starting to process {len(names)} company names from '{filename}'")

        # Group similar names
        groups = self.group_similar_names(names)

        # Build mappings and audit log
        mappings = []
        audit_entries = []

        for group_id, group in enumerate(groups):
            # Select canonical name
            canonical = self.select_canonical_name(group)
            alternatives = [n for n in group if n != canonical]

            # Create mapping for each name in the group
            for original_name in group:
                confidence = self.calculate_confidence(original_name, canonical)

                mapping = {
                    "original_name": original_name,
                    "canonical_name": canonical,
                    "confidence_score": confidence,
                    "group_id": group_id,
                    "alternatives": alternatives if original_name == canonical else [canonical] + alternatives
                }
                mappings.append(mapping)

                # Create audit entry
                if original_name == canonical:
                    reasoning = "Selected as canonical name (shortest/simplest in group)"
                else:
                    reasoning = f"Matched to '{canonical}' with {confidence:.2%} confidence"

                audit_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "original_name": original_name,
                    "canonical_name": canonical,
                    "confidence_score": confidence,
                    "group_id": group_id,
                    "reasoning": reasoning
                }
                audit_entries.append(audit_entry)

        # Build audit log
        audit_log = {
            "filename": filename,
            "processed_at": start_time.isoformat(),
            "total_names": len(names),
            "total_groups": len(groups),
            "entries": audit_entries
        }

        # Build summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = {
            "total_input_names": len(names),
            "total_groups_created": len(groups),
            "reduction_percentage": (1 - len(groups) / len(names)) * 100 if names else 0,
            "average_group_size": len(names) / len(groups) if groups else 0,
            "processing_time_seconds": processing_time
        }

        logger.info(f"Processing complete: {len(groups)} groups created in {processing_time:.2f}s")

        return {
            "mappings": mappings,
            "audit_log": audit_log,
            "summary": summary
        }
