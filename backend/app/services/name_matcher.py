"""
Core name matching and standardization service.
Uses fuzzy string matching to group similar company names.
"""
import re
from typing import List, Dict, Tuple, Set
from datetime import datetime
from rapidfuzz import fuzz

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

    def calculate_confidence(self, name1: str, name2: str) -> float:
        """
        Calculate confidence score for a name match.

        Uses multiple fuzzy matching algorithms and combines them with
        a weighted average for more robust matching.

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

        # Use multiple matching strategies for robustness
        ratio = fuzz.ratio(norm1, norm2)
        token_sort = fuzz.token_sort_ratio(norm1, norm2)
        token_set = fuzz.token_set_ratio(norm1, norm2)

        # Weighted average (token_set is most forgiving of word order)
        score = (ratio * 0.3 + token_sort * 0.3 + token_set * 0.4)

        return score / 100.0

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
