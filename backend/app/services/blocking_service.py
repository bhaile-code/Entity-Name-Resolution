"""
Blocking and stratified sampling service for entity resolution.

This module provides blocking key generation and stratified reservoir sampling
to efficiently sample pairwise comparisons for GMM threshold calibration.

Key Features:
- Hybrid token+phonetic blocking keys (first token + Metaphone)
- Stratified reservoir sampling (80/20 proportional+floor allocation)
- Cross-block sampling (5% for rare matches)
- Graceful fallbacks for edge cases
- Reproducible sampling with fixed RNG seed

Example:
    >>> key_gen = BlockingKeyGenerator()
    >>> key_gen.generate_key("Apple Inc.")
    'apple_APL'

    >>> sampler = StratifiedReservoirSampler(max_pairs=50000)
    >>> result = sampler.sample_pairs(names, blocking_keys)
    >>> len(result['pairs'])
    50000
"""

import re
import random
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from metaphone import doublemetaphone
from unidecode import unidecode

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class BlockingKeyGenerator:
    """
    Generates blocking keys using hybrid token+phonetic approach.

    Blocking keys group potentially similar names together for efficient
    comparison. The hybrid approach combines:
    - First significant token (after stopword filtering)
    - Metaphone phonetic code (for spelling variants)

    Examples:
        >>> gen = BlockingKeyGenerator()
        >>> gen.generate_key("Apple Inc.")
        'apple_APL'
        >>> gen.generate_key("Microsoft Corporation")
        'microsoft_MKRS'
        >>> gen.generate_key("3M Company")  # No phonetic for digits
        '3m'
        >>> gen.generate_key("The A Team")  # Stopword filtered
        'team_TM'

    Attributes:
        STOP_WORDS: Minimal set of stopwords to filter {the, a, an}
    """

    STOP_WORDS = frozenset({'the', 'a', 'an'})

    def __init__(self):
        """Initialize the blocking key generator."""
        self._phonetic_skip_counter = Counter()

    def _normalize_for_blocking(self, name: str) -> str:
        """
        Normalize a name for blocking key generation.

        Steps:
        - Convert to lowercase
        - Apply accent folding (São → Sao)
        - Remove punctuation
        - Normalize whitespace

        Args:
            name: Company name to normalize

        Returns:
            Normalized name string

        Examples:
            >>> gen._normalize_for_blocking("Apple, Inc.")
            'apple inc'
            >>> gen._normalize_for_blocking("São Paulo Tech")
            'sao paulo tech'
        """
        if not name or not isinstance(name, str):
            return ""

        # Apply accent folding first
        normalized = unidecode(name)

        # Convert to lowercase
        normalized = normalized.lower().strip()

        # Remove punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _extract_first_token(self, name: str) -> str:
        """
        Extract first significant token after stopword filtering.

        Rules:
        - Normalize name first
        - Split into tokens
        - Skip stopwords: {the, a, an}
        - Return first non-stopword token
        - If all tokens are stopwords → return empty string (fallback to full name)

        Args:
            name: Company name to extract token from

        Returns:
            First significant token or empty string

        Examples:
            >>> gen._extract_first_token("The Apple Company")
            'apple'
            >>> gen._extract_first_token("A Better Solution")
            'better'
            >>> gen._extract_first_token("Microsoft")
            'microsoft'
            >>> gen._extract_first_token("The A Team")
            'team'
            >>> gen._extract_first_token("A & A")
            ''  # No valid tokens, will fallback
        """
        normalized = self._normalize_for_blocking(name)
        tokens = normalized.split()

        for token in tokens:
            if token not in self.STOP_WORDS and len(token) > 0:
                return token

        return ""  # No valid tokens found

    def _should_use_phonetics(self, token: str) -> bool:
        """
        Determine if a token should be processed by Metaphone.

        Skips phonetics for:
        - Empty strings
        - Single-character tokens
        - Tokens containing digits (e.g., "3m", "7eleven")
        - Short all-caps acronyms (e.g., "ibm", "ge" after normalization)

        Args:
            token: Normalized token to check

        Returns:
            True if token should use phonetics, False otherwise

        Examples:
            >>> gen._should_use_phonetics("apple")
            True
            >>> gen._should_use_phonetics("3m")
            False
            >>> gen._should_use_phonetics("ge")
            False
            >>> gen._should_use_phonetics("a")
            False
        """
        if not token or len(token) <= 1:
            return False

        # Skip tokens with numbers
        if any(char.isdigit() for char in token):
            return False

        # Skip very short tokens (likely acronyms after normalization)
        if len(token) <= 2:
            return False

        return True

    def _generate_phonetic_code(self, token: str) -> Optional[str]:
        """
        Generate phonetic code with graceful fallback.

        Uses Double Metaphone algorithm to generate phonetic representation.
        Tracks failures for systematic gap detection.

        Returns None if:
        - Token has digits (e.g., "3m", "7eleven")
        - Token is very short acronym (len <= 2)
        - doublemetaphone returns empty/None

        Args:
            token: Normalized token to encode

        Returns:
            Primary Metaphone code or None if unavailable

        Examples:
            >>> gen._generate_phonetic_code("apple")
            'APL'
            >>> gen._generate_phonetic_code("microsoft")
            'MKRS'
            >>> gen._generate_phonetic_code("3m")
            None  # Has digits
            >>> gen._generate_phonetic_code("ge")
            None  # Too short
        """
        # Check if token should use phonetics
        if not self._should_use_phonetics(token):
            self._phonetic_skip_counter['non_phonetic_token'] += 1
            return None

        try:
            # Attempt phonetic encoding (primary code only)
            phonetic = doublemetaphone(token)[0]

            if not phonetic:
                self._phonetic_skip_counter['no_usable_code'] += 1
                return None

            return phonetic
        except Exception as e:
            # Catch any unexpected errors from doublemetaphone
            logger.debug(f"Phonetic encoding failed for '{token}': {e}")
            self._phonetic_skip_counter['encoding_error'] += 1
            return None

    def generate_key(self, name: str) -> str:
        """
        Generate hybrid token+phonetic blocking key with fallback.

        Format: "{token}_{phonetic}" or "{token}" if phonetic unavailable

        Fallback strategy:
        1. Extract first significant token
        2. If no token → use full normalized name
        3. Generate phonetic code for token
        4. If phonetic available → "{token}_{phonetic}"
        5. If no phonetic → "{token}"

        Args:
            name: Company name to generate key for

        Returns:
            Blocking key string

        Examples:
            >>> gen.generate_key("Apple Inc.")
            'apple_APL'
            >>> gen.generate_key("3M Corporation")
            '3m'  # No phonetic, digit token
            >>> gen.generate_key("A B Testing")
            'testing_TSTN'  # Stopwords filtered
            >>> gen.generate_key("The A Team")
            'team_TM'
            >>> gen.generate_key("A & A")
            'a a'  # Fallback to full normalized (no valid tokens)
        """
        # Extract first significant token
        token = self._extract_first_token(name)

        if not token:
            # Edge case: no tokens after stopword filtering
            # Fallback to full normalized name
            normalized = self._normalize_for_blocking(name)
            logger.debug(f"No valid token for '{name}', using full normalized: '{normalized}'")
            return normalized

        # Attempt phonetic encoding
        phonetic = self._generate_phonetic_code(token)

        # Combine or fallback to token only
        if phonetic:
            return f"{token}_{phonetic}"
        else:
            return token

    def get_phonetic_stats(self) -> Dict[str, int]:
        """
        Get phonetic encoding statistics for systematic gap detection.

        Returns:
            Dictionary with skip counters:
            - non_phonetic_token: Tokens with digits/too short
            - no_usable_code: Metaphone returned empty
            - encoding_error: Unexpected errors

        Example:
            >>> gen.generate_key("Apple Inc.")
            >>> gen.generate_key("3M Company")
            >>> gen.get_phonetic_stats()
            {'non_phonetic_token': 1, 'no_usable_code': 0, 'encoding_error': 0}
        """
        return dict(self._phonetic_skip_counter)


class StratifiedReservoirSampler:
    """
    Implements stratified reservoir sampling for pairwise comparisons.

    Samples pairs using three strategies:
    1. Within-block sampling (95%): Groups potentially similar names
    2. Cross-block sampling (5%): Captures rare cross-cluster matches
    3. Proportional+floor allocation: Balances block coverage

    Configuration:
    - max_pairs: Total sampling budget (default: 50,000)
    - within_block_pct: % for within-block (default: 0.95)
    - cross_block_pct: % for cross-block (default: 0.05)
    - proportional_pct: % of within-block allocated proportionally (default: 0.80)
    - floor_pct: % of within-block allocated evenly (default: 0.20)
    - min_block_size: Minimum block size, skip singletons (default: 2)
    - max_block_pairs: Cap pairs per block (default: 5,000)
    - rng_seed: Random seed for reproducibility (default: 42)

    Example:
        >>> sampler = StratifiedReservoirSampler(max_pairs=50000, rng_seed=42)
        >>> blocking_keys = {name: key_gen.generate_key(name) for name in names}
        >>> result = sampler.sample_pairs(names, blocking_keys)
        >>> len(result['pairs'])
        50000
        >>> result['metadata']['total_blocks']
        90
    """

    def __init__(
        self,
        max_pairs: int = None,
        within_block_pct: float = None,
        cross_block_pct: float = None,
        proportional_pct: float = None,
        floor_pct: float = None,
        min_block_size: int = None,
        max_block_pairs: int = None,
        rng_seed: int = None
    ):
        """
        Initialize the stratified reservoir sampler.

        Args:
            max_pairs: Total pairs to sample (default from settings.GMM_MAX_PAIRS)
            within_block_pct: % within-block (default from settings.SAMPLING_WITHIN_BLOCK_PCT)
            cross_block_pct: % cross-block (default from settings.SAMPLING_CROSS_BLOCK_PCT)
            proportional_pct: % proportional allocation (default from settings.SAMPLING_PROPORTIONAL_PCT)
            floor_pct: % floor allocation (default from settings.SAMPLING_FLOOR_PCT)
            min_block_size: Min block size (default from settings.BLOCKING_MIN_BLOCK_SIZE)
            max_block_pairs: Max pairs per block (default from settings.BLOCKING_MAX_BLOCK_PAIRS)
            rng_seed: Random seed (default from settings.SAMPLING_RNG_SEED)
        """
        # Load configuration from settings with override support and defaults
        # Use getattr with defaults for graceful fallback if settings not yet added
        self.max_pairs = max_pairs if max_pairs is not None else getattr(settings, 'GMM_MAX_PAIRS', 50000)
        self.within_block_pct = within_block_pct if within_block_pct is not None else getattr(settings, 'SAMPLING_WITHIN_BLOCK_PCT', 0.95)
        self.cross_block_pct = cross_block_pct if cross_block_pct is not None else getattr(settings, 'SAMPLING_CROSS_BLOCK_PCT', 0.05)
        self.proportional_pct = proportional_pct if proportional_pct is not None else getattr(settings, 'SAMPLING_PROPORTIONAL_PCT', 0.80)
        self.floor_pct = floor_pct if floor_pct is not None else getattr(settings, 'SAMPLING_FLOOR_PCT', 0.20)
        self.min_block_size = min_block_size if min_block_size is not None else getattr(settings, 'BLOCKING_MIN_BLOCK_SIZE', 2)
        self.max_block_pairs = max_block_pairs if max_block_pairs is not None else getattr(settings, 'BLOCKING_MAX_BLOCK_PAIRS', 5000)
        self.rng_seed = rng_seed if rng_seed is not None else getattr(settings, 'SAMPLING_RNG_SEED', 42)

        # Initialize RNG with fixed seed for reproducibility
        self.rng = random.Random(self.rng_seed)

        logger.info(f"Initialized StratifiedReservoirSampler with seed={self.rng_seed}, max_pairs={self.max_pairs}")

    def _create_blocks(
        self,
        names: List[str],
        blocking_keys: Dict[str, str]
    ) -> Tuple[Dict[str, List[str]], int]:
        """
        Group names into blocks by blocking key and filter singletons.

        Args:
            names: List of company names
            blocking_keys: Dict mapping name → blocking key

        Returns:
            Tuple of (blocks dict, num singletons filtered)
            - blocks: Dict mapping blocking_key → list of names
            - singletons: Count of singleton blocks filtered

        Example:
            >>> blocks, singletons = sampler._create_blocks(
            ...     ["Apple Inc.", "Apple Computer", "Microsoft"],
            ...     {"Apple Inc.": "apple_APL", "Apple Computer": "apple_APL", "Microsoft": "microsoft_MKRS"}
            ... )
            >>> blocks
            {'apple_APL': ['Apple Inc.', 'Apple Computer']}
            >>> singletons
            1  # Microsoft was a singleton
        """
        # Group names by blocking key
        raw_blocks = defaultdict(list)
        for name in names:
            key = blocking_keys.get(name, "")
            if key:  # Skip names without valid keys
                raw_blocks[key].append(name)

        # Filter out singletons (blocks with size < min_block_size)
        blocks = {}
        singletons = 0

        for key, block_names in raw_blocks.items():
            if len(block_names) >= self.min_block_size:
                blocks[key] = block_names
            else:
                singletons += len(block_names)

        logger.debug(f"Created {len(blocks)} blocks from {len(names)} names, filtered {singletons} singletons")

        return blocks, singletons

    def _calculate_block_stats(self, block: List[str]) -> Dict[str, int]:
        """
        Calculate statistics for a single block.

        Args:
            block: List of names in the block

        Returns:
            Dict with:
            - total_pairs: Total possible pairs in block (capped)
            - capped: Whether block hit max_block_pairs cap

        Example:
            >>> stats = sampler._calculate_block_stats(["Apple", "Apple Inc.", "Apple Computer"])
            >>> stats
            {'total_pairs': 3, 'capped': False}
        """
        n = len(block)
        total_pairs = n * (n - 1) // 2
        capped = total_pairs > self.max_block_pairs

        if capped:
            total_pairs = self.max_block_pairs

        return {
            'total_pairs': total_pairs,
            'capped': capped
        }

    def _allocate_budget(
        self,
        blocks: Dict[str, List[str]],
        total_budget: int
    ) -> Dict[str, int]:
        """
        Allocate sampling budget across blocks using proportional+floor strategy.

        Strategy:
        - 80% proportional to block pairs (larger blocks get more samples)
        - 20% evenly distributed (ensures small blocks get represented)

        Args:
            blocks: Dict mapping blocking_key → list of names
            total_budget: Total pairs to allocate across blocks

        Returns:
            Dict mapping blocking_key → num_pairs_to_sample

        Example:
            >>> allocation = sampler._allocate_budget(
            ...     {'apple_APL': ['Apple Inc.', 'Apple Computer', 'Apple'],  # 3 pairs
            ...      'microsoft_MKRS': ['Microsoft', 'Microsoft Corp']},  # 1 pair
            ...     100
            ... )
            >>> allocation
            {'apple_APL': 70, 'microsoft_MKRS': 30}  # Proportional + floor
        """
        if not blocks:
            return {}

        # Calculate total pairs available (capped per block)
        block_pairs = {}
        total_available_pairs = 0

        for key, block in blocks.items():
            stats = self._calculate_block_stats(block)
            block_pairs[key] = stats['total_pairs']
            total_available_pairs += stats['total_pairs']

        # If total available is less than budget, use all available
        if total_available_pairs <= total_budget:
            logger.debug(f"Total available pairs ({total_available_pairs}) <= budget ({total_budget}), using all")
            return block_pairs

        # Allocate budget: proportional + floor
        proportional_budget = int(total_budget * self.proportional_pct)
        floor_budget = int(total_budget * self.floor_pct)

        allocation = {}
        num_blocks = len(blocks)
        floor_per_block = floor_budget // num_blocks if num_blocks > 0 else 0

        # Proportional allocation
        for key, pairs_available in block_pairs.items():
            proportion = pairs_available / total_available_pairs if total_available_pairs > 0 else 0
            proportional_alloc = int(proportional_budget * proportion)
            allocation[key] = proportional_alloc + floor_per_block

        # Distribute any remainder due to rounding
        total_allocated = sum(allocation.values())
        remainder = total_budget - total_allocated

        if remainder > 0:
            # Give remainder to largest blocks
            sorted_keys = sorted(block_pairs.keys(), key=lambda k: block_pairs[k], reverse=True)
            for i in range(remainder):
                key = sorted_keys[i % len(sorted_keys)]
                allocation[key] += 1

        # Ensure no block gets more than available
        for key in allocation:
            allocation[key] = min(allocation[key], block_pairs[key])

        logger.debug(f"Allocated {sum(allocation.values())} pairs across {len(allocation)} blocks")

        return allocation

    def _reservoir_sample_within_block(
        self,
        block: List[str],
        sample_size: int
    ) -> Tuple[List[Tuple[str, str]], Dict[str, any]]:
        """
        Reservoir sample pairs within a single block using Algorithm R.

        On-the-fly implementation that never materializes full pair list.
        Ensures every pair has equal probability even when total_pairs > sample_size.

        Args:
            block: List of names in the block
            sample_size: Number of pairs to sample

        Returns:
            Tuple of (sampled_pairs, stats)
            - sampled_pairs: List of (name1, name2) tuples
            - stats: Dict with 'total_pairs', 'sampled_pairs', 'capped'

        Example:
            >>> pairs, stats = sampler._reservoir_sample_within_block(
            ...     ["Apple", "Apple Inc.", "Apple Computer"],
            ...     2
            ... )
            >>> len(pairs)
            2
            >>> stats
            {'total_pairs': 3, 'sampled_pairs': 2, 'capped': False}
        """
        n = len(block)
        total_pairs = n * (n - 1) // 2

        # Check if block should be capped
        capped = total_pairs > self.max_block_pairs
        if capped:
            effective_total_pairs = self.max_block_pairs
        else:
            effective_total_pairs = total_pairs

        # Determine actual sample size
        actual_sample_size = min(sample_size, effective_total_pairs)

        # Reservoir sampling using Algorithm R (on-the-fly)
        reservoir = []
        pair_idx = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair = (block[i], block[j])

                if len(reservoir) < actual_sample_size:
                    # Fill reservoir
                    reservoir.append(pair)
                else:
                    # Randomly replace with decreasing probability
                    replace_idx = self.rng.randint(0, pair_idx)
                    if replace_idx < actual_sample_size:
                        reservoir[replace_idx] = pair

                pair_idx += 1

                # Early termination if we've processed max_block_pairs
                if capped and pair_idx >= self.max_block_pairs:
                    break

            # Break outer loop if capped
            if capped and pair_idx >= self.max_block_pairs:
                break

        stats = {
            'total_pairs': effective_total_pairs,
            'sampled_pairs': len(reservoir),
            'capped': capped
        }

        return reservoir, stats

    def _reservoir_sample_from_list(
        self,
        pairs: List[Tuple[str, str]],
        sample_size: int
    ) -> List[Tuple[str, str]]:
        """
        Reservoir sample from a list of pairs (helper for cross-block).

        Args:
            pairs: List of (name1, name2) tuples
            sample_size: Number of pairs to sample

        Returns:
            Sampled pairs (up to sample_size)

        Example:
            >>> pairs = [("Apple", "Microsoft"), ("Google", "Amazon"), ("Tesla", "Meta")]
            >>> sampled = sampler._reservoir_sample_from_list(pairs, 2)
            >>> len(sampled)
            2
        """
        if len(pairs) <= sample_size:
            return pairs

        return self.rng.sample(pairs, sample_size)

    def _uniform_sample_cross_block(
        self,
        blocks: Dict[str, List[str]],
        sample_size: int
    ) -> List[Tuple[str, str]]:
        """
        Sample uniform random pairs from DIFFERENT blocks (two-stage sampling).

        Strategy:
        1. Randomly pick block pairs (k1, k2) where k1 != k2
        2. For each block pair, generate candidate name pairs
        3. Reservoir sample final pairs from candidates

        This approach balances memory efficiency with correctness, avoiding
        materialization of millions of cross-block pairs.

        Args:
            blocks: Dict mapping blocking_key → list of names
            sample_size: Number of cross-block pairs to sample

        Returns:
            List of (name1, name2) tuples from different blocks

        Example:
            >>> cross_pairs = sampler._uniform_sample_cross_block(
            ...     {'apple_APL': ['Apple Inc.'], 'microsoft_MKRS': ['Microsoft']},
            ...     10
            ... )
            >>> len(cross_pairs)
            1  # Only 1 possible cross-block pair
        """
        if len(blocks) < 2:
            logger.debug("Less than 2 blocks, no cross-block pairs possible")
            return []

        # Stage 1: Generate all possible block pairs
        block_keys = list(blocks.keys())
        block_pairs = []
        for i, k1 in enumerate(block_keys):
            for k2 in block_keys[i+1:]:
                block_pairs.append((k1, k2))

        # If we need more pairs than block combinations allow, sample all block pairs
        # Otherwise, randomly sample block pairs
        if len(block_pairs) * 10 < sample_size:  # Heuristic: if we need many pairs per block pair
            sampled_block_pairs = block_pairs
        else:
            num_block_pairs = min(len(block_pairs), max(1, sample_size // 10))
            sampled_block_pairs = self.rng.sample(block_pairs, num_block_pairs)

        # Stage 2: Generate candidate name pairs from sampled block pairs
        candidate_pairs = []
        for k1, k2 in sampled_block_pairs:
            for n1 in blocks[k1]:
                for n2 in blocks[k2]:
                    candidate_pairs.append((n1, n2))
                    # Early termination if we have way more than needed
                    if len(candidate_pairs) >= sample_size * 10:
                        break
                if len(candidate_pairs) >= sample_size * 10:
                    break

        # Stage 3: Reservoir sample final pairs
        sampled = self._reservoir_sample_from_list(candidate_pairs, sample_size)

        logger.debug(f"Cross-block sampling: {len(sampled)} pairs from {len(candidate_pairs)} candidates")

        return sampled

    def sample_pairs(
        self,
        names: List[str],
        blocking_keys: Dict[str, str]
    ) -> Dict:
        """
        Main orchestrator for stratified reservoir sampling.

        Steps:
        1. Create blocks from blocking keys
        2. Filter singletons
        3. Calculate block statistics
        4. Allocate within-block budget (80% proportional + 20% floor)
        5. Sample within-block pairs (Algorithm R per block)
        6. Sample cross-block pairs (two-stage sampling)
        7. Shuffle and combine results
        8. Log comprehensive statistics

        Args:
            names: List of company names
            blocking_keys: Dict mapping name → blocking key

        Returns:
            Dict with:
            - pairs: List of (name1, name2) tuples
            - metadata: Dict with statistics and timings

        Example:
            >>> result = sampler.sample_pairs(names, blocking_keys)
            >>> len(result['pairs'])
            50000
            >>> result['metadata']['total_blocks']
            90
            >>> result['metadata']['within_block_pairs']
            47500
            >>> result['metadata']['cross_block_pairs']
            2500
        """
        start_time = time.time()

        logger.info(f"Starting stratified reservoir sampling for {len(names)} names")
        logger.info(f"Configuration: max_pairs={self.max_pairs}, within={self.within_block_pct:.0%}, "
                   f"cross={self.cross_block_pct:.0%}, prop={self.proportional_pct:.0%}, "
                   f"floor={self.floor_pct:.0%}, seed={self.rng_seed}")

        # Step 1: Create blocks and filter singletons
        blocks, singletons_filtered = self._create_blocks(names, blocking_keys)

        if not blocks:
            logger.warning("No valid blocks created (all singletons?), returning empty sample")
            return {
                'pairs': [],
                'metadata': {
                    'total_names': len(names),
                    'total_blocks': 0,
                    'singletons_filtered': singletons_filtered,
                    'largest_block_size': 0,
                    'largest_block_pairs': 0,
                    'blocks_capped': 0,
                    'within_block_pairs': 0,
                    'cross_block_pairs': 0,
                    'elapsed_time': time.time() - start_time
                }
            }

        # Step 2: Calculate block statistics
        block_stats = {}
        for key, block in blocks.items():
            block_stats[key] = self._calculate_block_stats(block)

        largest_block_key = max(blocks.keys(), key=lambda k: len(blocks[k]))
        largest_block_size = len(blocks[largest_block_key])
        largest_block_pairs = block_stats[largest_block_key]['total_pairs']
        blocks_capped = sum(1 for stats in block_stats.values() if stats['capped'])

        logger.info(f"Block statistics: {len(blocks)} blocks, avg_size={sum(len(b) for b in blocks.values())/len(blocks):.1f}, "
                   f"largest={largest_block_size} names ({largest_block_pairs} pairs), "
                   f"capped={blocks_capped}, singletons={singletons_filtered}")

        # Step 3: Allocate budgets
        within_block_budget = int(self.max_pairs * self.within_block_pct)
        cross_block_budget = int(self.max_pairs * self.cross_block_pct)

        allocation = self._allocate_budget(blocks, within_block_budget)

        # Step 4: Sample within-block pairs with progress logging
        within_block_pairs = []
        num_blocks = len(blocks)
        progress_interval = max(1, num_blocks // 10)  # Log every 10%

        logger.info(f"Sampling within-block pairs (budget: {within_block_budget})")

        for idx, (key, block) in enumerate(blocks.items()):
            sample_size = allocation.get(key, 0)
            if sample_size > 0:
                pairs, stats = self._reservoir_sample_within_block(block, sample_size)
                within_block_pairs.extend(pairs)

                # DEBUG logging for per-block details
                logger.debug(f"Block '{key}': size={len(block)}, total_pairs={stats['total_pairs']}, "
                           f"sampled={stats['sampled_pairs']}, capped={stats['capped']}")

            # Progress logging
            if (idx + 1) % progress_interval == 0 or (idx + 1) == num_blocks:
                progress_pct = ((idx + 1) / num_blocks) * 100
                logger.info(f"Sampling progress: {progress_pct:.0f}% ({idx + 1}/{num_blocks} blocks)")

        # Step 5: Sample cross-block pairs
        logger.info(f"Sampling cross-block pairs (budget: {cross_block_budget})")
        cross_block_pairs = self._uniform_sample_cross_block(blocks, cross_block_budget)

        # Step 6: Combine and shuffle
        all_pairs = within_block_pairs + cross_block_pairs
        self.rng.shuffle(all_pairs)  # Shuffle to mix within/cross-block pairs

        # Trim to exact max_pairs if we went over due to rounding
        all_pairs = all_pairs[:self.max_pairs]

        elapsed_time = time.time() - start_time

        # Step 7: Log final statistics
        logger.info(f"Sampling complete: {len(all_pairs)} total pairs "
                   f"({len(within_block_pairs)} within-block, {len(cross_block_pairs)} cross-block) "
                   f"in {elapsed_time:.2f}s")

        metadata = {
            'total_names': len(names),
            'total_blocks': len(blocks),
            'singletons_filtered': singletons_filtered,
            'largest_block_size': largest_block_size,
            'largest_block_pairs': largest_block_pairs,
            'blocks_capped': blocks_capped,
            'within_block_pairs': len(within_block_pairs),
            'cross_block_pairs': len(cross_block_pairs),
            'elapsed_time': elapsed_time
        }

        return {
            'pairs': all_pairs,
            'metadata': metadata
        }
