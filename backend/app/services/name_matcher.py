"""
Core name matching and standardization service.
Uses fuzzy string matching to group similar company names.
"""
import re
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime
from rapidfuzz import fuzz
from metaphone import doublemetaphone
from unidecode import unidecode

from app.config import settings
from app.utils.logger import setup_logger
from app.services.gmm_threshold_service import GMMThresholdCalculator
from app.services.blocking_service import BlockingKeyGenerator, StratifiedReservoirSampler
from app.services.embedding_service import create_embedding_service
from app.services.hac_clustering_service import HACClusteringService
import numpy as np

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

    def __init__(
        self,
        similarity_threshold: float = None,
        use_adaptive_threshold: bool = False,
        embedding_mode: str = None,
        clustering_mode: str = None,
        hac_threshold: float = None,
        hac_linkage: str = None
    ):
        """
        Initialize the name matcher.

        Args:
            similarity_threshold: Minimum similarity score (0-100) to group names (for fixed mode)
                                 Defaults to value from settings if not provided
            use_adaptive_threshold: If True, use GMM-based adaptive thresholding (deprecated, use clustering_mode)
            embedding_mode: Embedding mode ('openai-small', 'openai-large', 'local', 'disabled')
                           Defaults to settings.DEFAULT_EMBEDDING_MODE if not provided
            clustering_mode: 'fixed', 'adaptive_gmm', or 'hac'. If not provided, infers from use_adaptive_threshold
            hac_threshold: HAC distance threshold (0-1 range). Defaults to settings.HAC_DISTANCE_THRESHOLD
            hac_linkage: HAC linkage method. Defaults to settings.HAC_LINKAGE_METHOD
        """
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.common_suffixes = settings.CORPORATE_SUFFIXES

        # Determine clustering mode
        if clustering_mode is not None:
            self.clustering_mode = clustering_mode
        elif use_adaptive_threshold:
            self.clustering_mode = 'adaptive_gmm'
        else:
            self.clustering_mode = settings.CLUSTERING_MODE

        # Backward compatibility
        self.use_adaptive_threshold = (self.clustering_mode == 'adaptive_gmm')

        # Initialize mode-specific services
        self.gmm_calculator = None
        self.hac_service = None

        if self.clustering_mode == 'adaptive_gmm':
            self.gmm_calculator = GMMThresholdCalculator(min_samples=settings.GMM_MIN_SAMPLES)
        elif self.clustering_mode == 'hac':
            hac_threshold = hac_threshold if hac_threshold is not None else settings.HAC_DISTANCE_THRESHOLD
            hac_linkage = hac_linkage or settings.HAC_LINKAGE_METHOD
            self.hac_service = HACClusteringService(threshold=hac_threshold, linkage_method=hac_linkage)

        # Initialize embedding service
        self.embedding_mode = embedding_mode or settings.DEFAULT_EMBEDDING_MODE
        self.embedding_service = create_embedding_service(self.embedding_mode)

        mode_desc = {
            'fixed': f"fixed threshold: {self.similarity_threshold}%",
            'adaptive_gmm': "adaptive GMM",
            'hac': f"HAC (threshold={self.hac_service.threshold:.3f}, linkage={self.hac_service.linkage_method})"
        }.get(self.clustering_mode, self.clustering_mode)

        embed_status = f"embeddings: {self.embedding_mode}" if self.embedding_service else "embeddings: disabled"
        logger.info(f"Initialized NameMatcher with {mode_desc}, {embed_status}")

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

        Hybrid approach combining:
        - WRatio (weighted ratio): 40% - handles typos and variations
        - token_set_ratio: 15% - handles word order (reduced from 40% to fix shared-word problem)
        - Semantic embeddings: 45% - understands context and meaning (NEW)
        - Phonetic matching: ±2-4 points bonus/penalty

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

        # Component 1: WRatio (handles typos and character-level differences)
        wratio = fuzz.WRatio(norm1, norm2)

        # Component 2: token_set (handles word order, but reduced weight)
        token_set = fuzz.token_set_ratio(norm1, norm2)

        # Component 3: Semantic embeddings (NEW - understands meaning)
        embedding_score = 0.0
        if self.embedding_service:
            try:
                # Get semantic similarity (returns 0.0-1.0)
                embedding_sim = self.embedding_service.similarity(norm1, norm2)
                embedding_score = embedding_sim * 100  # Convert to 0-100 scale
            except Exception as e:
                logger.warning(f"Embedding similarity failed, using fuzzy only: {e}")
                # Fallback: redistribute weights if embeddings fail
                wratio_weight = settings.WRATIO_WEIGHT / (settings.WRATIO_WEIGHT + settings.TOKEN_SET_WEIGHT)
                token_set_weight = settings.TOKEN_SET_WEIGHT / (settings.WRATIO_WEIGHT + settings.TOKEN_SET_WEIGHT)
                base_score = (wratio * wratio_weight + token_set * token_set_weight) * 100
                phonetic_bonus = self._calculate_phonetic_bonus(norm1, norm2)
                final_score = max(0, min(100, base_score + phonetic_bonus))
                return final_score / 100.0

        # Weighted combination using configured weights
        base_score = (
            wratio * settings.WRATIO_WEIGHT +
            token_set * settings.TOKEN_SET_WEIGHT +
            embedding_score * settings.EMBEDDING_WEIGHT
        )

        # Apply phonetic bonus/penalty
        phonetic_bonus = self._calculate_phonetic_bonus(norm1, norm2)
        adjusted_score = base_score + phonetic_bonus

        # Clamp to [0, 100] range
        final_score = max(0, min(100, adjusted_score))

        return final_score / 100.0

    def _collect_pairwise_scores(self, names: List[str], max_pairs: int = None) -> Tuple[List[Tuple[str, str, float, bool]], Dict]:
        """
        Collect similarity scores using stratified reservoir sampling.

        Uses hybrid token+phonetic blocking to create strata, then samples:
        - 95% within-block pairs (potentially similar names)
        - 5% cross-block pairs (rare matches)
        - 80% proportional allocation + 20% floor per block

        This approach provides better representation than sequential sampling,
        ensuring all names get proportional coverage regardless of position.

        Args:
            names: List of company names
            max_pairs: Maximum number of pairs to collect (None = use settings.GMM_MAX_PAIRS)

        Returns:
            Tuple of (pairs, metadata):
            - pairs: List of tuples (name1, name2, composite_score, phonetic_agree)
            - metadata: Dict with blocking/sampling statistics
        """
        # Use max_pairs from settings if not provided
        if max_pairs is None:
            max_pairs = settings.GMM_MAX_PAIRS

        logger.info(f"Using stratified reservoir sampling for {len(names)} names (max {max_pairs} pairs)")

        # Step 1: Generate blocking keys
        key_generator = BlockingKeyGenerator()
        blocking_keys = {}

        for name in names:
            blocking_keys[name] = key_generator.generate_key(name)

        logger.info(f"Generated {len(set(blocking_keys.values()))} unique blocking keys")

        # Step 2: Sample pairs using stratified reservoir sampling
        sampler = StratifiedReservoirSampler(max_pairs=max_pairs)
        sampling_result = sampler.sample_pairs(names, blocking_keys)

        sampled_pairs = sampling_result['pairs']
        sampling_metadata = sampling_result['metadata']

        logger.info(f"Sampled {len(sampled_pairs)} pairs using stratified approach "
                   f"({sampling_metadata['within_block_pairs']} within-block, "
                   f"{sampling_metadata['cross_block_pairs']} cross-block)")

        # Step 3: Calculate similarity scores for sampled pairs
        logger.info(f"Calculating similarity scores for {len(sampled_pairs)} sampled pairs")

        scored_pairs = []
        score_log_interval = max(1000, len(sampled_pairs) // 10)  # Log every 10%

        for idx, (name1, name2) in enumerate(sampled_pairs):
            # Calculate base composite score
            score = self.calculate_confidence(name1, name2)

            # Check if phonetics agree
            norm1 = self.normalize_name(name1)
            norm2 = self.normalize_name(name2)
            phonetic_bonus = self._calculate_phonetic_bonus(norm1, norm2)
            phonetic_agree = phonetic_bonus > 0  # True if bonus is +4

            scored_pairs.append((name1, name2, score, phonetic_agree))

            # Progress logging
            if (idx + 1) % score_log_interval == 0:
                progress_pct = ((idx + 1) / len(sampled_pairs)) * 100
                logger.info(f"Scoring progress: {idx + 1:,}/{len(sampled_pairs):,} pairs ({progress_pct:.0f}%)")

        # Step 4: Add phonetic stats to metadata
        phonetic_stats = key_generator.get_phonetic_stats()
        sampling_metadata['phonetic_stats'] = phonetic_stats

        logger.info(f"Completed scoring {len(scored_pairs)} pairs")
        if phonetic_stats:
            logger.info(f"Phonetic encoding stats: {phonetic_stats}")

        return scored_pairs, sampling_metadata

    def _build_similarity_matrix(self, names: List[str]) -> np.ndarray:
        """
        Build a full pairwise similarity matrix for HAC clustering.

        Args:
            names: List of company names

        Returns:
            Square numpy array of pairwise similarities (n x n)
        """
        n = len(names)
        similarity_matrix = np.zeros((n, n))

        # Diagonal is 1.0 (name is identical to itself)
        np.fill_diagonal(similarity_matrix, 1.0)

        logger.info(f"Building {n}x{n} similarity matrix for HAC ({n*(n-1)//2} pairs)")

        # Calculate all pairwise similarities
        total_pairs = n * (n - 1) // 2
        pairs_calculated = 0
        progress_interval = max(1000, total_pairs // 10)  # Log every 10% or 1000 pairs

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate similarity
                similarity = self.calculate_confidence(names[i], names[j])

                # Fill both triangles (symmetric matrix)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

                pairs_calculated += 1
                if pairs_calculated % progress_interval == 0:
                    pct = (pairs_calculated / total_pairs) * 100
                    logger.info(f"Similarity matrix progress: {pairs_calculated:,}/{total_pairs:,} pairs ({pct:.0f}%)")

        logger.info(f"Completed building similarity matrix: {n}x{n} ({total_pairs:,} pairs)")

        return similarity_matrix

    def _should_group_adaptive(
        self,
        score: float,
        phonetic_agree: bool,
        t_low: float,
        s_90: float,
        t_high: float
    ) -> bool:
        """
        Determine if a pair should be grouped using three-zone decision logic.

        Zones:
        - Auto-accept: score >= t_high → group
        - Promotion: s_90 <= score < t_high AND phonetic_agree → group
        - Reject: score <= t_low → do not group

        Args:
            score: Similarity score (0.0 to 1.0)
            phonetic_agree: Whether names have phonetic agreement
            t_low: Lower threshold (reject below)
            s_90: 90% confidence threshold
            t_high: High threshold (auto-accept above)

        Returns:
            True if pair should be grouped, False otherwise
        """
        # Auto-accept zone
        if score >= t_high:
            return True

        # Promotion zone (requires phonetic agreement)
        if s_90 <= score < t_high and phonetic_agree:
            return True

        # Reject zone
        if score <= t_low:
            return False

        # Middle zone (s_90 > score > t_low) without phonetic agreement → reject
        return False

    def _calculate_adjusted_confidence(
        self,
        score: float,
        phonetic_agree: bool,
        p_same: float,
        t_low: float,
        s_90: float,
        t_high: float
    ) -> float:
        """
        Calculate adjusted confidence score based on zone and GMM posterior.

        Zone-based rules:
        - Auto-accept (score >= t_high): 100 × p_same + (+4 if phonetic)
        - Promotion (s_90 <= score < t_high, phonetic_agree): 100 × p_same + 4 - penalty, capped at 89
        - Reject (score <= t_low): 100 × p_same (usually < 50)

        Args:
            score: Raw similarity score (0.0 to 1.0)
            phonetic_agree: Whether phonetics agree
            p_same: Posterior probability from GMM
            t_low: Lower threshold
            s_90: 90% confidence threshold
            t_high: High threshold

        Returns:
            Adjusted confidence score (0.0 to 1.0)
        """
        base_confidence = 100.0 * p_same

        # Auto-accept zone
        if score >= t_high:
            adjusted = base_confidence + (4.0 if phonetic_agree else 0.0)
            return min(adjusted / 100.0, 1.0)

        # Promotion zone
        if s_90 <= score < t_high and phonetic_agree:
            penalty = self.gmm_calculator.calculate_margin_penalty(score, s_90, t_high)
            adjusted = base_confidence + 4.0 - penalty
            # Cap at 89% to distinguish from auto-accepts
            adjusted = min(adjusted, 89.0)
            return adjusted / 100.0

        # Reject zone or other
        return base_confidence / 100.0

    def group_similar_names(
        self,
        names: List[str],
        adaptive_thresholds: Optional[Dict[str, float]] = None,
        pairwise_data: Optional[Dict[Tuple[str, str], Tuple[float, bool]]] = None
    ) -> List[List[str]]:
        """
        Group similar company names together using greedy clustering.

        Algorithm:
        - Iterate through names
        - For each unprocessed name, start a new group
        - Add all similar names (above threshold) to the group
        - Mark all grouped names as processed

        Args:
            names: List of company names to group
            adaptive_thresholds: Optional dict with 't_low', 's_90', 't_high' for adaptive mode
            pairwise_data: Optional dict mapping (name1, name2) -> (score, phonetic_agree)

        Returns:
            List of groups, where each group is a list of similar names
        """
        if not names:
            return []

        groups: List[List[str]] = []
        processed: Set[str] = set()

        # Determine if using adaptive mode
        use_adaptive = adaptive_thresholds is not None and pairwise_data is not None

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

                # Get similarity and phonetic info
                if use_adaptive:
                    # Look up precomputed scores
                    pair_key = (name, other_name) if (name, other_name) in pairwise_data else (other_name, name)
                    score, phonetic_agree = pairwise_data.get(pair_key, (0.0, False))

                    # Use three-zone decision logic
                    should_group = self._should_group_adaptive(
                        score,
                        phonetic_agree,
                        adaptive_thresholds['t_low'],
                        adaptive_thresholds['s_90'],
                        adaptive_thresholds['t_high']
                    )
                else:
                    # Fixed threshold mode
                    score = self.calculate_confidence(name, other_name)
                    should_group = score >= (self.similarity_threshold / 100.0)

                if should_group:
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
            - summary: Statistics about the processing (includes threshold_info)
            - gmm_metadata: Optional GMM metadata if adaptive mode used
        """
        start_time = datetime.now()
        logger.info(f"Starting to process {len(names)} company names from '{filename}'")

        # Initialize variables for adaptive thresholding
        adaptive_thresholds = None
        pairwise_data = None
        threshold_info = {}
        gmm_metadata = None
        hac_metadata = None
        gmm = None
        sampling_metadata = None

        # Step 1: Handle different clustering modes
        if self.clustering_mode == 'adaptive_gmm':
            logger.info("Collecting pairwise similarity scores for GMM fitting")
            pairs, sampling_metadata = self._collect_pairwise_scores(names)

            # Extract scores for GMM
            scores = [score for _, _, score, _ in pairs]

            # Attempt to fit GMM and calculate adaptive thresholds
            thresholds_result = self.gmm_calculator.calculate_adaptive_thresholds(scores)

            if thresholds_result is not None:
                # GMM fitting successful
                adaptive_thresholds = thresholds_result
                gmm = self.gmm_calculator.gmm

                # Build pairwise lookup dictionary
                pairwise_data = {
                    (name1, name2): (score, phonetic_agree)
                    for name1, name2, score, phonetic_agree in pairs
                }

                threshold_info = {
                    "method": "adaptive_gmm",
                    "t_low": adaptive_thresholds['t_low'],
                    "s_90": adaptive_thresholds['s_90'],
                    "t_high": adaptive_thresholds['t_high'],
                    "fixed_threshold": None,
                    "fallback_reason": None
                }

                gmm_metadata = self.gmm_calculator.get_gmm_metadata(gmm, len(scores))

                logger.info(
                    f"Adaptive thresholds: T_LOW={adaptive_thresholds['t_low']:.3f}, "
                    f"S_90={adaptive_thresholds['s_90']:.3f}, T_HIGH={adaptive_thresholds['t_high']:.3f}"
                )
            else:
                # GMM fitting failed - fallback to fixed threshold
                logger.warning(
                    f"GMM fitting failed (insufficient data: {len(scores)} pairs < {settings.GMM_MIN_SAMPLES} min). "
                    "Falling back to fixed threshold."
                )
                threshold_info = {
                    "method": "fixed",
                    "t_low": settings.GMM_FALLBACK_T_LOW / 100.0,
                    "s_90": None,
                    "t_high": settings.GMM_FALLBACK_T_HIGH / 100.0,
                    "fixed_threshold": self.similarity_threshold,
                    "fallback_reason": f"Insufficient data: {len(scores)} pairs < {settings.GMM_MIN_SAMPLES} minimum"
                }

        elif self.clustering_mode == 'hac':
            # HAC mode - build full similarity matrix and cluster
            logger.info("Using HAC clustering mode")
            similarity_matrix = self._build_similarity_matrix(names)

            # Run HAC clustering
            hac_clusters, hac_metadata = self.hac_service.cluster_names(names, similarity_matrix)

            # Convert HAC output format to groups list format
            groups = [hac_clusters[cluster_id] for cluster_id in sorted(hac_clusters.keys())]

            # Set threshold info
            threshold_info = {
                "method": "hac",
                "hac_threshold": self.hac_service.threshold,
                "hac_linkage": self.hac_service.linkage_method,
                "fixed_threshold": None,
                "t_low": None,
                "s_90": None,
                "t_high": None,
                "fallback_reason": None
            }

            logger.info(f"HAC produced {len(groups)} groups from {len(names)} names")

        else:
            # Fixed threshold mode (default)
            threshold_info = {
                "method": "fixed",
                "t_low": None,
                "s_90": None,
                "t_high": None,
                "fixed_threshold": self.similarity_threshold,
                "fallback_reason": None
            }

        # Step 2: Group similar names (skip if already grouped by HAC)
        if self.clustering_mode != 'hac':
            groups = self.group_similar_names(names, adaptive_thresholds, pairwise_data)
        # else: groups already set by HAC above

        # Step 3: Build mappings and audit log
        mappings = []
        audit_entries = []

        for group_id, group in enumerate(groups):
            # Select canonical name
            canonical = self.select_canonical_name(group)
            alternatives = [n for n in group if n != canonical]

            # Create mapping for each name in the group
            for original_name in group:
                # Calculate confidence score
                if adaptive_thresholds and gmm:
                    # Use adjusted confidence with GMM posterior
                    base_score = self.calculate_confidence(original_name, canonical)

                    # Get phonetic agreement
                    norm1 = self.normalize_name(original_name)
                    norm2 = self.normalize_name(canonical)
                    phonetic_bonus = self._calculate_phonetic_bonus(norm1, norm2)
                    phonetic_agree = phonetic_bonus > 0

                    # Get posterior probability
                    p_same = self.gmm_calculator.calculate_posterior_probability(gmm, base_score)

                    # For single-member groups or self-match, set confidence to 1.0
                    if original_name == canonical:
                        confidence = 1.0
                    else:
                        confidence = self._calculate_adjusted_confidence(
                            base_score,
                            phonetic_agree,
                            p_same,
                            adaptive_thresholds['t_low'],
                            adaptive_thresholds['s_90'],
                            adaptive_thresholds['t_high']
                        )
                else:
                    # Fixed threshold mode - use original logic
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
            "processing_time_seconds": processing_time,
            "threshold_info": threshold_info
        }

        logger.info(f"Processing complete: {len(groups)} groups created in {processing_time:.2f}s")

        result = {
            "mappings": mappings,
            "audit_log": audit_log,
            "summary": summary
        }

        # Add GMM metadata if available
        if gmm_metadata:
            result["gmm_metadata"] = gmm_metadata

        # Add HAC metadata if available
        if hac_metadata:
            result["hac_metadata"] = hac_metadata

        # Add sampling metadata if available
        if sampling_metadata:
            result["sampling_metadata"] = sampling_metadata

        return result
