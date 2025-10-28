"""
GMM-based adaptive threshold calculation service.
Uses Gaussian Mixture Model to determine data-driven thresholds for name matching.
"""
import numpy as np
from typing import Optional, Dict, Tuple, List
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GMMThresholdCalculator:
    """
    Calculates adaptive thresholds using a 2-component Gaussian Mixture Model.

    The GMM identifies two clusters in similarity scores:
    - Low cluster: Different companies (low similarity)
    - High cluster: Same companies (high similarity)

    Thresholds are calculated based on posterior probabilities:
    - T_LOW: P(same|score) = 0.02 (reject threshold)
    - S_90: P(same|score) = 0.90 (promotion eligibility)
    - T_HIGH: P(same|score) = 0.98 (auto-accept threshold)
    """

    def __init__(self, min_samples: int = 50):
        """
        Initialize the GMM threshold calculator.

        Args:
            min_samples: Minimum number of samples required for reliable GMM fitting
        """
        self.min_samples = min_samples
        self.gmm: Optional[GaussianMixture] = None
        self.high_cluster_idx: Optional[int] = None

    def fit_gmm(self, scores: List[float]) -> Optional[GaussianMixture]:
        """
        Fit a 2-component Gaussian Mixture Model on similarity scores.

        Args:
            scores: List of similarity scores (0.0 to 1.0)

        Returns:
            Fitted GaussianMixture model, or None if insufficient data or fit fails
        """
        if len(scores) < self.min_samples:
            logger.warning(
                f"Insufficient samples for GMM: {len(scores)} < {self.min_samples}. "
                "Falling back to fixed threshold."
            )
            return None

        try:
            # Reshape for sklearn (needs 2D array)
            X = np.array(scores).reshape(-1, 1)

            # STABILITY FIX: Use k-means initialization for more reproducible results
            # random_state=42 ensures consistent k-means initialization
            # Note: Do NOT sort scores - it changes the distribution and breaks GMM
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=42,
                max_iter=200,
                init_params='kmeans'
            )
            gmm.fit(X)

            # Identify high cluster (same companies) - has higher mean
            means = gmm.means_.flatten()
            self.high_cluster_idx = np.argmax(means)

            self.gmm = gmm
            logger.info(
                f"GMM fitted successfully on {len(scores)} samples. "
                f"Cluster means: {means[0]:.3f}, {means[1]:.3f}"
            )

            return gmm

        except Exception as e:
            logger.error(f"Failed to fit GMM: {e}", exc_info=True)
            return None

    def calculate_posterior_probability(self, gmm: GaussianMixture, score: float) -> float:
        """
        Calculate P(same|score) - posterior probability that a pair belongs to the "same" cluster.

        Args:
            gmm: Fitted GaussianMixture model
            score: Similarity score (0.0 to 1.0)

        Returns:
            Posterior probability that the pair is the same company (0.0 to 1.0)
        """
        if gmm is None or self.high_cluster_idx is None:
            raise ValueError("GMM must be fitted before calculating posteriors")

        X = np.array([[score]])

        # Get posterior probabilities for both clusters
        posteriors = gmm.predict_proba(X)[0]

        # Return probability of high cluster (same company)
        return posteriors[self.high_cluster_idx]

    def find_threshold_for_posterior(
        self,
        gmm: GaussianMixture,
        target_prob: float,
        search_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Optional[float]:
        """
        Find the score where P(same|score) = target_prob using binary search.

        Args:
            gmm: Fitted GaussianMixture model
            target_prob: Target posterior probability (e.g., 0.90 for S_90)
            search_range: Range to search within (min_score, max_score)

        Returns:
            Score where P(same|score) = target_prob, or None if not found
        """
        if gmm is None:
            return None

        def objective(score: float) -> float:
            """Objective function: P(same|score) - target_prob"""
            return self.calculate_posterior_probability(gmm, score) - target_prob

        try:
            # Check if target_prob is achievable in the search range
            prob_at_min = self.calculate_posterior_probability(gmm, search_range[0])
            prob_at_max = self.calculate_posterior_probability(gmm, search_range[1])

            # If target is outside range, return boundary
            if target_prob <= prob_at_min:
                return search_range[0]
            if target_prob >= prob_at_max:
                return search_range[1]

            # Binary search for the crossing point
            threshold = brentq(objective, search_range[0], search_range[1], xtol=1e-4)
            return float(threshold)

        except ValueError as e:
            logger.warning(f"Could not find threshold for P={target_prob}: {e}")
            return None

    def calculate_adaptive_thresholds(
        self,
        scores: List[float]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate T_LOW, S_90, and T_HIGH from similarity scores using GMM.

        Args:
            scores: List of similarity scores (0.0 to 1.0)

        Returns:
            Dictionary with keys: 't_low', 's_90', 't_high', or None if calculation fails
        """
        gmm = self.fit_gmm(scores)

        if gmm is None:
            return None

        # Calculate thresholds based on posterior probabilities
        t_low = self.find_threshold_for_posterior(gmm, target_prob=0.02)
        s_90 = self.find_threshold_for_posterior(gmm, target_prob=0.90)
        t_high = self.find_threshold_for_posterior(gmm, target_prob=0.98)

        if t_low is None or s_90 is None or t_high is None:
            logger.warning("Failed to calculate one or more thresholds")
            return None

        thresholds = {
            't_low': t_low,
            's_90': s_90,
            't_high': t_high
        }

        logger.info(
            f"Adaptive thresholds calculated: "
            f"T_LOW={t_low:.3f}, S_90={s_90:.3f}, T_HIGH={t_high:.3f}"
        )

        return thresholds

    def calculate_margin_penalty(self, score: float, s_90: float, t_high: float) -> float:
        """
        Calculate margin penalty for promoted pairs in the [S_90, T_HIGH) zone.

        Penalty is linearly interpolated:
        - At T_HIGH: penalty = 0
        - At S_90: penalty = 10

        Args:
            score: Similarity score
            s_90: Lower bound of promotion zone
            t_high: Upper bound of promotion zone (auto-accept threshold)

        Returns:
            Penalty value (0 to 10)
        """
        if score >= t_high:
            return 0.0
        if score <= s_90:
            return 10.0

        # Linear interpolation: penalty = 10 × (1 - (score - s_90) / (t_high - s_90))
        normalized_position = (score - s_90) / (t_high - s_90)
        penalty = 10.0 * (1.0 - normalized_position)

        return penalty

    def get_gmm_metadata(self, gmm: GaussianMixture, total_pairs: int) -> Dict:
        """
        Extract metadata from fitted GMM for transparency and debugging.

        Args:
            gmm: Fitted GaussianMixture model
            total_pairs: Total number of pairs analyzed

        Returns:
            Dictionary containing cluster statistics
        """
        if gmm is None:
            return {}

        means = gmm.means_.flatten().tolist()
        variances = gmm.covariances_.flatten().tolist()
        weights = gmm.weights_.tolist()

        return {
            'cluster_means': means,
            'cluster_variances': variances,
            'cluster_weights': weights,
            'total_pairs_analyzed': total_pairs,
            'high_cluster_index': self.high_cluster_idx
        }
