"""
Hierarchical Agglomerative Clustering (HAC) service for entity resolution.

Provides deterministic, threshold-based clustering as an alternative to GMM adaptive thresholding.
HAC is more stable and reproducible than GMM, working consistently with or without embeddings.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


class HACClusteringService:
    """
    Hierarchical Agglomerative Clustering service.

    Uses scipy's implementation of HAC with average linkage to cluster similar names.
    More deterministic and stable than GMM-based approaches.
    """

    def __init__(self, threshold: float = 0.42, linkage_method: str = 'average'):
        """
        Initialize HAC clustering service.

        Args:
            threshold: Distance threshold for cutting dendrogram (0-1 range).
                      Lower = more clusters (stricter), Higher = fewer clusters (looser).
                      Default 0.42 means similarity must be >= 0.58 to group.
            linkage_method: Method for computing distances between clusters.
                          Options: 'average' (default), 'single', 'complete', 'ward'
        """
        self.threshold = threshold
        self.linkage_method = linkage_method

        logger.info(
            f"Initialized HACClusteringService with threshold={threshold:.3f}, "
            f"linkage={linkage_method}"
        )

    def cluster_names(
        self,
        names: List[str],
        similarity_matrix: np.ndarray
    ) -> Tuple[Dict[int, List[str]], Dict[str, any]]:
        """
        Cluster names using Hierarchical Agglomerative Clustering.

        Args:
            names: List of company names to cluster
            similarity_matrix: Square matrix of pairwise similarities (0-1 range)

        Returns:
            Tuple of:
            - clusters: Dict mapping cluster_id -> list of names in that cluster
            - metadata: Dict with clustering statistics and info
        """
        n = len(names)

        if n == 0:
            logger.warning("Empty names list provided")
            return {}, self._get_empty_metadata()

        if n == 1:
            logger.info("Single name provided, returning single cluster")
            return {0: [names[0]]}, self._get_single_name_metadata()

        # Validate similarity matrix
        if similarity_matrix.shape != (n, n):
            raise ValueError(
                f"Similarity matrix shape {similarity_matrix.shape} doesn't match "
                f"names count {n}"
            )

        # Convert similarity to distance (HAC works with distances)
        distance_matrix = 1.0 - similarity_matrix

        # Ensure diagonal is zero (distance from name to itself)
        np.fill_diagonal(distance_matrix, 0.0)

        # Ensure non-negative distances (clamp any floating point errors)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

        # Convert to condensed distance matrix (required by scipy)
        # Only upper triangle, no diagonal
        try:
            condensed_distances = squareform(distance_matrix, checks=False)
        except ValueError as e:
            logger.error(f"Failed to convert distance matrix: {e}")
            raise ValueError(f"Invalid distance matrix: {e}")

        # Perform hierarchical clustering
        logger.info(
            f"Running HAC on {n} names with {self.linkage_method} linkage, "
            f"threshold={self.threshold:.3f}"
        )

        try:
            # Build linkage matrix (dendrogram)
            Z = linkage(condensed_distances, method=self.linkage_method)

            # Cut dendrogram at threshold to get cluster assignments
            cluster_labels = fcluster(Z, t=self.threshold, criterion='distance')

            logger.info(f"HAC produced {len(set(cluster_labels))} clusters")

        except Exception as e:
            logger.error(f"HAC clustering failed: {e}", exc_info=True)
            raise RuntimeError(f"HAC clustering failed: {e}")

        # Convert cluster labels to dict of name lists
        clusters = {}
        for name, label in zip(names, cluster_labels):
            label = int(label)  # Convert from numpy int to Python int
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(name)

        # Generate metadata
        metadata = self._generate_metadata(
            names=names,
            clusters=clusters,
            linkage_matrix=Z,
            distance_matrix=distance_matrix
        )

        return clusters, metadata

    def _generate_metadata(
        self,
        names: List[str],
        clusters: Dict[int, List[str]],
        linkage_matrix: np.ndarray,
        distance_matrix: np.ndarray
    ) -> Dict[str, any]:
        """Generate clustering metadata for analysis."""

        n_names = len(names)
        n_clusters = len(clusters)

        # Calculate cluster size statistics
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        avg_cluster_size = np.mean(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        singleton_clusters = sum(1 for size in cluster_sizes if size == 1)

        # Calculate distance statistics
        # Get all pairwise distances (upper triangle only)
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        avg_distance = np.mean(distances)

        # Calculate within-cluster distances (cohesion)
        within_cluster_distances = []
        for cluster_names in clusters.values():
            if len(cluster_names) > 1:
                # Get indices of names in this cluster
                indices = [names.index(name) for name in cluster_names]
                # Get pairwise distances within cluster
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        within_cluster_distances.append(distance_matrix[idx1, idx2])

        avg_within_cluster_distance = (
            np.mean(within_cluster_distances) if within_cluster_distances else 0.0
        )

        # Reduction percentage
        reduction_pct = ((n_names - n_clusters) / n_names * 100) if n_names > 0 else 0.0

        metadata = {
            'method': 'hac',
            'linkage_method': self.linkage_method,
            'threshold': self.threshold,
            'total_names': n_names,
            'total_clusters': n_clusters,
            'reduction_percentage': round(reduction_pct, 2),
            'avg_cluster_size': round(avg_cluster_size, 2),
            'max_cluster_size': max_cluster_size,
            'singleton_clusters': singleton_clusters,
            'avg_distance_all_pairs': round(avg_distance, 4),
            'avg_distance_within_clusters': round(avg_within_cluster_distance, 4),
            'cophenetic_distance': round(self._calculate_cophenetic_correlation(
                linkage_matrix, distance_matrix
            ), 4)
        }

        logger.info(
            f"HAC metadata: {n_clusters} clusters, {reduction_pct:.1f}% reduction, "
            f"avg_size={avg_cluster_size:.2f}"
        )

        return metadata

    def _calculate_cophenetic_correlation(
        self,
        linkage_matrix: np.ndarray,
        distance_matrix: np.ndarray
    ) -> float:
        """
        Calculate cophenetic correlation coefficient.

        Measures how well the dendrogram preserves pairwise distances.
        Higher values (close to 1.0) indicate better clustering quality.
        """
        try:
            from scipy.cluster.hierarchy import cophenet
            from scipy.spatial.distance import squareform

            condensed_distances = squareform(distance_matrix, checks=False)
            c, _ = cophenet(linkage_matrix, condensed_distances)
            return c
        except Exception as e:
            logger.warning(f"Failed to calculate cophenetic correlation: {e}")
            return 0.0

    def _get_empty_metadata(self) -> Dict[str, any]:
        """Return metadata for empty input."""
        return {
            'method': 'hac',
            'linkage_method': self.linkage_method,
            'threshold': self.threshold,
            'total_names': 0,
            'total_clusters': 0,
            'reduction_percentage': 0.0,
            'avg_cluster_size': 0.0,
            'max_cluster_size': 0,
            'singleton_clusters': 0,
            'avg_distance_all_pairs': 0.0,
            'avg_distance_within_clusters': 0.0,
            'cophenetic_distance': 0.0
        }

    def _get_single_name_metadata(self) -> Dict[str, any]:
        """Return metadata for single name input."""
        return {
            'method': 'hac',
            'linkage_method': self.linkage_method,
            'threshold': self.threshold,
            'total_names': 1,
            'total_clusters': 1,
            'reduction_percentage': 0.0,
            'avg_cluster_size': 1.0,
            'max_cluster_size': 1,
            'singleton_clusters': 1,
            'avg_distance_all_pairs': 0.0,
            'avg_distance_within_clusters': 0.0,
            'cophenetic_distance': 1.0
        }


def convert_similarity_threshold_to_distance(similarity_threshold: float) -> float:
    """
    Convert a similarity threshold (higher = more similar) to a distance threshold.

    Args:
        similarity_threshold: Similarity threshold in range [0, 100] or [0, 1]

    Returns:
        Distance threshold in range [0, 1] suitable for HAC

    Example:
        similarity_threshold=85 (85% similar) -> distance_threshold=0.15
        similarity_threshold=0.85 -> distance_threshold=0.15
    """
    # Normalize to 0-1 range if needed
    if similarity_threshold > 1.0:
        similarity_threshold = similarity_threshold / 100.0

    # Distance = 1 - similarity
    distance_threshold = 1.0 - similarity_threshold

    return distance_threshold


def recommend_threshold(
    similarity_matrix: np.ndarray,
    target_reduction: Optional[float] = None
) -> float:
    """
    Recommend an HAC distance threshold based on similarity distribution.

    Args:
        similarity_matrix: Square matrix of pairwise similarities
        target_reduction: Optional target reduction percentage (e.g., 0.30 for 30% reduction)

    Returns:
        Recommended distance threshold
    """
    # Get all pairwise similarities (upper triangle only)
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    # Calculate percentiles
    p25 = np.percentile(similarities, 25)
    p50 = np.percentile(similarities, 50)
    p75 = np.percentile(similarities, 75)
    p90 = np.percentile(similarities, 90)

    logger.info(
        f"Similarity distribution: P25={p25:.3f}, P50={p50:.3f}, "
        f"P75={p75:.3f}, P90={p90:.3f}"
    )

    if target_reduction is not None:
        # Heuristic: use percentile that roughly matches target reduction
        # More reduction -> lower percentile (more permissive distance threshold)
        if target_reduction >= 0.50:
            threshold_similarity = p25
        elif target_reduction >= 0.30:
            threshold_similarity = p50
        elif target_reduction >= 0.15:
            threshold_similarity = p75
        else:
            threshold_similarity = p90
    else:
        # Default: use 75th percentile (moderately conservative)
        threshold_similarity = p75

    distance_threshold = 1.0 - threshold_similarity

    logger.info(
        f"Recommended distance threshold: {distance_threshold:.3f} "
        f"(similarity: {threshold_similarity:.3f})"
    )

    return distance_threshold
