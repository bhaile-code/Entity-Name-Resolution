"""
Unit tests for GMM threshold service.
Tests the Gaussian Mixture Model-based adaptive thresholding functionality.
"""
import pytest
import numpy as np
from app.services.gmm_threshold_service import GMMThresholdCalculator


class TestGMMThresholdCalculator:
    """Test suite for GMMThresholdCalculator class."""

    def test_fit_gmm_with_sufficient_data(self):
        """Test GMM fitting with sufficient bimodal data."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate synthetic bimodal data
        # Low cluster (different companies): mean=0.3, std=0.1
        # High cluster (same companies): mean=0.9, std=0.05
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)
        np.random.shuffle(scores)

        # Fit GMM
        gmm = calculator.fit_gmm(scores)

        assert gmm is not None
        assert calculator.high_cluster_idx is not None
        assert 0 <= calculator.high_cluster_idx <= 1

        # Verify high cluster has higher mean
        means = gmm.means_.flatten()
        assert means[calculator.high_cluster_idx] > means[1 - calculator.high_cluster_idx]

    def test_fit_gmm_with_insufficient_data(self):
        """Test GMM fitting fails gracefully with insufficient data."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Only 30 samples (< 50 minimum)
        scores = [0.5] * 30

        gmm = calculator.fit_gmm(scores)

        assert gmm is None

    def test_calculate_posterior_probability(self):
        """Test posterior probability calculation."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate bimodal data
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)

        gmm = calculator.fit_gmm(scores)
        assert gmm is not None

        # Test posterior probabilities
        # Low score should have low P(same)
        p_same_low = calculator.calculate_posterior_probability(gmm, 0.3)
        assert 0 <= p_same_low < 0.5

        # High score should have high P(same)
        p_same_high = calculator.calculate_posterior_probability(gmm, 0.9)
        assert p_same_high > 0.5

    def test_find_threshold_for_posterior(self):
        """Test threshold finding for specific posterior probabilities."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate bimodal data
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)

        gmm = calculator.fit_gmm(scores)
        assert gmm is not None

        # Find S_90 threshold
        s_90 = calculator.find_threshold_for_posterior(gmm, target_prob=0.90)
        assert s_90 is not None
        assert 0 <= s_90 <= 1

        # Verify the threshold gives approximately the target probability
        p_at_s90 = calculator.calculate_posterior_probability(gmm, s_90)
        assert abs(p_at_s90 - 0.90) < 0.01  # Within 1% tolerance

    def test_calculate_adaptive_thresholds(self):
        """Test complete adaptive threshold calculation."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate bimodal data
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)

        thresholds = calculator.calculate_adaptive_thresholds(scores)

        assert thresholds is not None
        assert 't_low' in thresholds
        assert 's_90' in thresholds
        assert 't_high' in thresholds

        # Verify threshold ordering: T_LOW < S_90 < T_HIGH
        assert thresholds['t_low'] < thresholds['s_90'] < thresholds['t_high']

        # Verify thresholds are in valid range
        assert 0 <= thresholds['t_low'] <= 1
        assert 0 <= thresholds['s_90'] <= 1
        assert 0 <= thresholds['t_high'] <= 1

    def test_calculate_margin_penalty(self):
        """Test margin penalty calculation for promotion zone."""
        calculator = GMMThresholdCalculator(min_samples=50)

        s_90 = 0.85
        t_high = 0.95

        # At T_HIGH: penalty should be 0
        penalty_high = calculator.calculate_margin_penalty(t_high, s_90, t_high)
        assert penalty_high == 0.0

        # At S_90: penalty should be 10
        penalty_low = calculator.calculate_margin_penalty(s_90, s_90, t_high)
        assert penalty_low == 10.0

        # In the middle: penalty should be between 0 and 10
        mid_score = (s_90 + t_high) / 2
        penalty_mid = calculator.calculate_margin_penalty(mid_score, s_90, t_high)
        assert 0 < penalty_mid < 10

        # At the midpoint, penalty should be close to 5
        assert abs(penalty_mid - 5.0) < 0.1

    def test_get_gmm_metadata(self):
        """Test GMM metadata extraction."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate bimodal data
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)

        gmm = calculator.fit_gmm(scores)
        assert gmm is not None

        metadata = calculator.get_gmm_metadata(gmm, len(scores))

        assert 'cluster_means' in metadata
        assert 'cluster_variances' in metadata
        assert 'cluster_weights' in metadata
        assert 'total_pairs_analyzed' in metadata
        assert 'high_cluster_index' in metadata

        assert len(metadata['cluster_means']) == 2
        assert len(metadata['cluster_variances']) >= 2
        assert len(metadata['cluster_weights']) == 2
        assert metadata['total_pairs_analyzed'] == len(scores)

        # Weights should sum to approximately 1
        assert abs(sum(metadata['cluster_weights']) - 1.0) < 0.01

    def test_fallback_with_insufficient_samples(self):
        """Test that insufficient samples returns None for thresholds."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Only 20 samples
        scores = [0.5 + i * 0.01 for i in range(20)]

        thresholds = calculator.calculate_adaptive_thresholds(scores)

        assert thresholds is None

    def test_threshold_boundary_conditions(self):
        """Test threshold finding at boundary conditions."""
        calculator = GMMThresholdCalculator(min_samples=50)

        # Generate bimodal data
        np.random.seed(42)
        low_scores = np.random.normal(0.3, 0.1, 100).clip(0, 1)
        high_scores = np.random.normal(0.9, 0.05, 100).clip(0, 1)
        scores = list(low_scores) + list(high_scores)

        gmm = calculator.fit_gmm(scores)
        assert gmm is not None

        # Test T_LOW (P=0.02)
        t_low = calculator.find_threshold_for_posterior(gmm, target_prob=0.02)
        assert t_low is not None
        p_at_t_low = calculator.calculate_posterior_probability(gmm, t_low)
        assert abs(p_at_t_low - 0.02) < 0.05  # Within 5% tolerance

        # Test T_HIGH (P=0.98)
        t_high = calculator.find_threshold_for_posterior(gmm, target_prob=0.98)
        assert t_high is not None
        p_at_t_high = calculator.calculate_posterior_probability(gmm, t_high)
        assert abs(p_at_t_high - 0.98) < 0.05  # Within 5% tolerance
