"""
Unit tests for feature engineering module
Tests FeatureNormalizer and MarketFeatureEngine
"""

import pytest
import numpy as np
from app.agents.features import FeatureNormalizer, MarketFeatureEngine


class TestFeatureNormalizer:
    """Test FeatureNormalizer class"""
    
    def test_initialization(self):
        """Should initialize with correct default values"""
        normalizer = FeatureNormalizer(feature_dim=52)
        
        assert normalizer.feature_mean.shape == (52,)
        assert normalizer.feature_std.shape == (52,)
        assert np.all(normalizer.feature_mean == 0.0)
        assert np.all(normalizer.feature_std == 1.0)
        assert normalizer.count == 0
        assert normalizer.momentum == 0.99
    
    def test_initialization_custom_momentum(self):
        """Should initialize with custom momentum"""
        normalizer = FeatureNormalizer(feature_dim=10, momentum=0.95)
        
        assert normalizer.momentum == 0.95
    
    def test_update_first_batch(self):
        """Should set stats directly on first update"""
        normalizer = FeatureNormalizer(feature_dim=3)
        
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalizer.update(features)
        
        assert np.allclose(normalizer.feature_mean, features)
        assert np.allclose(normalizer.feature_std, [1e-8, 1e-8, 1e-8], atol=1e-7)
        assert normalizer.count == 1
    
    def test_update_batch_1d(self):
        """Should handle 1D feature arrays"""
        normalizer = FeatureNormalizer(feature_dim=3)
        
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalizer.update(features)
        
        assert normalizer.count == 1
        assert normalizer.feature_mean.shape == (3,)
    
    def test_update_batch_2d(self):
        """Should handle 2D feature arrays (batch)"""
        normalizer = FeatureNormalizer(feature_dim=3)
        
        features = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        normalizer.update(features)
        
        expected_mean = np.mean(features, axis=0)
        assert np.allclose(normalizer.feature_mean, expected_mean)
        assert normalizer.count == 1
    
    def test_update_exponential_moving_average(self):
        """Should update using exponential moving average after first batch"""
        normalizer = FeatureNormalizer(feature_dim=2, momentum=0.9)
        
        # First update
        features1 = np.array([1.0, 2.0], dtype=np.float32)
        normalizer.update(features1)
        
        # Second update
        features2 = np.array([3.0, 4.0], dtype=np.float32)
        normalizer.update(features2)
        
        # Check EMA calculation
        expected_mean = 0.9 * features1 + 0.1 * features2
        assert np.allclose(normalizer.feature_mean, expected_mean)
        assert normalizer.count == 2
    
    def test_normalize_before_update(self):
        """Should return original features if not updated yet"""
        normalizer = FeatureNormalizer(feature_dim=3)
        
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalized = normalizer.normalize(features)
        
        # Before update, mean=0, std=1, so should return features as-is
        assert np.allclose(normalized, features)
    
    def test_normalize_after_update(self):
        """Should normalize features correctly after update"""
        normalizer = FeatureNormalizer(feature_dim=3)
        
        # Update with a batch to get meaningful std
        features_batch = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        normalizer.update(features_batch)
        
        # Normalize a new feature vector
        new_features = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        normalized = normalizer.normalize(new_features)
        
        # Should normalize using learned stats (mean and std from batch)
        expected = (new_features - normalizer.feature_mean) / normalizer.feature_std
        assert np.allclose(normalized, expected, atol=1e-6)
    
    def test_normalize_different_features(self):
        """Should normalize new features using learned stats"""
        normalizer = FeatureNormalizer(feature_dim=2)
        
        # Update with first batch
        features1 = np.array([10.0, 20.0], dtype=np.float32)
        normalizer.update(features1)
        
        # Normalize different features
        features2 = np.array([15.0, 25.0], dtype=np.float32)
        normalized = normalizer.normalize(features2)
        
        # Should normalize using learned mean and std
        expected = (features2 - normalizer.feature_mean) / normalizer.feature_std
        assert np.allclose(normalized, expected)
    
    def test_std_minimum_threshold(self):
        """Should enforce minimum std threshold to avoid division by zero"""
        normalizer = FeatureNormalizer(feature_dim=2)
        
        # Update with constant features (zero std)
        features = np.array([5.0, 5.0], dtype=np.float32)
        normalizer.update(features)
        
        # Std should be at least 1e-8
        assert np.all(normalizer.feature_std >= 1e-8)
        
        # Normalization should not cause division by zero
        normalized = normalizer.normalize(features)
        assert np.all(np.isfinite(normalized))


class TestMarketFeatureEngine:
    """Test MarketFeatureEngine class"""
    
    def test_initialization_with_normalization(self):
        """Should initialize with normalization enabled"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        assert engine.enable_normalization is True
        assert engine.normalizer is not None
        assert engine.normalizer.feature_mean.shape[0] == 52
    
    def test_initialization_without_normalization(self):
        """Should initialize without normalization"""
        engine = MarketFeatureEngine(enable_normalization=False)
        
        assert engine.enable_normalization is False
        assert engine.normalizer is None
    
    def test_extract_advanced_features_no_normalization(self):
        """Should return original features when normalization disabled"""
        engine = MarketFeatureEngine(enable_normalization=False)
        
        observation = np.random.randn(52).astype(np.float32)
        features = engine.extract_advanced_features(observation)
        
        assert np.array_equal(features, observation)
    
    def test_extract_advanced_features_normalization_not_updated(self):
        """Should return original features if normalizer not updated yet"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        observation = np.random.randn(52).astype(np.float32)
        features = engine.extract_advanced_features(observation)
        
        # Should return original since count is 0
        assert np.array_equal(features, observation)
    
    def test_extract_advanced_features_normalization_updated(self):
        """Should return normalized features after update"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        # Update with some observations
        observations = np.random.randn(10, 52).astype(np.float32)
        engine.update_statistics(observations)
        
        # Extract features
        observation = np.random.randn(52).astype(np.float32)
        features = engine.extract_advanced_features(observation)
        
        # Should be normalized
        assert not np.array_equal(features, observation)
        assert features.shape == (52,)
        assert features.dtype == np.float32
    
    def test_update_statistics_with_normalization(self):
        """Should update normalizer statistics when normalization enabled"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        observations = np.random.randn(5, 52).astype(np.float32)
        engine.update_statistics(observations)
        
        assert engine.normalizer.count == 1
        assert engine.normalizer.feature_mean.shape == (52,)
    
    def test_update_statistics_without_normalization(self):
        """Should not update when normalization disabled"""
        engine = MarketFeatureEngine(enable_normalization=False)
        
        observations = np.random.randn(5, 52).astype(np.float32)
        engine.update_statistics(observations)
        
        # Should not crash, but normalizer should remain None
        assert engine.normalizer is None
    
    def test_update_statistics_1d(self):
        """Should handle 1D observation arrays"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        observation = np.random.randn(52).astype(np.float32)
        engine.update_statistics(observation)
        
        assert engine.normalizer.count == 1
    
    def test_update_statistics_2d(self):
        """Should handle 2D observation arrays (batch)"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        observations = np.random.randn(10, 52).astype(np.float32)
        engine.update_statistics(observations)
        
        assert engine.normalizer.count == 1
    
    def test_feature_dimension_mismatch(self):
        """Should raise error on dimension mismatches"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        # Update with correct dimension first
        correct_obs = np.random.randn(52).astype(np.float32)
        engine.update_statistics(correct_obs)
        
        # Wrong dimension observation (50 instead of 52)
        wrong_obs = np.random.randn(50).astype(np.float32)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            engine.extract_advanced_features(wrong_obs)
    
    def test_integration_workflow(self):
        """Test complete workflow: update -> extract -> normalize"""
        engine = MarketFeatureEngine(enable_normalization=True)
        
        # Simulate training: update with batch of observations
        batch_observations = np.random.randn(100, 52).astype(np.float32)
        engine.update_statistics(batch_observations)
        
        # Extract features for new observation
        new_observation = np.random.randn(52).astype(np.float32)
        features = engine.extract_advanced_features(new_observation)
        
        # Features should be normalized
        assert features.shape == (52,)
        assert np.all(np.isfinite(features))
        
        # Mean should be close to zero (normalized)
        assert abs(np.mean(features)) < 10.0  # Reasonable range after normalization

