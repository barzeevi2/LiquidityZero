"""
feature engineering for market making RL
enhances raw observations with normalized and derived features
"""

import numpy as np
from typing import Dict, Optional


class FeatureNormalizer:
    """
    online normalization for features using running stats
    """

    def __init__(self, feature_dim: int, momentum: float = 0.99):
        """
        Initialize feature normalizer
        
        Args:
            feature_dim: dimension of feature vectors
            momentum: exponential moving average decay factor (default 0.99)
        """
        self.feature_mean = np.zeros(feature_dim, dtype=np.float32)
        self.feature_std = np.ones(feature_dim, dtype=np.float32)
        self.count = 0
        self.momentum = momentum

    def update(self, features: np.ndarray):
        """
        update running stats for features
        """
        # Validate dimension
        expected_dim = self.feature_mean.shape[0]
        if features.ndim > 1:
            actual_dim = features.shape[1]
        else:
            actual_dim = features.shape[0]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
        
        batch_mean = np.mean(features, axis=0) if features.ndim > 1 else features
        batch_std = np.std(features, axis=0) if features.ndim > 1 else np.zeros_like(features)
        if self.count == 0:
            self.feature_mean = batch_mean
            self.feature_std = np.maximum(batch_std, 1e-8)
        else:
            #exponential moving average
            self.feature_mean = (
                self.momentum * self.feature_mean + (1-self.momentum) * batch_mean
            )
            self.feature_std = np.maximum(
                np.sqrt(
                    self.momentum * self.feature_std**2 + (1-self.momentum) * batch_std**2
                ), 1e-8
            )
        self.count += 1
    

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        normalize features using running stats
        """
        # Validate dimension
        expected_dim = self.feature_mean.shape[0]
        if features.ndim > 1:
            actual_dim = features.shape[1]
        else:
            actual_dim = features.shape[0]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
        
        return (features - self.feature_mean) / self.feature_std
    


class MarketFeatureEngine:
    """
    enhanced feature engineering for market making
    """

    def __init__(self, enable_normalization: bool = True):
        self.enable_normalization = enable_normalization
        self.normalizer: Optional[FeatureNormalizer] = None
        if self.enable_normalization:
            self.normalizer = FeatureNormalizer(feature_dim=52)

    def extract_advanced_features(self, observation: np.ndarray) -> np.ndarray:
        """
        extract additional market signals from raw observation
        takes in an observation (raw 52 dim obs from env)
        returns enhanced feature vector
        """
        #original features are already in obs
        #we can add derived features here if needed
        # for now retur normalized version
        if self.normalizer and self.normalizer.count > 0:
            return self.normalizer.normalize(observation)
        
        return observation

    def update_statistics(self, observations: np.ndarray):
        """
        update normalization stats
        """
        if self.normalizer:
            self.normalizer.update(observations)