"""
Unit tests for custom PPO policy network
Tests MarketMakingFeatureExtractor and MarketMakingPolicy
"""

import pytest
import torch
import numpy as np
from gymnasium import spaces
from unittest.mock import MagicMock, patch

from app.agents.policy import MarketMakingFeatureExtractor, MarketMakingPolicy


class TestMarketMakingFeatureExtractor:
    """Test MarketMakingFeatureExtractor class"""
    
    @pytest.fixture
    def observation_space(self):
        """Create a 52-dimensional observation space"""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(52,),
            dtype=np.float32
        )
    
    def test_initialization_default(self, observation_space):
        """Should initialize with default parameters"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space
        )
        
        assert extractor.features_dim == 256
        assert extractor.normalize_input is True
        assert extractor.input_norm is not None
        assert extractor.fc1 is not None
        assert extractor.fc2 is not None
        assert extractor.fc3 is not None
    
    def test_initialization_custom_params(self, observation_space):
        """Should initialize with custom parameters"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            features_dim=128,
            normalize_input=False,
            dropout=0.2
        )
        
        assert extractor.features_dim == 128
        assert extractor.normalize_input is False
        assert extractor.input_norm is None
        assert extractor.dropout1.p == 0.2
        assert extractor.dropout2.p == 0.2
    
    def test_forward_2d_tensor_with_normalization(self, observation_space):
        """Should process 2D tensor (batch, features) with normalization"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            normalize_input=True
        )
        
        batch_size = 32
        observations = torch.randn(batch_size, 52)
        
        output = extractor.forward(observations)
        
        assert output.shape == (batch_size, 256)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_2d_tensor_without_normalization(self, observation_space):
        """Should process 2D tensor without normalization"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            normalize_input=False
        )
        
        batch_size = 16
        observations = torch.randn(batch_size, 52)
        
        output = extractor.forward(observations)
        
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_forward_3d_tensor(self, observation_space):
        """Should handle 3D tensor (batch, seq, features) by flattening"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            normalize_input=True
        )
        extractor.eval()  # Set to eval mode for BatchNorm with variable batch sizes
        
        batch_size = 8
        seq_len = 10
        observations = torch.randn(batch_size, seq_len, 52)
        
        # Should now properly handle 3D tensors by flattening and reshaping
        output = extractor.forward(observations)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connections(self, observation_space):
        """Should use residual connections correctly"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space
        )
        
        observations = torch.randn(4, 52)
        output = extractor.forward(observations)
        
        # Output should be reasonable (not all zeros or NaNs)
        assert torch.abs(output).mean() > 0.0
        assert not torch.isnan(output).any()
    
    def test_dropout_training_mode(self, observation_space):
        """Should apply dropout in training mode"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            dropout=0.5
        )
        extractor.train()
        
        observations = torch.randn(10, 52)
        output1 = extractor.forward(observations)
        output2 = extractor.forward(observations)
        
        # In training mode, outputs should differ due to dropout
        assert not torch.equal(output1, output2)
    
    def test_dropout_eval_mode(self, observation_space):
        """Should not apply dropout in eval mode"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            dropout=0.5
        )
        extractor.eval()
        
        observations = torch.randn(10, 52)
        output1 = extractor.forward(observations)
        output2 = extractor.forward(observations)
        
        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_different_input_dimensions(self, observation_space):
        """Should handle different input dimensions correctly"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            features_dim=128
        )
        
        # Test with different batch sizes
        # Note: BatchNorm requires >1 sample in training mode, so we use eval mode for single sample
        for batch_size in [1, 4, 16, 32]:
            if batch_size == 1:
                extractor.eval()  # Eval mode for single sample
            else:
                extractor.train()  # Training mode for multiple samples
            observations = torch.randn(batch_size, 52)
            output = extractor.forward(observations)
            assert output.shape == (batch_size, 128)
    
    def test_gradient_flow(self, observation_space):
        """Should allow gradients to flow through the network"""
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space
        )
        
        observations = torch.randn(4, 52, requires_grad=True)
        output = extractor.forward(observations)
        
        # Compute a dummy loss and backprop
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist
        assert observations.grad is not None
        assert not torch.isnan(observations.grad).any()


class TestMarketMakingPolicy:
    """Test MarketMakingPolicy class"""
    
    @pytest.fixture
    def observation_space(self):
        """Create observation space"""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(52,),
            dtype=np.float32
        )
    
    @pytest.fixture
    def action_space(self):
        """Create action space (MultiDiscrete for market making)"""
        return spaces.MultiDiscrete([21, 10, 21, 10])
    
    @pytest.fixture
    def lr_schedule(self):
        """Create a simple learning rate schedule"""
        def schedule(progress_remaining):
            return 3e-4 * progress_remaining
        return schedule
    
    def test_initialization_default(self, observation_space, action_space, lr_schedule):
        """Should initialize with default network architecture"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Check that policy was created successfully
        assert policy is not None
        assert hasattr(policy, 'features_extractor')
        assert isinstance(policy.features_extractor, MarketMakingFeatureExtractor)
    
    def test_initialization_custom_net_arch(self, observation_space, action_space, lr_schedule):
        """Should initialize with custom network architecture"""
        custom_net_arch = [
            {"pi": [128, 128], "vf": [128, 128]}
        ]
        
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=custom_net_arch
        )
        
        assert policy is not None
    
    def test_initialization_custom_features_dim(self, observation_space, action_space, lr_schedule):
        """Should initialize with custom features dimension"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_dim=128
        )
        
        assert policy.features_extractor.features_dim == 128
    
    def test_initialization_without_normalization(self, observation_space, action_space, lr_schedule):
        """Should initialize without input normalization"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            normalize_input=False
        )
        
        assert policy.features_extractor.normalize_input is False
    
    def test_forward_pass(self, observation_space, action_space, lr_schedule):
        """Should perform forward pass through policy"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        # Create dummy observation
        observation = torch.randn(4, 52)
        
        # Forward pass
        with torch.no_grad():
            features = policy.extract_features(observation)
            assert features.shape == (4, 256)
    
    def test_policy_output_shapes(self, observation_space, action_space, lr_schedule):
        """Should produce correct output shapes for policy and value"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        observation = torch.randn(8, 52)
        
        with torch.no_grad():
            # Get action distribution
            dist = policy.get_distribution(observation)
            assert dist is not None
            
            # Get value estimate
            values = policy.predict_values(observation)
            assert values.shape == (8, 1)
    
    def test_action_prediction(self, observation_space, action_space, lr_schedule):
        """Should predict actions correctly"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        observation = np.random.randn(52).astype(np.float32)
        
        # Predict action (deterministic)
        action, _ = policy.predict(observation, deterministic=True)
        
        assert action.shape == (4,)  # MultiDiscrete action space
        assert action.dtype == np.int64 or action.dtype == np.int32
        assert all(0 <= action[i] < action_space.nvec[i] for i in range(4))
    
    def test_stochastic_action_prediction(self, observation_space, action_space, lr_schedule):
        """Should predict stochastic actions"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        observation = np.random.randn(52).astype(np.float32)
        
        # Predict action (stochastic)
        action1, _ = policy.predict(observation, deterministic=False)
        action2, _ = policy.predict(observation, deterministic=False)
        
        # Actions might be different (stochastic)
        assert action1.shape == (4,)
        assert action2.shape == (4,)
    
    def test_batch_processing(self, observation_space, action_space, lr_schedule):
        """Should handle batch of observations"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule
        )
        
        batch_observations = np.random.randn(16, 52).astype(np.float32)
        
        # Process batch
        actions, values = policy.predict(batch_observations, deterministic=True)
        
        assert len(actions) == 16
        assert all(a.shape == (4,) for a in actions)
    
    def test_feature_extractor_integration(self, observation_space, action_space, lr_schedule):
        """Should use custom feature extractor correctly"""
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_dim=128,
            normalize_input=True
        )
        
        # Check feature extractor configuration
        assert policy.features_extractor.features_dim == 128
        assert policy.features_extractor.normalize_input is True
        
        # Test forward pass
        observation = torch.randn(4, 52)
        with torch.no_grad():
            features = policy.extract_features(observation)
            assert features.shape == (4, 128)

