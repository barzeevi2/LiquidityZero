"""
custom PPO policy network for market making 
actor-critic architecture with feature normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class MarketMakingFeatureExtractor(BaseFeaturesExtractor):
    """
    custom feature extractor with nornalization and residual connections
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        normalize_input: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0] #52

        self.normalize_input = normalize_input
        
        #batch normalization (only if normalize_input is True)
        if normalize_input:
            self.input_norm = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.input_norm = None

        #feature processing layers with residual connections (always needed)
        self.fc1 = nn.Linear(input_dim, features_dim)
        self.bn1 = nn.BatchNorm1d(features_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(features_dim, features_dim)
        self.bn2 = nn.BatchNorm1d(features_dim)
        self.dropout2 = nn.Dropout(dropout)

        #residual connection
        self.residual_proj = nn.Linear(input_dim, features_dim) if input_dim != features_dim else nn.Identity()

        self.fc3 = nn.Linear(features_dim, features_dim)
        self.bn3 = nn.BatchNorm1d(features_dim)
    

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        forward pass through feature extractor
        takes in observations (batch, 52) tensor
        returns features (batch, features_dim) tensor
        """

        x = observations
        
        # Handle 3D tensors by flattening sequence dimension
        # (3D tensors are not standard for RL training, but we support them)
        original_shape = None
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            original_shape = (batch_size, seq_len)
            x = x.view(-1, features)  # Flatten to (batch*seq, features)
        
        #input normalization
        if self.normalize_input and self.input_norm is not None:
            x = self.input_norm(x)
        
        #first layer with residual connection
        residual = self.residual_proj(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        #second layer with residual connection
        x = x + residual
        residual2 = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #third layer
        x = x + residual2
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Reshape back to 3D if needed
        if original_shape is not None:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x



class MarketMakingPolicy(ActorCriticPolicy):
    """
    custom PPO policy for market making with specialized architecture
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        features_dim: int = 256,
        normalize_input: bool = True,
        dropout: float = 0.1,
        net_arch: Optional[List] = None,
        **kwargs
    ):
        #default network architecture if not specified
        if net_arch is None:
            net_arch = [
                {"pi": [256, 256], "vf": [256, 256]} #separate networks for policy and value
            ]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch = net_arch,
            features_extractor_class = MarketMakingFeatureExtractor,
            features_extractor_kwargs = {
                "features_dim": features_dim,
                "normalize_input": normalize_input,
                "dropout": dropout
            },
            **kwargs
        )