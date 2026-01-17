"""
Hyperparameters configuration for market making RL agent
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    """
    training hyperparameters
    """
    #env
    symbol: str = "BTC/USDT"
    initial_cash: float = 10000.0
    max_steps: int = 1000
    n_envs: int = 4

    #training
    total_timesteps: int = 100_000
    learning_rate:float = 3e-4
    batch_size:int = 64
    n_steps: int = 2048
    n_epochs: int = 10

    #PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05  
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    #network architecture
    features_dim: int = 256
    normalize_input: bool = True
    dropout: float = 0.1
    net_arch: list = None

    #Callbacks
    save_freq: int = 10000
    eval_freq: int = 20000
    eval_episodes: int = 10

    #paths
    model_save_path: str = "models/market_making_ppo"
    tensorboard_log: str = "logs/tensorboard"

    def __post_init__(self):
        """
        post-initialization setup
        """
        if self.net_arch is None:
            self.net_arch = [{'pi': [256, 256], 'vf': [256, 256]}]

    def to_dict(self) -> Dict[str, Any]:
        """
        convert to dictionary for training
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

#prefefined configurations
CONFIGS = {
    "fast": TrainingConfig(
        total_timesteps = 10000,
        n_envs = 2,
        n_steps = 512,
        n_epochs = 4,
        learning_rate = 1e-3,
    ),

    "default": TrainingConfig(),

    "stable": TrainingConfig(
        learning_rate=1e-4,
        batch_size=128,
        n_steps=4096,
        n_epochs=20,
        clip_range=0.1,
        ent_coef=0.05,
        gamma=0.995
    ),
    "aggressive": TrainingConfig(
        learning_rate=5e-4,
        batch_size=32,
        n_steps=1024,
        n_epochs=5,
        clip_range=0.3,
        ent_coef=0.001,
        gamma=0.95
    )

}