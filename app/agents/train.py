"""
training script for market making RL agent using PPO
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from app.sim.market_env import MarketMakingEnv
from app.agents.policy import MarketMakingPolicy
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TrainingMetricsCallback(BaseCallback):
    """
    custom callback to log training metrics
    """

    def __init__(self, verbose = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_pnls = []
    

    def _on_step(self) -> bool:
        #log metrics from info dict
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])


            #extract pnl from info if available
            if 'portfolio' in self.locals.get('infos', [{}])[0]:
                portfolio = self.locals['infos'][0]['portfolio']
                if 'total_pnl' in portfolio:
                    self.episode_pnls.append(portfolio['total_pnl'])
        return True
    

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            logger.info(f"Training completed. Metrics:")
            logger.info(f"Average reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"Average length: {np.mean(self.episode_lengths):.2f}")
            if self.episode_pnls:
                logger.info(f"Average PNL: {np.mean(self.episode_pnls):.2f}")
            else:
                logger.info("No PNL data available")
        


def make_env(
    rank: int=0,
    seed: int=0,
    start_time: datetime=None,
    end_time: datetime=None,
    random_start: bool=True
) ->MarketMakingEnv:
    """
    create and wrap environment for training
    takes in:
     rank: process rank for parallel training
     seed: random seed
     start_time: for data loading
     end_time: for data loading
     random_start: whether to start at random time or at start_time
    """
    def _init():
        env = MarketMakingEnv(
            symbol = settings.SYMBOL,
            initial_cash = settings.INITIAL_CASH,
            max_steps = settings.MAX_STEPS,
            price_tick_size = settings.PRICE_TICK_SIZE,
            quantity_precision = settings.QUANTITY_PRECISION,
            max_quantity = settings.MAX_QUANTITY,
            n_price_levels = settings.N_PRICE_LEVELS,
            n_quantity_levels = settings.N_QUANTITY_LEVELS,
            lookback_window = settings.LOOKBACK_WINDOW
        )

  
        #wrap with Monitor for statistics

        log_dir = Path("logs") / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(log_dir / f"monitor_{rank}.csv"))

        return env
    

    set_random_seed(seed + rank)
    return _init



def create_training_envs(
    n_envs: int = 4,
    start_time: datetime = None,
    end_time: datetime = None,
    random_start: bool = True
):
    """
    create vectorized environments for parallel training
    """

    if n_envs ==1:
        return DummyVecEnv([make_env(0, 0, start_time, end_time, random_start)])

    else:
        return SubprocVecEnv([
            make_env(i, 0, start_time, end_time, random_start)
             for i in range(n_envs)
        ])


def train_agent(
    total_timesteps: int = 1000000,
    n_envs: int = 4,
    learning_rate: float = 2e-4,  
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 15, 
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.05,  
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    features_dim: int = 256,
    normalize_input: bool = True,
    dropout: float = 0.1,
    save_freq: int = 10000,
    eval_freq: int = 20000,
    eval_episodes: int = 10,
    tensorboard_log: str = "logs/tensorboard",
    model_save_path: str = "models/market_making_ppo",
    use_custom_policy: bool = True,
    start_time: datetime = None,
    end_time: datetime = None,
):
    """
    training market making agent using PPO
       Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        batch_size: Batch size for updates
        n_steps: Steps per environment before update
        n_epochs: Number of optimization epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping norm
        features_dim: Feature extractor dimension
        normalize_input: Whether to normalize inputs
        dropout: Dropout rate
        save_freq: Frequency of model checkpoints
        eval_freq: Frequency of evaluation
        eval_episodes: Number of episodes for evaluation
        tensorboard_log: TensorBoard log directory
        model_save_path: Path to save model
        use_custom_policy: Whether to use custom policy
        start_time: Start time for training data
        end_time: End time for training data
    """

    logger.info("-"*40)
    logger.info("Starting Market Making RL Training")
    logger.info("-"*40)

    #create directories
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    if start_time is None:
        end_time = datetime.now()
 
        start_time = end_time - timedelta(days=7)  # Use 7 days for better diversity
    

    logger.info(f"Training from {start_time} to {end_time}")
    logger.info(f"Total training timesteps: {total_timesteps}")
    logger.info(f"Number of parallel environments: {n_envs}")

    #create environments
    logger.info("Creating training environments...")
    train_env = create_training_envs(
        n_envs = n_envs,
        start_time = start_time,
        end_time = end_time,
        random_start = True
    )

    #create eval environment - use random_start=True to match training
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(0, 42, start_time, end_time, True)]) 

    #policy kwargs
    policy_kwargs = {}
    if use_custom_policy:
        policy_kwargs = {
            'features_dim': features_dim,
            'normalize_input': normalize_input,
            'dropout': dropout,
            'net_arch': [{'pi': [256, 256], 'vf': [256, 256]}]
        }
        policy_class = MarketMakingPolicy
    else:
        policy_class = "MlpPolicy"
    

    #create model
    logger.info("Initializing PPO model...")
    model = PPO(
        policy = policy_class,
        env = train_env,
        learning_rate = learning_rate,
        n_steps = n_steps,
        batch_size = batch_size,
        n_epochs = n_epochs,
        gamma = gamma,
        gae_lambda = gae_lambda,
        clip_range = clip_range,
        ent_coef = ent_coef,
        vf_coef = vf_coef,  # 0.5 default - good for stable value learning
        max_grad_norm = max_grad_norm,
        policy_kwargs = policy_kwargs,
        tensorboard_log = tensorboard_log,
        verbose = 1,
        device = "auto",
        
        clip_range_vf = 0.2,  
        normalize_advantage = True,  
    )

    #setup callbacks
    callbacks = []

    #checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq = save_freq,
        save_path = Path(model_save_path).parent / "checkpoints",
        name_prefix = "ppo_market_making",
        save_replay_buffer = True,
        save_vecnormalize = True,
    )
    callbacks.append(checkpoint_callback)

    #evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = Path(model_save_path).parent / "best_model",
        log_path = Path(model_save_path).parent / "eval_logs",
        eval_freq = eval_freq,
        n_eval_episodes = eval_episodes,
        deterministic = True,
        render = False
    )
    callbacks.append(eval_callback)

    #metrics callback
    metrics_callback = TrainingMetricsCallback(verbose = 1)
    callbacks.append(metrics_callback)


    #Train

    logger.info("Starting training...")
    logger.info(f"Model will be saved in {model_save_path}")
    logger.info(f"TensorBoard logs will be saved in {tensorboard_log}")
    logger.info("-"*40)


    try:
        model.learn(
            total_timesteps = total_timesteps,
            callback = callbacks,
            progress_bar = True
        )

        #save final model
        final_path = f"{model_save_path}_final"
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        final_path = f"{model_save_path}_interrupted"
        model.save(final_path)
        logger.info(f"Interrupted model saved to {final_path}")

    finally:
        logger.info("Training completed")
        logger.info("-"*40)
        train_env.close()
        eval_env.close()


def main():
    """
    main training function
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train market making RL agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (reduced from 3e-4 for more stability in longer training)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--n-epochs", type=int, default=15, help="Epochs per update (increased from 10 to allow more gradual policy updates and reduce clip fraction)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient (increased for better exploration)")
    parser.add_argument("--features-dim", type=int, default=256, help="Feature extractor dimension")
    parser.add_argument("--no-custom-policy", action="store_true", help="Use default MLP policy")
    parser.add_argument("--model-path", type=str, default="models/market_making_ppo", help="Model save path")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data to use")

    args = parser.parse_args()

    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)


    train_agent(
        total_timesteps = args.timesteps,
        n_envs = args.n_envs,
        learning_rate = args.lr,
        batch_size = args.batch_size,
        n_steps = args.n_steps,
        n_epochs = args.n_epochs,
        gamma = args.gamma,
        clip_range = args.clip_range,
        ent_coef = args.ent_coef,
        features_dim = args.features_dim,
        use_custom_policy = not args.no_custom_policy,
        start_time = start_time,
        end_time = end_time
    )


if __name__ == "__main__":
    main()

