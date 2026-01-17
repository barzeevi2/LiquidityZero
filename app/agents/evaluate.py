"""
evaluation script for trained market making agent 
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


from app.sim.market_env import MarketMakingEnv
from app.core.config import settings



def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
):
    """
    evaluate trained market making agent

    takes in a 
    model path: path to saved model
    n_episodes: number of episodes to evaluate
    render: whether to render episodes
    deterministic: whether to use deterministic policy
    """
    
    print("-"*40)
    print(f"Evaluating model: {model_path}")
    print(f"Number of episodes: {n_episodes}")
    print("-"*40)

    model = PPO.load(model_path)

    #create evaluation environment
    def make_env():
        env = MarketMakingEnv(
            symbol=settings.SYMBOL,
            initial_cash = settings.INITIAL_CASH,
            max_steps = settings.MAX_STEPS
        )
        env = Monitor(env)
        return env
    

    env = DummyVecEnv([make_env])

    #evaluation metrics

    episode_rewards = []
    episode_lengths = []
    episode_pnls = []
    episode_returns = []

    print(f"running {n_episodes} evaluation episodes...")
    print("-"*40)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        if render:
            print(f"\nEpisode {episode + 1}")
            print("-"*40)
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

            if render and step_count % 100 == 0:
                print(f"Step {step_count}: Reward {reward[0]:.2f}")
        

        #final metrics
        if 'episode' in info[0]:
            episode_info = info[0]['episode']
            episode_rewards.append(episode_info['r'])
            episode_lengths.append(episode_info['l'])
        
        if 'portfolio' in info[0]:
            portfolio = info[0]['portfolio']
            episode_pnls.append(portfolio.get('total_pnl', 0.0))
            episode_returns.append(portfolio.get('return_pct', 0.0))
        
        if render:
            print(f"Episode {episode + 1} completed:")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Steps: {step_count}")
            if portfolio:
                print(f"  Total PnL: ${portfolio.get('total_pnl', 0):.2f}")
                print(f"  Return: {portfolio.get('return_pct', 0):.2f}%")
    

    #summary statistics
     # Print summary statistics
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(episode_rewards):.4f}")
    print(f"  Std:  {np.std(episode_rewards):.4f}")
    print(f"  Min:  {np.min(episode_rewards):.4f}")
    print(f"  Max:  {np.max(episode_rewards):.4f}")
    
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {np.mean(episode_lengths):.1f}")
    print(f"  Std:  {np.std(episode_lengths):.1f}")
    
    if episode_pnls:
        print(f"\nProfit & Loss:")
        print(f"  Mean: ${np.mean(episode_pnls):.2f}")
        print(f"  Std:  ${np.std(episode_pnls):.2f}")
        print(f"  Min:  ${np.min(episode_pnls):.2f}")
        print(f"  Max:  ${np.max(episode_pnls):.2f}")
        print(f"  Win Rate: {np.mean(np.array(episode_pnls) > 0) * 100:.1f}%")
    

    if episode_returns:
        print(f"\nReturns:")
        print(f"  Mean: {np.mean(episode_returns):.2f}%")
        print(f"  Std:  {np.std(episode_returns):.2f}%")
        print(f"  Sharpe Ratio: {np.mean(episode_returns) / (np.std(episode_returns) + 1e-8):.2f}")
    
    print("="*70)
    
    env.close()
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'pnls': episode_pnls,
        'returns': episode_returns
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate market making agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        render=args.render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()