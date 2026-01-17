"""
Analyze training results from monitor files and training output
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def analyze_monitor_files(log_dir="logs/training"):
    """Analyze monitor CSV files"""
    monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    
    if not monitor_files:
        print(f"No monitor files found in {log_dir}")
        return None
    
    all_rewards = []
    all_lengths = []
    
    for monitor_file in monitor_files:
        try:
            # Skip header lines
            df = pd.read_csv(monitor_file, skiprows=1)
            if 'r' in df.columns:
                all_rewards.extend(df['r'].tolist())
            if 'l' in df.columns:
                all_lengths.extend(df['l'].tolist())
        except Exception as e:
            print(f"Error reading {monitor_file}: {e}")
    
    if not all_rewards:
        return None
    
    return {
        'rewards': np.array(all_rewards),
        'lengths': np.array(all_lengths) if all_lengths else None
    }

def extract_eval_metrics_from_log(log_file="train_output.txt"):
    """Extract evaluation metrics from training log"""
    if not os.path.exists(log_file):
        return None
    
    eval_metrics = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Eval num_timesteps=" in line and "episode_reward=" in line:
                # Parse: Eval num_timesteps=40000, episode_reward=0.02 +/- 0.00
                parts = line.split(',')
                timesteps = None
                reward = None
                
                for part in parts:
                    if 'num_timesteps=' in part:
                        timesteps = int(part.split('=')[1].strip())
                    if 'episode_reward=' in part:
                        reward_str = part.split('=')[1].strip().split()[0]
                        reward = float(reward_str)
                
                if timesteps and reward is not None:
                    eval_metrics.append({'timesteps': timesteps, 'reward': reward})
    
    return eval_metrics

def plot_training_progress(eval_metrics, output_dir="logs/analysis"):
    """Plot training progress"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not eval_metrics:
        print("No evaluation metrics to plot")
        return
    
    timesteps = [m['timesteps'] for m in eval_metrics]
    rewards = [m['reward'] for m in eval_metrics]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, rewards, 'b-', linewidth=2, label='Eval Reward')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress: Evaluation Reward over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "training_progress.png")
    plt.savefig(plot_path)
    print(f"Training progress plot saved to {plot_path}")
    plt.close()

def main():
    print("="*70)
    print("TRAINING RESULTS ANALYSIS")
    print("="*70)
    
    # Analyze monitor files
    print("\n1. Analyzing monitor files...")
    monitor_data = analyze_monitor_files()
    
    if monitor_data:
        rewards = monitor_data['rewards']
        print(f"\nEpisode Rewards (from monitor files):")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Std:  {np.std(rewards):.4f}")
        print(f"  Min:  {np.min(rewards):.4f}")
        print(f"  Max:  {np.max(rewards):.4f}")
        print(f"  Median: {np.median(rewards):.4f}")
        
        positive_count = np.sum(rewards > 0)
        print(f"  Positive episodes: {positive_count}/{len(rewards)} ({100*positive_count/len(rewards):.1f}%)")
        
        # Analyze recent performance (last 100 episodes)
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            print(f"\n  Recent Performance (last 100 episodes):")
            print(f"    Mean: {np.mean(recent_rewards):.4f}")
            print(f"    Positive: {np.sum(recent_rewards > 0)}/100 ({100*np.sum(recent_rewards > 0)/100:.1f}%)")
    
    # Extract evaluation metrics
    print("\n2. Extracting evaluation metrics from training log...")
    eval_metrics = extract_eval_metrics_from_log()
    
    if eval_metrics:
        print(f"\nEvaluation Metrics (during training):")
        print(f"  Total evaluations: {len(eval_metrics)}")
        
        rewards = [m['reward'] for m in eval_metrics]
        print(f"  Mean reward: {np.mean(rewards):.4f}")
        print(f"  Final reward (1M timesteps): {rewards[-1]:.4f}")
        print(f"  Best reward: {np.max(rewards):.4f}")
        print(f"  Worst reward: {np.min(rewards):.4f}")
        
        # Find trend
        if len(rewards) >= 10:
            early_avg = np.mean(rewards[:10])
            late_avg = np.mean(rewards[-10:])
            print(f"\n  Improvement:")
            print(f"    Early (first 10): {early_avg:.4f}")
            print(f"    Late (last 10): {late_avg:.4f}")
            print(f"    Change: {late_avg - early_avg:+.4f}")
        
        # Plot progress
        plot_training_progress(eval_metrics)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if eval_metrics:
        final_reward = eval_metrics[-1]['reward']
        if final_reward > 0:
            print(f"✓ Final evaluation reward is POSITIVE: {final_reward:.2f}")
        else:
            print(f"✗ Final evaluation reward is NEGATIVE: {final_reward:.2f}")
    
    if monitor_data:
        win_rate = 100 * np.sum(monitor_data['rewards'] > 0) / len(monitor_data['rewards'])
        if win_rate > 50:
            print(f"✓ Win rate: {win_rate:.1f}% (more positive than negative episodes)")
        else:
            print(f"✗ Win rate: {win_rate:.1f}% (more negative than positive episodes)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

