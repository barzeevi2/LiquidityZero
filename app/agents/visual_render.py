#!/usr/bin/env python3
"""
Visual rendering script for MarketMakingEnv
Creates real-time charts showing market state, portfolio performance, and trading activity
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import asyncio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.sim.market_env import MarketMakingEnv
from app.sim.market_simulator import MarketSimulator


def create_mock_snapshots():
    """Create mock orderbook snapshots"""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    snapshots = []
    
    for i in range(200):
        price_base = 50000.0 + np.sin(i / 10) * 100 + (i % 20) * 2
        snapshot = {
            'datetime': (base_time + timedelta(seconds=i)).isoformat(),
            'timestamp': (base_time + timedelta(seconds=i)).timestamp(),
            'symbol': 'BTC/USDT',
            'bids': [
                [price_base, 1.5 + np.random.random() * 0.5],
                [price_base - 1, 2.0 + np.random.random() * 0.5],
                [price_base - 2, 3.0 + np.random.random() * 0.5]
            ],
            'asks': [
                [price_base + 1, 1.2 + np.random.random() * 0.5],
                [price_base + 2, 2.5 + np.random.random() * 0.5],
                [price_base + 3, 1.8 + np.random.random() * 0.5]
            ],
            'best_bid': price_base,
            'best_ask': price_base + 1,
            'spread': 1.0,
            'spread_pct': 0.002
        }
        snapshots.append(snapshot)
    
    return snapshots


async def setup_mock_simulator(env):
    """Setup mock simulator"""
    snapshots = create_mock_snapshots()
    mock_simulator = MarketSimulator(symbol="BTC/USDT", initial_cash=10000.0)
    mock_simulator.snapshots = snapshots
    
    async def mock_load(start_time, end_time):
        pass
    
    mock_simulator.load_historical_data = mock_load
    env.simulator = mock_simulator
    return mock_simulator


def run_episode_with_visualization(num_steps=50, save_animation=False):
    """Run episode and create visualizations"""
    
    # Create environment
    env = MarketMakingEnv(
        symbol="BTC/USDT",
        initial_cash=10000.0,
        max_steps=num_steps
    )
    
    # Setup mock data
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    # Data storage for plotting
    timesteps = []
    mid_prices = []
    portfolio_values = []
    cash_history = []
    position_history = []
    realized_pnl_history = []
    unrealized_pnl_history = []
    rewards_history = []
    spreads = []
    
    # Run episode
    print(f"Running episode for {num_steps} steps...")
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        state = env.simulator.get_current_state()
        mid_price = state['mid_price']
        portfolio = info['portfolio']
        
        timesteps.append(step)
        mid_prices.append(mid_price if mid_price else 0)
        portfolio_values.append(portfolio.get('total_value', 10000.0))
        cash_history.append(portfolio.get('cash', 10000.0))
        position_history.append(portfolio.get('position', 0.0))
        realized_pnl_history.append(portfolio.get('realized_pnl', 0.0))
        unrealized_pnl_history.append(portfolio.get('unrealized_pnl', 0.0))
        rewards_history.append(reward)
        spreads.append(state.get('spread', 0) if state.get('spread') else 0)
        
        if terminated or truncated:
            break
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price Chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timesteps, mid_prices, 'b-', linewidth=2, label='Mid Price')
    ax1.set_title('Market Price Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Portfolio Value
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timesteps, portfolio_values, 'g-', linewidth=2, label='Total Value')
    ax2.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial Value')
    ax2.set_title('Portfolio Value', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value (USDT)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Cash and Position
    ax3 = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(timesteps, cash_history, 'b-', linewidth=2, label='Cash')
    line2 = ax3_twin.plot(timesteps, position_history, 'r-', linewidth=2, label='Position')
    ax3.set_title('Cash & Position', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Cash (USDT)', color='b')
    ax3_twin.set_ylabel('Position (BTC)', color='r')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. PnL
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(timesteps, realized_pnl_history, 'g-', linewidth=2, label='Realized PnL', alpha=0.7)
    ax4.plot(timesteps, unrealized_pnl_history, 'b--', linewidth=2, label='Unrealized PnL', alpha=0.7)
    total_pnl = [r + u for r, u in zip(realized_pnl_history, unrealized_pnl_history)]
    ax4.plot(timesteps, total_pnl, 'k-', linewidth=2, label='Total PnL')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_title('Profit & Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('PnL (USDT)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Rewards and Spread
    ax5 = fig.add_subplot(gs[2, 1])
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(timesteps, rewards_history, 'purple', linewidth=2, label='Reward', alpha=0.7)
    line2 = ax5_twin.plot(timesteps, spreads, 'orange', linewidth=1, label='Spread', alpha=0.5)
    ax5.set_title('Rewards & Spread', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Reward', color='purple')
    ax5_twin.set_ylabel('Spread (USDT)', color='orange')
    ax5.tick_params(axis='y', labelcolor='purple')
    ax5_twin.tick_params(axis='y', labelcolor='orange')
    ax5.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    
    # Add summary text
    final_value = portfolio_values[-1] if portfolio_values else 10000.0
    total_return = ((final_value - 10000.0) / 10000.0) * 100
    total_reward = sum(rewards_history)
    final_pnl = total_pnl[-1] if total_pnl else 0.0
    
    summary_text = f"""
    Episode Summary:
    • Total Steps: {len(timesteps)}
    • Final Portfolio Value: ${final_value:.2f}
    • Total Return: {total_return:.2f}%
    • Total Reward: {total_reward:.6f}
    • Final PnL: ${final_pnl:.2f}
    • Avg Spread: ${np.mean(spreads):.2f}
    """
    
    fig.suptitle('Market Making Environment - Episode Visualization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add text box with summary
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_animation:
        plt.savefig('episode_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to episode_visualization.png")
    else:
        plt.show()
    
    return fig


def realtime_visualization(num_steps=30):
    """Real-time visualization (updates as episode runs)"""
    
    # Create environment
    env = MarketMakingEnv(
        symbol="BTC/USDT",
        initial_cash=10000.0,
        max_steps=num_steps
    )
    
    # Setup mock data
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    # Initialize plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Real-Time Market Making Visualization', fontsize=16, fontweight='bold')
    
    # Data storage
    max_points = num_steps
    timesteps = deque(maxlen=max_points)
    mid_prices = deque(maxlen=max_points)
    portfolio_values = deque(maxlen=max_points)
    rewards = deque(maxlen=max_points)
    positions = deque(maxlen=max_points)
    
    # Initialize plots
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    line1 = ax1.plot([], [], 'b-', linewidth=2)[0]
    line2 = ax2.plot([], [], 'g-', linewidth=2)[0]
    line3 = ax3.plot([], [], 'r-', linewidth=2)[0]
    line4 = ax4.plot([], [], 'purple', linewidth=2)[0]
    
    ax1.set_title('Mid Price')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Portfolio Value')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value (USDT)')
    ax2.axhline(y=10000, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Position')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Position (BTC)')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Reward')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Reward')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    def update(frame):
        if frame >= num_steps:
            return line1, line2, line3, line4
        
        # Take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get data
        state = env.simulator.get_current_state()
        mid_price = state['mid_price'] or 0
        portfolio = info['portfolio']
        portfolio_value = portfolio.get('total_value', 10000.0)
        position = portfolio.get('position', 0.0)
        
        # Update data
        timesteps.append(frame)
        mid_prices.append(mid_price)
        portfolio_values.append(portfolio_value)
        rewards.append(reward)
        positions.append(position)
        
        # Update plots
        line1.set_data(list(timesteps), list(mid_prices))
        line2.set_data(list(timesteps), list(portfolio_values))
        line3.set_data(list(timesteps), list(positions))
        line4.set_data(list(timesteps), list(rewards))
        
        # Update axis limits
        if timesteps:
            ax1.set_xlim(0, max(timesteps) + 1)
            ax1.set_ylim(min(mid_prices) * 0.999, max(mid_prices) * 1.001)
            ax2.set_xlim(0, max(timesteps) + 1)
            ax3.set_xlim(0, max(timesteps) + 1)
            ax4.set_xlim(0, max(timesteps) + 1)
        
        if terminated or truncated:
            return line1, line2, line3, line4
        
        return line1, line2, line3, line4
    
    print("Starting real-time visualization...")
    print("Close the window to stop.")
    
    ani = FuncAnimation(fig, update, interval=500, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return ani


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual rendering for MarketMakingEnv")
    parser.add_argument(
        '--mode',
        choices=['static', 'realtime'],
        default='static',
        help='Visualization mode: static (save/show plot) or realtime (animated)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of steps to run'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save plot to file (for static mode)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'realtime':
            realtime_visualization(num_steps=args.steps)
        else:
            run_episode_with_visualization(num_steps=args.steps, save_animation=args.save)
        
        print("\n✓ Visualization complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

