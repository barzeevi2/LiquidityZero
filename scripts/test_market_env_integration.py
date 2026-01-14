#!/usr/bin/env python3
"""
Integration test script for MarketMakingEnv
Tests realistic environment scenarios, full episode runs, and rendering
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sim.market_env import MarketMakingEnv
from app.sim.exceptions import SimulationError
import asyncio


def create_mock_snapshots():
    """Create mock orderbook snapshots for testing"""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    snapshots = []
    
    # Create 100 snapshots with varying prices (simulating market movement)
    for i in range(100):
        price_base = 50000.0 + (i % 20) * 5  # Oscillating price
        snapshot = {
            'datetime': (base_time + timedelta(seconds=i)).isoformat(),
            'timestamp': (base_time + timedelta(seconds=i)).timestamp(),
            'symbol': 'BTC/USDT',
            'bids': [
                [price_base, 1.5],
                [price_base - 1, 2.0],
                [price_base - 2, 3.0]
            ],
            'asks': [
                [price_base + 1, 1.2],
                [price_base + 2, 2.5],
                [price_base + 3, 1.8]
            ],
            'best_bid': price_base,
            'best_ask': price_base + 1,
            'spread': 1.0,
            'spread_pct': 0.002
        }
        snapshots.append(snapshot)
    
    return snapshots


async def setup_mock_simulator(env):
    """Setup mock simulator with test data"""
    from unittest.mock import MagicMock, AsyncMock, patch
    from app.sim.market_simulator import MarketSimulator
    
    snapshots = create_mock_snapshots()
    
    # Create a real simulator but with mocked database connection
    mock_simulator = MarketSimulator(symbol="BTC/USDT", initial_cash=10000.0)
    mock_simulator.snapshots = snapshots
    
    # Mock the database load
    async def mock_load(start_time, end_time):
        # Already have snapshots, just mark as loaded
        pass
    
    mock_simulator.load_historical_data = mock_load
    env.simulator = mock_simulator
    
    return mock_simulator


def test_basic_episode():
    """Test a basic episode run"""
    print("\n" + "="*60)
    print("Test 1: Basic Episode Run")
    print("="*60)
    
    env = MarketMakingEnv(
        symbol="BTC/USDT",
        initial_cash=10000.0,
        max_steps=10,
        price_tick_size=0.01,
        max_quantity=1.0
    )
    
    # Setup mock data
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    
    # Reset environment
    obs, info = env.reset(options={'start_index': 0})
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")
    
    total_reward = 0.0
    for step in range(10):
        # Random action: place orders around mid price
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: reward={reward:.6f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"\nTotal reward: {total_reward:.6f}")
    print(f"Final portfolio value: ${info['portfolio'].get('total_value', 0):.2f}")
    print("✓ Basic episode test passed\n")


def test_action_space():
    """Test action space sampling and interpretation"""
    print("\n" + "="*60)
    print("Test 2: Action Space")
    print("="*60)
    
    env = MarketMakingEnv(
        price_tick_size=0.1,
        max_quantity=2.0,
        n_price_levels=21,
        n_quantity_levels=10
    )
    
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    mid_price = env.simulator.orderbook.get_mid_price()
    print(f"Mid price: {mid_price}")
    
    # Test different actions
    test_actions = [
        np.array([10, 5, 10, 5]),  # Mid price, half quantity
        np.array([0, 10, 20, 10]),  # Below mid bid, above mid ask, max quantity
        np.array([20, 0, 0, 0]),   # Only bid, no ask
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nAction {i + 1}: {action}")
        bid_offset = int(action[0]) - 10
        ask_offset = int(action[2]) - 10
        bid_price = mid_price + (bid_offset * env.price_tick_size)
        ask_price = mid_price + (ask_offset * env.price_tick_size)
        
        print(f"  Bid offset: {bid_offset} ticks -> Price: {bid_price:.2f}")
        print(f"  Ask offset: {ask_offset} ticks -> Price: {ask_price:.2f}")
        print(f"  Bid quantity index: {action[1]}, Ask quantity index: {action[3]}")
    
    print("✓ Action space test passed\n")


def test_observation_space():
    """Test observation space structure"""
    print("\n" + "="*60)
    print("Test 3: Observation Space")
    print("="*60)
    
    env = MarketMakingEnv()
    
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min: {obs.min():.4f}, max: {obs.max():.4f}")
    print(f"Observation mean: {obs.mean():.4f}, std: {obs.std():.4f}")
    
    # Check observation components
    state = env.simulator.get_current_state()
    print(f"\nState components:")
    print(f"  Best bid: {state['best_bid']}")
    print(f"  Best ask: {state['best_ask']}")
    print(f"  Mid price: {state['mid_price']}")
    print(f"  Spread: {state['spread']}")
    print(f"  Portfolio cash: {state['portfolio']['cash']}")
    
    assert obs.shape == (52,), f"Expected shape (52,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    
    print("✓ Observation space test passed\n")


def test_reward_calculation():
    """Test reward calculation over multiple steps"""
    print("\n" + "="*60)
    print("Test 4: Reward Calculation")
    print("="*60)
    
    env = MarketMakingEnv(initial_cash=10000.0)
    
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    rewards = []
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        portfolio = info['portfolio']
        print(f"Step {step + 1}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Realized PnL: ${portfolio.get('realized_pnl', 0):.2f}")
        print(f"  Unrealized PnL: ${portfolio.get('unrealized_pnl', 0):.2f}")
        print(f"  Total Value: ${portfolio.get('total_value', 0):.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nAverage reward: {np.mean(rewards):.6f}")
    print(f"Total reward: {sum(rewards):.6f}")
    print("✓ Reward calculation test passed\n")


def test_rendering():
    """Test rendering functionality"""
    print("\n" + "="*60)
    print("Test 5: Rendering")
    print("="*60)
    
    env = MarketMakingEnv()
    
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    print("Rendering initial state:")
    env.render()
    
    # Run a few steps and render
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nRendering after step {step + 1}:")
        env.render()
        
        if terminated or truncated:
            break
    
    print("✓ Rendering test passed\n")


def test_reset_options():
    """Test reset with different options"""
    print("\n" + "="*60)
    print("Test 6: Reset Options")
    print("="*60)
    
    env = MarketMakingEnv(max_steps=20)
    
    asyncio.run(setup_mock_simulator(env))
    
    # Test reset with start_index
    obs, info = env.reset(options={'start_index': 10})
    print(f"Reset with start_index=10: step={info['step']}")
    
    # Test reset with random_start
    obs, info = env.reset(options={'random_start': True})
    print(f"Reset with random_start: step={info['step']}")
    
    # Test default reset
    obs, info = env.reset()
    print(f"Default reset: step={info['step']}")
    
    print("✓ Reset options test passed\n")


def test_episode_termination():
    """Test episode termination conditions"""
    print("\n" + "="*60)
    print("Test 7: Episode Termination")
    print("="*60)
    
    env = MarketMakingEnv(max_steps=5)
    
    asyncio.run(setup_mock_simulator(env))
    env.simulator.reset(start_index=0)
    obs, info = env.reset(options={'start_index': 0})
    
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        print(f"Step {step_count}: terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    assert step_count <= env.max_steps, "Should not exceed max_steps"
    print("✓ Episode termination test passed\n")


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("MarketMakingEnv Integration Tests")
    print("="*60)
    
    try:
        test_basic_episode()
        test_action_space()
        test_observation_space()
        test_reward_calculation()
        test_rendering()
        test_reset_options()
        test_episode_termination()
        
        print("\n" + "="*60)
        print("All integration tests passed! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

