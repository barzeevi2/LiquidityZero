"""
Integration test script for agents module
Tests FeatureEngine and Policy integration with MarketMakingEnv
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.features import MarketFeatureEngine, FeatureNormalizer
from app.sim.market_env import MarketMakingEnv
from gymnasium import spaces

# Import torch-dependent modules conditionally
try:
    import torch
    from app.agents.policy import MarketMakingFeatureExtractor, MarketMakingPolicy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_feature_engine_with_env():
    """Test MarketFeatureEngine with real environment observations"""
    print("\n" + "="*60)
    print("Testing MarketFeatureEngine with MarketMakingEnv")
    print("="*60)
    
    # Initialize environment
    env = MarketMakingEnv(
        symbol="BTC/USDT",
        initial_cash=10000.0,
        max_steps=10,  # Short episode for testing
        price_tick_size=0.01
    )
    
    # Initialize feature engine
    feature_engine = MarketFeatureEngine(enable_normalization=True)
    
    # Reset environment - need to setup mock data first
    print("\n1. Setting up environment with mock data...")
    try:
        # Create mock snapshots for testing
        from datetime import datetime, timedelta
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        snapshots = []
        for i in range(50):
            price_base = 50000.0 + (i % 10) * 5
            snapshot = {
                'datetime': (base_time + timedelta(seconds=i)).isoformat(),
                'timestamp': (base_time + timedelta(seconds=i)).timestamp(),
                'symbol': 'BTC/USDT',
                'bids': [[price_base, 1.5], [price_base - 1, 2.0]],
                'asks': [[price_base + 1, 1.2], [price_base + 2, 2.5]],
            }
            snapshots.append(snapshot)
        
        # Setup simulator with mock data
        from app.sim.market_simulator import MarketSimulator
        simulator = MarketSimulator(symbol="BTC/USDT", initial_cash=10000.0)
        simulator.snapshots = snapshots
        env.simulator = simulator
        env.simulator.reset(start_index=0)
        
        obs, info = env.reset(options={'start_index': 0})
        print(f"   ✓ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Extract features before normalization
    print("\n2. Extracting features (before normalization update)...")
    try:
        features_before = feature_engine.extract_advanced_features(obs)
        print(f"   ✓ Feature extraction successful")
        print(f"   Features shape: {features_before.shape}")
        print(f"   Features should equal observation (not updated yet): {np.allclose(features_before, obs)}")
    except Exception as e:
        print(f"   ✗ Feature extraction failed: {e}")
        return False
    
    # Update statistics with observations
    print("\n3. Updating normalization statistics...")
    try:
        # Collect a batch of observations
        observations_batch = [obs]
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            observations_batch.append(obs)
        
        observations_array = np.array(observations_batch)
        feature_engine.update_statistics(observations_array)
        print(f"   ✓ Statistics updated")
        print(f"   Normalizer count: {feature_engine.normalizer.count}")
        print(f"   Mean range: [{feature_engine.normalizer.feature_mean.min():.4f}, "
              f"{feature_engine.normalizer.feature_mean.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Statistics update failed: {e}")
        return False
    
    # Extract features after normalization
    print("\n4. Extracting normalized features...")
    try:
        features_after = feature_engine.extract_advanced_features(obs)
        print(f"   ✓ Normalized feature extraction successful")
        print(f"   Features shape: {features_after.shape}")
        print(f"   Features mean: {features_after.mean():.6f}")
        print(f"   Features std: {features_after.std():.6f}")
        print(f"   Features differ from original: {not np.allclose(features_after, obs)}")
    except Exception as e:
        print(f"   ✗ Normalized feature extraction failed: {e}")
        return False
    
    print("\n✓ FeatureEngine integration test passed!")
    return True


def test_policy_with_env():
    """Test MarketMakingPolicy with real environment"""
    if not TORCH_AVAILABLE:
        print("\n⚠ PyTorch not available, skipping policy test")
        return None
    
    print("\n" + "="*60)
    print("Testing MarketMakingPolicy with MarketMakingEnv")
    print("="*60)
    
    # Initialize environment
    env = MarketMakingEnv(
        symbol="BTC/USDT",
        initial_cash=10000.0,
        max_steps=5,  # Very short for testing
        price_tick_size=0.01
    )
    
    # Create observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space
    
    # Create learning rate schedule
    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining
    
    print("\n1. Initializing MarketMakingPolicy...")
    try:
        policy = MarketMakingPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_dim=128,  # Smaller for testing
            normalize_input=True
        )
        print(f"   ✓ Policy initialized successfully")
        print(f"   Feature extractor type: {type(policy.features_extractor).__name__}")
        print(f"   Features dim: {policy.features_extractor.features_dim}")
    except Exception as e:
        print(f"   ✗ Policy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Reset environment - need to setup mock data first
    print("\n2. Setting up environment with mock data...")
    try:
        # Create mock snapshots for testing
        from datetime import datetime, timedelta
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        snapshots = []
        for i in range(20):
            price_base = 50000.0 + (i % 10) * 5
            snapshot = {
                'datetime': (base_time + timedelta(seconds=i)).isoformat(),
                'timestamp': (base_time + timedelta(seconds=i)).timestamp(),
                'symbol': 'BTC/USDT',
                'bids': [[price_base, 1.5], [price_base - 1, 2.0]],
                'asks': [[price_base + 1, 1.2], [price_base + 2, 2.5]],
            }
            snapshots.append(snapshot)
        
        # Setup simulator with mock data
        from app.sim.market_simulator import MarketSimulator
        simulator = MarketSimulator(symbol="BTC/USDT", initial_cash=10000.0)
        simulator.snapshots = snapshots
        env.simulator = simulator
        env.simulator.reset(start_index=0)
        
        obs, info = env.reset(options={'start_index': 0})
        print(f"   ✓ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
    except Exception as e:
        print(f"   ✗ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test feature extraction
    print("\n3. Testing feature extraction...")
    try:
        import torch
        policy.eval()  # Set to eval mode for single-sample inference (BatchNorm requirement)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = policy.extract_features(obs_tensor)
        print(f"   ✓ Feature extraction successful")
        print(f"   Input shape: {obs_tensor.shape}")
        print(f"   Output features shape: {features.shape}")
        print(f"   Features mean: {features.mean().item():.6f}")
        print(f"   Features std: {features.std().item():.6f}")
    except Exception as e:
        print(f"   ✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test action prediction
    print("\n4. Testing action prediction...")
    try:
        import torch
        action, _ = policy.predict(obs, deterministic=True)
        print(f"   ✓ Action prediction successful")
        print(f"   Action: {action}")
        print(f"   Action shape: {action.shape}")
        print(f"   Action valid: {all(0 <= action[i] < action_space.nvec[i] for i in range(4))}")
    except Exception as e:
        print(f"   ✗ Action prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test value prediction
    print("\n5. Testing value prediction...")
    try:
        import torch
        policy.eval()  # Ensure eval mode
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = policy.predict_values(obs_tensor)
        print(f"   ✓ Value prediction successful")
        print(f"   Value: {value.item():.6f}")
    except Exception as e:
        print(f"   ✗ Value prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full episode
    print("\n6. Testing full episode interaction...")
    try:
        obs, info = env.reset(options={'start_index': 0})
        total_reward = 0.0
        
        for step in range(3):
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"   ✓ Episode interaction successful")
        print(f"   Steps completed: {step + 1}")
        print(f"   Total reward: {total_reward:.6f}")
    except Exception as e:
        print(f"   ✗ Episode interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Policy integration test passed!")
    return True


def test_feature_extractor_standalone():
    """Test MarketMakingFeatureExtractor standalone"""
    if not TORCH_AVAILABLE:
        print("\n⚠ PyTorch not available, skipping feature extractor test")
        return None
    
    print("\n" + "="*60)
    print("Testing MarketMakingFeatureExtractor Standalone")
    print("="*60)
    
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(52,),
        dtype=np.float32
    )
    
    print("\n1. Initializing feature extractor...")
    try:
        extractor = MarketMakingFeatureExtractor(
            observation_space=observation_space,
            features_dim=128,
            normalize_input=True,
            dropout=0.1
        )
        print(f"   ✓ Feature extractor initialized")
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        return False
    
    print("\n2. Testing forward pass with 2D tensor...")
    try:
        import torch
        batch_size = 8
        observations = torch.randn(batch_size, 52)
        output = extractor.forward(observations)
        print(f"   ✓ Forward pass successful")
        print(f"   Input shape: {observations.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == (batch_size, 128)
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    print("\n3. Testing forward pass with 3D tensor...")
    try:
        extractor.eval()  # Eval mode for 3D tensors
        batch_size, seq_len = 4, 10
        observations = torch.randn(batch_size, seq_len, 52)
        # 3D tensor support: flattens sequence dimension, processes, then reshapes
        output = extractor.forward(observations)
        print(f"   ✓ 3D forward pass successful")
        print(f"   Input shape: {observations.shape}")
        print(f"   Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, 128)
    except Exception as e:
        print(f"   ✗ 3D forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ FeatureExtractor standalone test passed!")
    return True


def test_feature_normalizer_standalone():
    """Test FeatureNormalizer standalone"""
    print("\n" + "="*60)
    print("Testing FeatureNormalizer Standalone")
    print("="*60)
    
    print("\n1. Initializing normalizer...")
    try:
        normalizer = FeatureNormalizer(feature_dim=52, momentum=0.99)
        print(f"   ✓ Normalizer initialized")
        print(f"   Feature dim: 52")
        print(f"   Momentum: {normalizer.momentum}")
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        return False
    
    print("\n2. Testing update with batch...")
    try:
        batch = np.random.randn(100, 52).astype(np.float32)
        normalizer.update(batch)
        print(f"   ✓ Update successful")
        print(f"   Count: {normalizer.count}")
        print(f"   Mean range: [{normalizer.feature_mean.min():.4f}, "
              f"{normalizer.feature_mean.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Update failed: {e}")
        return False
    
    print("\n3. Testing normalization...")
    try:
        features = np.random.randn(52).astype(np.float32)
        normalized = normalizer.normalize(features)
        print(f"   ✓ Normalization successful")
        print(f"   Normalized mean: {normalized.mean():.6f}")
        print(f"   Normalized std: {normalized.std():.6f}")
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
        return False
    
    print("\n✓ FeatureNormalizer standalone test passed!")
    return True


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("AGENTS MODULE INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: FeatureNormalizer standalone
    try:
        results.append(("FeatureNormalizer Standalone", test_feature_normalizer_standalone()))
    except Exception as e:
        print(f"\n✗ FeatureNormalizer test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FeatureNormalizer Standalone", False))
    
    # Test 2: FeatureExtractor standalone
    try:
        results.append(("FeatureExtractor Standalone", test_feature_extractor_standalone()))
    except Exception as e:
        print(f"\n✗ FeatureExtractor test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FeatureExtractor Standalone", False))
    
    # Test 3: FeatureEngine with environment
    try:
        results.append(("FeatureEngine with Env", test_feature_engine_with_env()))
    except Exception as e:
        print(f"\n✗ FeatureEngine test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FeatureEngine with Env", False))
    
    # Test 4: Policy with environment (requires torch)
    try:
        results.append(("Policy with Env", test_policy_with_env()))
    except Exception as e:
        print(f"\n✗ Policy test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Policy with Env", False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
            passed += 1
        elif result is False:
            status = "✗ FAILED"
            failed += 1
        else:
            status = "⊘ SKIPPED"
            skipped += 1
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("="*60)
    
    if failed > 0:
        print("\n⚠ Some tests failed. Check output above for details.")
        return 1
    elif passed == len(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests were skipped.")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

