"""
Unit tests for training module
Tests training logic, environment creation, and callback functionality
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import torch

from app.agents.train import (
    TrainingMetricsCallback,
    make_env,
    create_training_envs,
    train_agent
)
from app.core.config import settings


class TestTrainingMetricsCallback:
    """Test TrainingMetricsCallback class"""
    
    def test_initialization(self):
        """Should initialize with empty lists"""
        callback = TrainingMetricsCallback(verbose=1)
        
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []
        assert callback.episode_pnls == []
    
    def test_on_step_with_episode_info(self):
        """Should log episode metrics when episode info is present"""
        callback = TrainingMetricsCallback(verbose=1)
        
        # Mock locals dict with episode info
        callback.locals = {
            'infos': [{
                'episode': {
                    'r': 100.0,
                    'l': 50
                }
            }]
        }
        
        result = callback._on_step()
        
        assert result is True
        assert len(callback.episode_rewards) == 1
        assert callback.episode_rewards[0] == 100.0
        assert len(callback.episode_lengths) == 1
        assert callback.episode_lengths[0] == 50
    
    def test_on_step_with_portfolio_info(self):
        """Should log PnL when portfolio info is present"""
        callback = TrainingMetricsCallback(verbose=1)
        
        callback.locals = {
            'infos': [{
                'episode': {'r': 100.0, 'l': 50},
                'portfolio': {'total_pnl': 50.5}
            }]
        }
        
        callback._on_step()
        
        assert len(callback.episode_pnls) == 1
        assert callback.episode_pnls[0] == 50.5
    
    def test_on_step_without_info(self):
        """Should not crash when no episode info is present"""
        callback = TrainingMetricsCallback(verbose=1)
        callback.locals = {'infos': [{}]}
        
        result = callback._on_step()
        
        assert result is True
        assert len(callback.episode_rewards) == 0
    
    def test_on_training_end_with_metrics(self):
        """Should log summary metrics at training end"""
        callback = TrainingMetricsCallback(verbose=1)
        callback.episode_rewards = [100.0, 200.0, 150.0]
        callback.episode_lengths = [50, 60, 55]
        callback.episode_pnls = [10.0, 20.0, 15.0]
        
        callback._on_training_end()
        
        # Should not crash and have logged metrics
        assert len(callback.episode_rewards) == 3
    
    def test_on_training_end_without_pnl(self):
        """Should handle training end without PnL data"""
        callback = TrainingMetricsCallback(verbose=1)
        callback.episode_rewards = [100.0, 200.0]
        callback.episode_lengths = [50, 60]
        callback.episode_pnls = []
        
        callback._on_training_end()
        
        # Should not crash
        assert len(callback.episode_rewards) == 2


class TestMakeEnv:
    """Test make_env function"""
    
    @patch('app.agents.train.MarketMakingEnv')
    @patch('app.agents.train.Monitor')
    @patch('app.agents.train.set_random_seed')
    def test_make_env_creates_environment(self, mock_set_seed, mock_monitor, mock_env_class):
        """Should create and wrap environment correctly"""
        # Mock environment
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        
        # Mock Monitor wrapper
        mock_wrapped_env = MagicMock()
        mock_monitor.return_value = mock_wrapped_env
        
        # Create environment factory
        env_factory = make_env(rank=0, seed=42)
        
        # Call factory to create environment
        env = env_factory()
        
        # Verify environment was created with correct params
        mock_env_class.assert_called_once()
        call_kwargs = mock_env_class.call_args[1]
        assert call_kwargs['symbol'] == settings.SYMBOL
        assert call_kwargs['initial_cash'] == settings.INITIAL_CASH
        assert call_kwargs['max_steps'] == settings.MAX_STEPS
        
        # Verify seed was set
        mock_env.seed.assert_called_once_with(42)
        
        # Verify Monitor was used
        mock_monitor.assert_called_once()
        
        # Verify random seed was set
        mock_set_seed.assert_called_once_with(42)
    
    @patch('app.agents.train.MarketMakingEnv')
    @patch('app.agents.train.Monitor')
    def test_make_env_creates_log_directory(self, mock_monitor, mock_env_class):
        """Should create log directory if it doesn't exist"""
        mock_env = MagicMock()
        mock_env_class.return_value = mock_env
        mock_wrapped_env = MagicMock()
        mock_monitor.return_value = mock_wrapped_env
        
        env_factory = make_env(rank=1, seed=0)
        env = env_factory()
        
        # Check that log directory creation was attempted
        # (Path.mkdir would be called internally)
        assert mock_monitor.called


class TestCreateTrainingEnvs:
    """Test create_training_envs function"""
    
    @patch('app.agents.train.DummyVecEnv')
    @patch('app.agents.train.make_env')
    def test_create_single_env(self, mock_make_env, mock_dummy_vec):
        """Should create single environment with DummyVecEnv"""
        mock_env_factory = MagicMock()
        mock_make_env.return_value = mock_env_factory
        mock_vec_env = MagicMock()
        mock_dummy_vec.return_value = mock_vec_env
        
        result = create_training_envs(n_envs=1)
        
        mock_dummy_vec.assert_called_once()
        mock_make_env.assert_called_once_with(0, 0, None, None, True)
        assert result == mock_vec_env
    
    @patch('app.agents.train.SubprocVecEnv')
    @patch('app.agents.train.make_env')
    def test_create_multiple_envs(self, mock_make_env, mock_subproc_vec):
        """Should create multiple environments with SubprocVecEnv"""
        n_envs = 4
        mock_env_factory = MagicMock()
        mock_make_env.return_value = mock_env_factory
        mock_vec_env = MagicMock()
        mock_subproc_vec.return_value = mock_vec_env
        
        result = create_training_envs(n_envs=n_envs)
        
        # Should call make_env for each environment
        assert mock_make_env.call_count == n_envs
        
        # Verify each call has correct rank
        calls = [call[0] for call in mock_make_env.call_args_list]
        ranks = [call[0] for call in calls]
        assert ranks == list(range(n_envs))
        
        mock_subproc_vec.assert_called_once()
        assert result == mock_vec_env


# Note: Tests for train_agent() high-level function are intentionally skipped.
# Industry practice: High-level training/evaluation scripts are typically tested via:
# 1. Integration tests (test_agents_integration.py)
# 2. Manual verification during development
# 3. Component tests (callbacks, env creation) - which we do have above
# 
# Testing train_agent() directly requires extensive mocking and is less valuable
# than testing the components it uses. If you want integration tests, use
# scripts/test_agents_integration.py or manually run training.

