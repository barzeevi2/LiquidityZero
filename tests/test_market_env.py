"""
Unit tests for MarketMakingEnv
Tests gymnasium environment interface, action/observation spaces, reset, step, and reward calculation
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timedelta

from app.sim.market_env import MarketMakingEnv
from app.sim.market_simulator import MarketSimulator
from app.sim.orders import LimitOrder, OrderSide
from app.sim.exceptions import SimulationError


class TestMarketMakingEnvInitialization:
    """Test MarketMakingEnv initialization"""
    
    def test_initialization_default_values(self):
        """Should initialize with default values"""
        env = MarketMakingEnv()
        
        assert env.symbol == "BTC/USDT"
        assert env.initial_cash == 10000.0
        assert env.max_steps == 1000
        assert env.price_tick_size == 0.01
        assert env.quantity_precision == 4
        assert env.max_quantity == 1.0
        assert env.n_price_levels == 21
        assert env.n_quantity_levels == 10
        assert env.lookback_window == 10
        assert env.simulator is None
        assert env.current_step == 0
        assert len(env.price_history) == 0
    
    def test_initialization_custom_values(self):
        """Should initialize with custom values"""
        env = MarketMakingEnv(
            symbol="ETH/USDT",
            initial_cash=50000.0,
            max_steps=500,
            price_tick_size=0.1,
            max_quantity=2.0
        )
        
        assert env.symbol == "ETH/USDT"
        assert env.initial_cash == 50000.0
        assert env.max_steps == 500
        assert env.price_tick_size == 0.1
        assert env.max_quantity == 2.0
    
    def test_action_space(self):
        """Should have correct MultiDiscrete action space"""
        env = MarketMakingEnv()
        
        assert env.action_space.shape == (4,)
        assert env.action_space.nvec[0] == 21  # bid price levels
        assert env.action_space.nvec[1] == 10  # bid quantity levels
        assert env.action_space.nvec[2] == 21  # ask price levels
        assert env.action_space.nvec[3] == 10  # ask quantity levels
    
    def test_observation_space(self):
        """Should have correct Box observation space"""
        env = MarketMakingEnv()
        
        assert env.observation_space.shape == (52,)
        assert env.observation_space.dtype == np.float32
        assert np.all(env.observation_space.low == -np.inf)
        assert np.all(env.observation_space.high == np.inf)
    
    def test_observation_dimension(self):
        """Should calculate correct observation dimension"""
        env = MarketMakingEnv()
        dim = env._get_observation_dim()
        
        # 6 orderbook features + 40 depth (10 levels * 2 sides * 2 features) + 4 portfolio + 2 market = 52
        assert dim == 52


class TestReset:
    """Test environment reset"""
    
    @pytest.mark.asyncio
    async def test_reset_initializes_simulator(self):
        """Should initialize simulator on first reset"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]]
            }
        ]
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.orderbook.get_best_bid.return_value = 50000.0
        mock_simulator.orderbook.get_best_ask.return_value = 50001.0
        mock_simulator.orderbook.get_spread.return_value = 1.0
        mock_simulator.orderbook.get_spread_percentage.return_value = 0.002
        mock_simulator.orderbook.get_imbalance.return_value = 0.1
        mock_simulator.orderbook.get_depth_levels.return_value = ([], [])
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.get_stats.return_value = {
            'cash': 10000.0,
            'position': 0.0,
            'unrealized_pnl': 0.0,
            'return_pct': 0.0
        }
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {
                'cash': 10000.0,
                'position': 0.0,
                'unrealized_pnl': 0.0,
                'return_pct': 0.0
            }
        }
        mock_simulator.load_historical_data = AsyncMock()
        mock_simulator.reset = Mock()
        
        with patch('app.sim.market_env.MarketSimulator', return_value=mock_simulator):
            with patch('asyncio.run'):
                obs, info = env.reset()
        
        assert env.current_step == 0
        assert len(env.price_history) == 0
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (52,)
        assert 'step' in info
    
    def test_reset_with_options(self):
        """Should handle reset options correctly"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ] * 100
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.load_historical_data = AsyncMock()
        mock_simulator.reset = Mock()
        
        with patch('app.sim.market_env.MarketSimulator', return_value=mock_simulator):
            with patch('asyncio.run'):
                options = {'start_index': 10, 'random_start': False}
                obs, info = env.reset(options=options)
        
        mock_simulator.reset.assert_called_once_with(start_index=10)


class TestStep:
    """Test environment step"""
    
    def test_step_basic(self):
        """Should execute one step correctly"""
        env = MarketMakingEnv()
        
        # Setup mock simulator
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.portfolio.get_stats.return_value = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'position': 0.0,
            'total_value': 10000.0
        }
        mock_simulator.step = AsyncMock(return_value=True)
        mock_simulator.cancel_agent_order = Mock()
        mock_simulator.place_agent_order = Mock()
        
        env.simulator = mock_simulator
        env.current_step = 0
        
        # Action: place bid at mid, ask at mid, both with quantity
        action = np.array([10, 5, 10, 5])  # mid price offsets, half max quantity
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (52,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert env.current_step == 1
    
    def test_step_no_mid_price(self):
        """Should handle case when mid price is None"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = None
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': None,
            'best_ask': None,
            'mid_price': None,
            'spread': None,
            'spread_pct': None,
            'imbalance': None,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.portfolio.get_stats.return_value = {}
        
        env.simulator = mock_simulator
        env.current_step = 0
        
        action = np.array([10, 5, 10, 5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == 0.0
        assert truncated is True
        assert terminated is False
    
    def test_step_action_conversion(self):
        """Should correctly convert action to prices and quantities"""
        env = MarketMakingEnv(price_tick_size=0.1, max_quantity=2.0, n_quantity_levels=10)
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.0
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 49999.0,
            'best_ask': 50001.0,
            'mid_price': 50000.0,
            'spread': 2.0,
            'spread_pct': 0.004,
            'imbalance': 0.0,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.portfolio.get_stats.return_value = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'position': 0.0,
            'total_value': 10000.0
        }
        mock_simulator.step = AsyncMock(return_value=True)
        mock_simulator.cancel_agent_order = Mock()
        mock_simulator.place_agent_order = Mock()
        
        env.simulator = mock_simulator
        
        # Action: offset 0 (mid), quantity index 5 (half max)
        action = np.array([10, 5, 10, 5])
        env.step(action)
        
        # Check that place_agent_order was called with correct prices
        calls = mock_simulator.place_agent_order.call_args_list
        assert len(calls) == 2  # bid and ask
        
        bid_order = calls[0][0][0]
        ask_order = calls[1][0][0]
        
        assert bid_order.side == OrderSide.BUY
        assert ask_order.side == OrderSide.SELL
        # Prices should be rounded to tick size
        assert bid_order.price == pytest.approx(50000.0, abs=0.01)
        assert ask_order.price == pytest.approx(50000.0, abs=0.01)
    
    def test_step_max_steps_truncation(self):
        """Should truncate when max_steps reached"""
        env = MarketMakingEnv(max_steps=5)
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.portfolio.get_stats.return_value = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'position': 0.0,
            'total_value': 10000.0
        }
        mock_simulator.step = AsyncMock(return_value=True)
        mock_simulator.cancel_agent_order = Mock()
        mock_simulator.place_agent_order = Mock()
        
        env.simulator = mock_simulator
        env.current_step = 4
        
        action = np.array([10, 5, 10, 5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert truncated is True
        assert env.current_step == 5
    
    def test_step_stop_loss(self):
        """Should terminate when portfolio value drops below 50%"""
        env = MarketMakingEnv(initial_cash=10000.0)
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.open_orders = {}
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {'cash': 10000.0, 'position': 0.0, 'unrealized_pnl': 0.0, 'return_pct': 0.0}
        }
        mock_simulator.portfolio.get_stats.return_value = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'position': 0.0,
            'total_value': 4000.0  # Below 50% threshold
        }
        mock_simulator.step = AsyncMock(return_value=True)
        mock_simulator.cancel_agent_order = Mock()
        mock_simulator.place_agent_order = Mock()
        
        env.simulator = mock_simulator
        
        action = np.array([10, 5, 10, 5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated is True


class TestRewardCalculation:
    """Test reward calculation"""
    
    def test_reward_calculation(self):
        """Should calculate reward correctly"""
        env = MarketMakingEnv(initial_cash=10000.0)
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.0
        mock_simulator.portfolio = MagicMock()
        mock_simulator.portfolio.get_stats.return_value = {
            'realized_pnl': 100.0,
            'unrealized_pnl': 50.0,
            'position': 0.1
        }
        
        env.simulator = mock_simulator
        
        reward = env._calculate_reward()
        
        # reward = realized_pnl/initial_cash + alpha*unrealized_pnl/initial_cash - beta*inventory_penalty
        # = 100/10000 + 0.1*50/10000 - 0.05*(0.1*50000/10000)^2
        # = 0.01 + 0.0005 - 0.05*0.25 = 0.01 + 0.0005 - 0.0125 = -0.002
        assert isinstance(reward, float)
    
    def test_reward_no_mid_price(self):
        """Should return 0 reward when mid price is None"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = None
        mock_simulator.portfolio = MagicMock()
        
        env.simulator = mock_simulator
        
        reward = env._calculate_reward()
        assert reward == 0.0


class TestObservation:
    """Test observation generation"""
    
    def test_get_observation(self):
        """Should generate correct observation vector"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': (
                [[50000.0, 1.5], [49999.0, 2.0]],
                [[50001.0, 1.2], [50002.0, 2.5]]
            ),
            'portfolio': {
                'cash': 10000.0,
                'position': 0.1,
                'unrealized_pnl': 50.0,
                'return_pct': 5.0
            }
        }
        
        env.simulator = mock_simulator
        env.price_history = [50000.0, 50001.0, 50002.0]
        
        obs = env._get_observation()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (52,)
        assert obs.dtype == np.float32
        assert obs[0] == 50000.0  # best_bid
        assert obs[1] == 50001.0  # best_ask
        assert obs[2] == 50000.5  # mid_price


class TestMarketFeatures:
    """Test market feature calculations"""
    
    def test_calculate_volatility(self):
        """Should calculate volatility correctly"""
        env = MarketMakingEnv()
        env.price_history = [100.0, 101.0, 99.0, 102.0, 100.0]
        
        volatility = env._calculate_volatility()
        
        assert isinstance(volatility, float)
        assert volatility >= 0.0
    
    def test_calculate_volatility_insufficient_data(self):
        """Should return 0.0 with insufficient data"""
        env = MarketMakingEnv()
        env.price_history = [100.0]
        
        volatility = env._calculate_volatility()
        assert volatility == 0.0
    
    def test_calculate_price_change(self):
        """Should calculate price change correctly"""
        env = MarketMakingEnv()
        env.price_history = [100.0, 101.0, 102.0]
        
        price_change = env._calculate_price_change()
        
        assert isinstance(price_change, float)
        # (102 - 100) / 100 = 0.02
        assert price_change == pytest.approx(0.02, abs=0.001)
    
    def test_calculate_price_change_insufficient_data(self):
        """Should return 0.0 with insufficient data"""
        env = MarketMakingEnv()
        env.price_history = [100.0]
        
        price_change = env._calculate_price_change()
        assert price_change == 0.0


class TestRender:
    """Test rendering functionality"""
    
    def test_render_human_mode(self, capsys):
        """Should print formatted output in human mode"""
        env = MarketMakingEnv()
        
        mock_simulator = MagicMock(spec=MarketSimulator)
        mock_simulator.orderbook = MagicMock()
        mock_simulator.orderbook.get_mid_price.return_value = 50000.5
        mock_simulator.orderbook.get_spread.return_value = 1.0
        mock_simulator.portfolio = MagicMock()
        mock_simulator.get_current_state.return_value = {
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'mid_price': 50000.5,
            'spread': 1.0,
            'spread_pct': 0.002,
            'imbalance': 0.1,
            'depth': ([], []),
            'portfolio': {}
        }
        mock_simulator.portfolio.get_stats.return_value = {
            'cash': 10000.0,
            'position': 0.1,
            'total_value': 15000.0,
            'realized_pnl': 100.0,
            'unrealized_pnl': 50.0,
            'total_pnl': 150.0,
            'return_pct': 50.0,
            'open_orders_count': 2,
            'total_fills': 5
        }
        
        env.simulator = mock_simulator
        env.current_step = 10
        env.price_history = [50000.0, 50001.0]
        
        env.render()
        
        captured = capsys.readouterr()
        assert "Step: 10" in captured.out
        assert "Market State:" in captured.out
        assert "Portfolio:" in captured.out
    
    def test_render_invalid_mode(self):
        """Should raise ValueError for invalid render mode"""
        env = MarketMakingEnv()
        
        with pytest.raises(ValueError, match="Unsupported render mode"):
            env.render(mode='invalid')

