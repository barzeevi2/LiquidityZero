"""
Unit tests for MarketSimulator
Tests historical data loading, simulation stepping, order placement, and state management
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from app.sim.market_simulator import MarketSimulator
from app.sim.orders import LimitOrder, OrderSide, OrderStatus
from app.sim.exceptions import SimulationError


class TestMarketSimulatorInitialization:
    """Test MarketSimulator initialization"""
    
    def test_initialization_default_values(self):
        """Should initialize with default values"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        assert sim.symbol == "BTC/USDT"
        assert sim.orderbook.symbol == "BTC/USDT"
        assert sim.match_engine is not None
        assert sim.portfolio.initial_cash == 10000.0
        assert sim.portfolio.symbol == "BTC/USDT"
        assert len(sim.snapshots) == 0
        assert sim.current_index == 0
        assert sim.current_time is None
        assert sim.is_running is False
    
    def test_initialization_custom_values(self):
        """Should initialize with custom cash and connection string"""
        sim = MarketSimulator(
            symbol="ETH/USDT",
            initial_cash=50000.0,
            connection_string="postgresql://test"
        )
        
        assert sim.symbol == "ETH/USDT"
        assert sim.portfolio.initial_cash == 50000.0
        assert sim.connection_string == "postgresql://test"


class TestLoadHistoricalData:
    """Test loading historical data from database"""
    
    @pytest.mark.asyncio
    async def test_load_historical_data_success(self):
        """Should successfully load historical data"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        mock_rows = [
            {
                'time': datetime(2024, 1, 1, 0, 0, 0),
                'symbol': 'BTC/USDT',
                'bids': [[50000.0, 1.5], [49999.0, 2.0]],
                'asks': [[50001.0, 1.2], [50002.0, 2.5]],
                'best_bid': 50000.0,
                'best_ask': 50001.0,
                'spread': 1.0,
                'spread_pct': 0.002
            },
            {
                'time': datetime(2024, 1, 1, 0, 0, 1),
                'symbol': 'BTC/USDT',
                'bids': [[50010.0, 1.6], [50009.0, 2.1]],
                'asks': [[50011.0, 1.3], [50012.0, 2.6]],
                'best_bid': 50010.0,
                'best_ask': 50011.0,
                'spread': 1.0,
                'spread_pct': 0.002
            }
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        mock_conn.close = AsyncMock()
        
        with patch('app.sim.market_simulator.asyncpg.connect', return_value=mock_conn):
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 1, 0, 0, 2)
            
            await sim.load_historical_data(start_time, end_time)
            
            assert len(sim.snapshots) == 2
            assert sim.snapshots[0]['symbol'] == 'BTC/USDT'
            assert sim.snapshots[0]['best_bid'] == 50000.0
            assert sim.snapshots[1]['best_bid'] == 50010.0
            mock_conn.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_historical_data_empty_results(self):
        """Should raise SimulationError when no data found"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()
        
        with patch('app.sim.market_simulator.asyncpg.connect', return_value=mock_conn):
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 1, 0, 0, 2)
            
            with pytest.raises(SimulationError, match="No orderbook snapshots found"):
                await sim.load_historical_data(start_time, end_time)
    
    @pytest.mark.asyncio
    async def test_load_historical_data_database_error(self):
        """Should raise SimulationError on database error"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        with patch('app.sim.market_simulator.asyncpg.connect', side_effect=Exception("Connection failed")):
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            end_time = datetime(2024, 1, 1, 0, 0, 2)
            
            with pytest.raises(SimulationError, match="Error loading historical data"):
                await sim.load_historical_data(start_time, end_time)


class TestReset:
    """Test resetting simulator"""
    
    def test_reset_to_start(self):
        """Should reset to start index 0"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        # Create mock snapshots
        sim.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]]
            },
            {
                'datetime': '2024-01-01T00:00:01+00:00',
                'bids': [[50010.0, 1.6]],
                'asks': [[50011.0, 1.3]]
            }
        ]
        
        sim.reset(0)
        
        assert sim.current_index == 0
        assert sim.current_time == datetime.fromisoformat('2024-01-01T00:00:00+00:00')
        assert sim.orderbook.get_best_bid() == 50000.0
        assert sim.portfolio.initial_cash == 10000.0
        assert len(sim.portfolio.open_orders) == 0
    
    def test_reset_to_specific_index(self):
        """Should reset to specific index"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]},
            {'datetime': '2024-01-01T00:00:01+00:00', 'bids': [[50010.0, 1.6]], 'asks': [[50011.0, 1.3]]},
            {'datetime': '2024-01-01T00:00:02+00:00', 'bids': [[50020.0, 1.7]], 'asks': [[50021.0, 1.4]]}
        ]
        
        sim.reset(1)
        
        assert sim.current_index == 1
        assert sim.current_time == datetime.fromisoformat('2024-01-01T00:00:01+00:00')
        assert sim.orderbook.get_best_bid() == 50010.0
    
    def test_reset_no_snapshots(self):
        """Should raise SimulationError when no snapshots loaded"""
        sim = MarketSimulator(symbol="BTC/USDT")
        sim.snapshots = []
        
        with pytest.raises(SimulationError, match="No snapshots loaded"):
            sim.reset(0)
    
    def test_reset_index_out_of_range(self):
        """Should raise SimulationError when index is out of range"""
        sim = MarketSimulator(symbol="BTC/USDT")
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ]
        
        with pytest.raises(SimulationError, match="out of range"):
            sim.reset(5)
        
        with pytest.raises(SimulationError, match="out of range"):
            sim.reset(-1)
    
    def test_reset_clears_portfolio_state(self):
        """Should reset portfolio to initial state"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=20000.0)
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ]
        
        # Modify portfolio state
        sim.portfolio.cash = 15000.0
        sim.portfolio.position = 0.1
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        sim.portfolio.place_order(order)
        
        # Reset should restore initial state
        sim.reset(0)
        
        assert sim.portfolio.cash == 20000.0
        assert sim.portfolio.position == 0.0
        assert len(sim.portfolio.open_orders) == 0


class TestStep:
    """Test simulation stepping"""
    
    @pytest.mark.asyncio
    async def test_step_success(self):
        """Should successfully step to next snapshot"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        sim.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]]
            },
            {
                'datetime': '2024-01-01T00:00:01+00:00',
                'bids': [[50010.0, 1.6]],
                'asks': [[50011.0, 1.3]]
            }
        ]
        
        sim.reset(0)
        assert sim.current_index == 0
        
        result = await sim.step()
        
        assert result is True
        assert sim.current_index == 1
        assert sim.current_time == datetime.fromisoformat('2024-01-01T00:00:01+00:00')
        assert sim.orderbook.get_best_bid() == 50010.0
    
    @pytest.mark.asyncio
    async def test_step_end_of_data(self):
        """Should return False at end of data"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]},
            {'datetime': '2024-01-01T00:00:01+00:00', 'bids': [[50010.0, 1.6]], 'asks': [[50011.0, 1.3]]}
        ]
        
        sim.reset(0)
        await sim.step()  # Move to index 1
        
        result = await sim.step()  # Try to step past end
        
        assert result is False
        assert sim.current_index == 1  # Should not have incremented
    
    @pytest.mark.asyncio
    async def test_step_matches_agent_orders(self):
        """Should match agent orders against updated orderbook"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
        
        sim.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]]
            },
            {
                'datetime': '2024-01-01T00:00:01+00:00',
                'bids': [[50010.0, 1.6]],
                'asks': [[50000.0, 0.5]]  # Price dropped, our buy order should match
            }
        ]
        
        sim.reset(0)
        
        # Place a buy limit order at 50000
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0
        )
        sim.place_agent_order(order)
        
        assert len(sim.portfolio.open_orders) == 1
        
        # Step should match the order
        await sim.step()
        
        # Order should be filled or partially filled
        assert order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert order.filled_quantity > 0


class TestPlaceAgentOrder:
    """Test placing agent orders"""
    
    def test_place_agent_order_buy(self):
        """Should place a buy limit order"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ]
        sim.reset(0)
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0
        )
        
        sim.place_agent_order(order)
        
        assert order.order_id in sim.portfolio.open_orders
        assert len(sim.portfolio.open_orders) == 1
        # Check that order is in orderbook agent_orders
        assert order.order_id in sim.orderbook.agent_orders
    
    def test_place_agent_order_sell(self):
        """Should place a sell limit order"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ]
        sim.reset(0)
        
        # First add position
        sim.portfolio.position = 0.5
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.3,
            price=51000.0
        )
        
        sim.place_agent_order(order)
        
        assert order.order_id in sim.portfolio.open_orders
        assert order.order_id in sim.orderbook.agent_orders


class TestCancelAgentOrder:
    """Test canceling agent orders"""
    
    def test_cancel_agent_order_success(self):
        """Should successfully cancel an agent order"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
        
        sim.snapshots = [
            {'datetime': '2024-01-01T00:00:00+00:00', 'bids': [[50000.0, 1.5]], 'asks': [[50001.0, 1.2]]}
        ]
        sim.reset(0)
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0
        )
        sim.place_agent_order(order)
        
        assert order.order_id in sim.portfolio.open_orders
        assert order.order_id in sim.orderbook.agent_orders
        
        result = sim.cancel_agent_order(order.order_id)
        
        assert result is True
        assert order.order_id not in sim.portfolio.open_orders
        assert order.order_id not in sim.orderbook.agent_orders
        assert order.status == OrderStatus.CANCELLED
    
    def test_cancel_agent_order_not_found(self):
        """Should return False when order not found"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        result = sim.cancel_agent_order("nonexistent_id")
        assert result is False


class TestGetCurrentState:
    """Test getting current market state"""
    
    def test_get_current_state(self):
        """Should return complete market state"""
        sim = MarketSimulator(symbol="BTC/USDT")
        
        sim.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5], [49999.0, 2.0]],
                'asks': [[50001.0, 1.2], [50002.0, 2.5]]
            }
        ]
        sim.reset(0)
        
        state = sim.get_current_state()
        
        assert 'timestamp' in state
        assert 'best_bid' in state
        assert 'best_ask' in state
        assert 'mid_price' in state
        assert 'spread' in state
        assert 'spread_pct' in state
        assert 'imbalance' in state
        assert 'depth' in state
        assert 'portfolio' in state
        
        assert state['best_bid'] == 50000.0
        assert state['best_ask'] == 50001.0
        assert state['mid_price'] == 50000.5
        assert state['timestamp'] == datetime.fromisoformat('2024-01-01T00:00:00+00:00')
        assert isinstance(state['portfolio'], dict)
    
    def test_get_current_state_with_position(self):
        """Should include portfolio statistics in state"""
        sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
        
        sim.snapshots = [
            {
                'datetime': '2024-01-01T00:00:00+00:00',
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]]
            }
        ]
        sim.reset(0)
        
        # Add position
        sim.portfolio.position = 0.1
        sim.portfolio.cash = 45000.0
        
        state = sim.get_current_state()
        
        assert 'portfolio' in state
        portfolio_stats = state['portfolio']
        assert 'cash' in portfolio_stats
        assert 'position' in portfolio_stats
        assert 'total_value' in portfolio_stats
        assert portfolio_stats['position'] == 0.1
        assert portfolio_stats['cash'] == 45000.0

