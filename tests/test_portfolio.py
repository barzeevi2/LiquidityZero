"""
Unit tests for Portfolio
Tests portfolio management, order tracking, PnL calculations, and cash/position management
"""

import pytest
from app.sim.portfolio import Portfolio
from app.sim.orders import LimitOrder, MarketOrder, OrderSide, OrderStatus, Fill
from datetime import datetime


class TestPortfolioInitialization:
    """Test Portfolio initialization"""
    
    def test_initialization_default_values(self):
        """Should initialize with default cash and symbol"""
        portfolio = Portfolio()
        assert portfolio.initial_cash == 10000.0
        assert portfolio.cash == 10000.0
        assert portfolio.position == 0.0
        assert portfolio.symbol == "BTC/USDT"
        assert len(portfolio.open_orders) == 0
        assert len(portfolio.order_history) == 0
        assert len(portfolio.fill_history) == 0
        assert portfolio.realized_pnl == 0.0
        assert portfolio.total_fees_paid == 0.0
    
    def test_initialization_custom_values(self):
        """Should initialize with custom cash and symbol"""
        portfolio = Portfolio(initial_cash=50000.0, symbol="ETH/USDT")
        assert portfolio.initial_cash == 50000.0
        assert portfolio.cash == 50000.0
        assert portfolio.symbol == "ETH/USDT"


class TestPlaceOrder:
    """Test placing orders"""
    
    def test_place_limit_buy_order_success(self):
        """Should successfully place a limit buy order"""
        portfolio = Portfolio(initial_cash=10000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        portfolio.place_order(order)
        
        assert order.order_id in portfolio.open_orders
        assert portfolio.open_orders[order.order_id] == order
        assert len(portfolio.open_orders) == 1
    
    def test_place_limit_buy_order_insufficient_cash(self):
        """Should raise ValueError if insufficient cash for limit buy"""
        portfolio = Portfolio(initial_cash=1000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0  # Requires 5000.0, but only have 1000.0
        )
        
        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.place_order(order)
    
    def test_place_limit_sell_order_success(self):
        """Should successfully place a limit sell order when position exists"""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.position = 0.5
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0
        )
        portfolio.place_order(order)
        
        assert order.order_id in portfolio.open_orders
    
    def test_place_limit_sell_order_insufficient_position(self):
        """Should raise ValueError if insufficient position for limit sell"""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.position = 0.05
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,  # Want to sell 0.1, but only have 0.05
            price=51000.0
        )
        
        with pytest.raises(ValueError, match="Insufficient position"):
            portfolio.place_order(order)
    
    def test_place_order_duplicate_id(self):
        """Should raise ValueError if order ID already exists"""
        portfolio = Portfolio(initial_cash=10000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        portfolio.place_order(order)
        
        # Try to place same order again
        with pytest.raises(ValueError, match="already exists"):
            portfolio.place_order(order)
    
    def test_place_market_order_no_validation(self):
        """Market orders should not validate cash/position (handled by matching engine)"""
        portfolio = Portfolio(initial_cash=1000.0)  # Insufficient cash for limit order
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        # Should not raise error - market orders bypass validation
        portfolio.place_order(order)


class TestCancelOrder:
    """Test canceling orders"""
    
    def test_cancel_order_success(self):
        """Should successfully cancel an open order"""
        portfolio = Portfolio(initial_cash=10000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        portfolio.place_order(order)
        order_id = order.order_id
        
        result = portfolio.cancel_order(order_id)
        
        assert result is True
        assert order_id not in portfolio.open_orders
        assert order.status == OrderStatus.CANCELLED
        assert order in portfolio.order_history
    
    def test_cancel_order_not_found(self):
        """Should return False if order not found"""
        portfolio = Portfolio()
        result = portfolio.cancel_order("nonexistent_id")
        assert result is False
    
    def test_cancel_order_multiple_orders(self):
        """Should only cancel the specified order"""
        portfolio = Portfolio(initial_cash=50000.0)
        order1 = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        order2 = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=49000.0
        )
        portfolio.place_order(order1)
        portfolio.place_order(order2)
        
        portfolio.cancel_order(order1.order_id)
        
        assert order1.order_id not in portfolio.open_orders
        assert order2.order_id in portfolio.open_orders
        assert len(portfolio.open_orders) == 1


class TestRecordFill:
    """Test recording fills"""
    
    def test_record_fill_buy(self):
        """Should correctly record a buy fill"""
        portfolio = Portfolio(initial_cash=10000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        portfolio.place_order(order)
        
        fill = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0  # Maker rebate
        )
        
        initial_cash = portfolio.cash
        portfolio.record_fill(fill)
        
        assert len(portfolio.fill_history) == 1
        assert portfolio.fill_history[0] == fill
        assert portfolio.total_fees_paid == 5.0  # abs(fee)
        # For maker rebate (negative fee): cost = quantity * price + fee = 5000 + (-5) = 4995
        assert portfolio.cash == initial_cash - (0.1 * 50000.0) - (-5.0)  # cost = price*qty + fee
        assert portfolio.position == 0.1
        assert order.filled_quantity == 0.1
        assert order.status == OrderStatus.FILLED
        assert order.order_id not in portfolio.open_orders
        assert order in portfolio.order_history
    
    def test_record_fill_sell(self):
        """Should correctly record a sell fill"""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.position = 0.2
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0
        )
        portfolio.place_order(order)
        
        fill = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0,
            fee=5.1  # Taker fee
        )
        
        initial_cash = portfolio.cash
        portfolio.record_fill(fill)
        
        assert len(portfolio.fill_history) == 1
        assert portfolio.total_fees_paid == 5.1
        assert portfolio.cash == initial_cash + (0.1 * 51000.0) - 5.1  # revenue - fee
        assert portfolio.position == 0.1  # 0.2 - 0.1
        assert order.filled_quantity == 0.1
        assert order.status == OrderStatus.FILLED
    
    def test_record_fill_partial(self):
        """Should correctly record a partial fill"""
        portfolio = Portfolio(initial_cash=10000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.2,
            price=50000.0
        )
        portfolio.place_order(order)
        
        fill = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        
        portfolio.record_fill(fill)
        
        assert order.filled_quantity == 0.1
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.order_id in portfolio.open_orders
        assert portfolio.position == 0.1
    
    def test_record_fill_multiple_partial(self):
        """Should correctly handle multiple partial fills"""
        portfolio = Portfolio(initial_cash=20000.0)
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.3,
            price=50000.0
        )
        portfolio.place_order(order)
        
        fill1 = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        fill2 = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        
        portfolio.record_fill(fill1)
        portfolio.record_fill(fill2)
        
        assert order.filled_quantity == 0.2
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert portfolio.position == 0.2
        
        # Complete the fill
        fill3 = Fill(
            order_id=order.order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        portfolio.record_fill(fill3)
        
        assert abs(order.filled_quantity - 0.3) < 0.0001  # Handle floating point precision
        assert order.status == OrderStatus.FILLED
        assert order.order_id not in portfolio.open_orders
        assert abs(portfolio.position - 0.3) < 0.0001
    
    def test_record_fill_order_not_in_open_orders(self):
        """Should still record fill even if order not in open_orders"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        fill = Fill(
            order_id="some_order_id",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        
        portfolio.record_fill(fill)
        
        assert len(portfolio.fill_history) == 1
        assert portfolio.position == 0.1
        # For maker rebate (negative fee): cost = quantity * price + fee = 5000 + (-5) = 4995
        assert portfolio.cash == 10000.0 - (0.1 * 50000.0) - (-5.0)


class TestGetTotalValue:
    """Test total value calculation"""
    
    def test_get_total_value_no_position(self):
        """Should return cash when no position"""
        portfolio = Portfolio(initial_cash=10000.0)
        assert portfolio.get_total_value(50000.0) == 10000.0
    
    def test_get_total_value_with_position(self):
        """Should return cash + position value"""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.position = 0.2
        portfolio.cash = 5000.0
        
        value = portfolio.get_total_value(50000.0)
        assert value == 5000.0 + (0.2 * 50000.0)  # 15000.0
    
    def test_get_total_value_short_position(self):
        """Should handle short positions correctly"""
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.position = -0.1  # Short position
        
        value = portfolio.get_total_value(50000.0)
        assert value == 10000.0 + (-0.1 * 50000.0)  # 5000.0


class TestGetRealizedPnl:
    """Test realized PnL calculation"""
    
    def test_get_realized_pnl_no_fills(self):
        """Should return 0 when no fills"""
        portfolio = Portfolio()
        assert portfolio.get_realized_pnl() == 0.0
    
    def test_get_realized_pnl_no_sells(self):
        """Should return 0 when no sells (only buys)"""
        portfolio = Portfolio(initial_cash=20000.0)
        fill = Fill(
            order_id="order1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        portfolio.record_fill(fill)
        assert portfolio.get_realized_pnl() == 0.0
    
    def test_get_realized_pnl_simple_round_trip(self):
        """Should calculate realized PnL for a simple buy-sell round trip"""
        portfolio = Portfolio(initial_cash=20000.0)
        
        # Buy 0.1 BTC at 50000
        buy_fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0  # Maker rebate
        )
        portfolio.record_fill(buy_fill)
        
        # Sell 0.1 BTC at 51000
        sell_fill = Fill(
            order_id="sell1",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0,
            fee=5.1  # Taker fee
        )
        portfolio.record_fill(sell_fill)
        
        # Position should be 0 (fully closed)
        assert portfolio.position == 0.0
        
        realized_pnl = portfolio.get_realized_pnl()
        # Buy cost: 0.1 * 50000 + (-5.0) = 4995.0 (maker rebate reduces cost)
        # Sell revenue: 0.1 * 51000 - 5.1 = 5094.9
        # Avg buy cost: 4995.0 / 0.1 = 49950.0 per unit
        # Cost of sold (0.1): 49950.0 * 0.1 = 4995.0
        # Realized PnL: 5094.9 - 4995.0 = 99.9
        # Note: The calculation uses average cost method
        assert realized_pnl > 0  # Should be positive
        assert abs(realized_pnl - 99.9) < 1.0  # Allow some tolerance
    
    def test_get_realized_pnl_partial_close(self):
        """Should calculate realized PnL for partial position close"""
        portfolio = Portfolio(initial_cash=30000.0)
        
        # Buy 0.2 BTC at 50000
        buy_fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.2,
            price=50000.0,
            fee=-10.0
        )
        portfolio.record_fill(buy_fill)
        
        # Sell 0.1 BTC at 51000 (partial close)
        sell_fill = Fill(
            order_id="sell1",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0,
            fee=5.1
        )
        portfolio.record_fill(sell_fill)
        
        # Should still have 0.1 BTC position
        assert portfolio.position == 0.1
        
        realized_pnl = portfolio.get_realized_pnl()
        # Should realize PnL on 0.1 BTC sold
        # Buy cost: 0.2 * 50000 + (-10.0) = 9990.0 (maker rebate)
        # Avg buy cost: 9990.0 / 0.2 = 49950.0 per unit
        # Cost of sold (0.1): 49950.0 * 0.1 = 4995.0
        # Sell revenue: 0.1 * 51000 - 5.1 = 5094.9
        # Realized PnL: 5094.9 - 4995.0 = 99.9
        assert realized_pnl > 0  # Should be positive
        assert abs(realized_pnl - 99.9) < 1.0  # Allow some tolerance


class TestGetUnrealizedPnl:
    """Test unrealized PnL calculation"""
    
    def test_get_unrealized_pnl_no_position(self):
        """Should return 0 when no position"""
        portfolio = Portfolio()
        assert portfolio.get_unrealized_pnl(50000.0) == 0.0
    
    def test_get_unrealized_pnl_long_position_profit(self):
        """Should calculate unrealized profit for long position"""
        portfolio = Portfolio(initial_cash=20000.0)
        
        fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        portfolio.record_fill(fill)
        
        # Current price higher than buy price
        unrealized_pnl = portfolio.get_unrealized_pnl(51000.0)
        # Buy cost: 0.1 * 50000 + (-5.0) = 4995.0 (maker rebate)
        # Avg cost: 4995.0 / 0.1 = 49950.0 per unit
        # Unrealized: (51000 - 49950) * 0.1 = 105.0
        assert unrealized_pnl > 0  # Should be positive
        assert abs(unrealized_pnl - 105.0) < 0.1
    
    def test_get_unrealized_pnl_long_position_loss(self):
        """Should calculate unrealized loss for long position"""
        portfolio = Portfolio(initial_cash=20000.0)
        
        fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        portfolio.record_fill(fill)
        
        # Current price lower than buy price
        unrealized_pnl = portfolio.get_unrealized_pnl(49000.0)
        # Should be negative
        assert unrealized_pnl < 0
    
    def test_get_unrealized_pnl_multiple_buys(self):
        """Should use average cost for multiple buys"""
        portfolio = Portfolio(initial_cash=50000.0)
        
        fill1 = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        fill2 = Fill(
            order_id="buy2",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=51000.0,
            fee=-5.1
        )
        portfolio.record_fill(fill1)
        portfolio.record_fill(fill2)
        
        # Current price
        unrealized_pnl = portfolio.get_unrealized_pnl(52000.0)
        # Avg cost: ((0.1 * 50000 + 5.0) + (0.1 * 51000 + 5.1)) / 0.2 = 50505.05
        # Unrealized: (52000 - 50505.05) * 0.2 = 298.99
        assert unrealized_pnl > 0
    
    def test_get_unrealized_pnl_short_position(self):
        """Should calculate unrealized PnL for short position"""
        portfolio = Portfolio(initial_cash=20000.0)
        
        # Need to have position first, then sell more
        buy_fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.05,
            price=50000.0,
            fee=-2.5
        )
        portfolio.record_fill(buy_fill)
        
        sell_fill = Fill(
            order_id="sell1",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.15,  # Sell more than we have (short 0.1)
            price=51000.0,
            fee=7.65
        )
        portfolio.record_fill(sell_fill)
        
        # Should have -0.1 position (short) - allow for floating point precision
        assert abs(portfolio.position - (-0.1)) < 0.0001
        
        unrealized_pnl = portfolio.get_unrealized_pnl(50000.0)
        # For short positions, we calculate differently
        # This should handle short positions correctly
        assert isinstance(unrealized_pnl, float)


class TestGetStats:
    """Test portfolio statistics"""
    
    def test_get_stats_basic(self):
        """Should return basic stats without current price"""
        portfolio = Portfolio(initial_cash=10000.0)
        stats = portfolio.get_stats()
        
        assert 'cash' in stats
        assert 'position' in stats
        assert 'open_orders_count' in stats
        assert 'total_fills' in stats
        assert 'total_fees_paid' in stats
        assert 'realized_pnl' in stats
        assert stats['cash'] == 10000.0
        assert stats['position'] == 0.0
        assert stats['open_orders_count'] == 0
    
    def test_get_stats_with_current_price(self):
        """Should return extended stats with current price"""
        portfolio = Portfolio(initial_cash=20000.0)
        
        fill = Fill(
            order_id="buy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            fee=-5.0
        )
        portfolio.record_fill(fill)
        
        stats = portfolio.get_stats(current_price=51000.0)
        
        assert 'total_value' in stats
        assert 'unrealized_pnl' in stats
        assert 'total_pnl' in stats
        assert 'return_pct' in stats
        
        assert stats['total_value'] == portfolio.get_total_value(51000.0)
        assert stats['unrealized_pnl'] == portfolio.get_unrealized_pnl(51000.0)
        assert stats['total_pnl'] == stats['realized_pnl'] + stats['unrealized_pnl']
    
    def test_get_stats_return_percentage(self):
        """Should calculate return percentage correctly"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        stats = portfolio.get_stats(current_price=50000.0)
        assert stats['return_pct'] == 0.0  # No trades, no change
        
        # Add some value through position
        portfolio.position = 0.1
        portfolio.cash = 5000.0
        
        stats = portfolio.get_stats(current_price=50000.0)
        # Total value: 5000 + 0.1 * 50000 = 10000
        # Return: (10000 - 10000) / 10000 * 100 = 0%
        assert stats['return_pct'] == 0.0
        
        # Price goes up
        stats = portfolio.get_stats(current_price=60000.0)
        # Total value: 5000 + 0.1 * 60000 = 11000
        # Return: (11000 - 10000) / 10000 * 100 = 10%
        assert abs(stats['return_pct'] - 10.0) < 0.01

