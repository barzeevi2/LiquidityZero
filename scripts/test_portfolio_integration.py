#!/usr/bin/env python3
"""
Integration test script for Portfolio
Tests realistic trading scenarios and end-to-end portfolio management flows
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sim.portfolio import Portfolio
from app.sim.orders import LimitOrder, MarketOrder, OrderSide, OrderStatus, Fill


def test_basic_portfolio_operations():
    """Test basic portfolio initialization and operations"""
    print("\n=== Test 1: Basic Portfolio Operations ===")
    
    portfolio = Portfolio(initial_cash=10000.0, symbol="BTC/USDT")
    
    assert portfolio.initial_cash == 10000.0
    assert portfolio.cash == 10000.0
    assert portfolio.position == 0.0
    assert portfolio.symbol == "BTC/USDT"
    print(f"✓ Portfolio initialized: cash={portfolio.cash}, position={portfolio.position}")
    
    # Test total value calculation
    value = portfolio.get_total_value(50000.0)
    assert value == 10000.0
    print(f"✓ Total value (no position): {value}")
    
    print("✓ Basic portfolio operations test passed\n")


def test_limit_order_placement():
    """Test placing and managing limit orders"""
    print("\n=== Test 2: Limit Order Placement ===")
    
    portfolio = Portfolio(initial_cash=50000.0)
    
    # Place a buy order
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=50000.0
    )
    portfolio.place_order(buy_order)
    print(f"✓ Placed buy order: {buy_order.quantity} @ {buy_order.price}")
    
    assert buy_order.order_id in portfolio.open_orders
    assert len(portfolio.open_orders) == 1
    
    # Try to place another order with same ID (should fail)
    try:
        portfolio.place_order(buy_order)
        assert False, "Should have raised ValueError for duplicate order ID"
    except ValueError as e:
        print(f"✓ Duplicate order correctly rejected: {e}")
    
    # Cancel the order
    result = portfolio.cancel_order(buy_order.order_id)
    assert result is True
    assert buy_order.order_id not in portfolio.open_orders
    assert buy_order.status == OrderStatus.CANCELLED
    print(f"✓ Order cancelled successfully")
    
    print("✓ Limit order placement test passed\n")


def test_buy_and_sell_scenario():
    """Test a complete buy-sell trading scenario"""
    print("\n=== Test 3: Buy and Sell Scenario ===")
    
    portfolio = Portfolio(initial_cash=20000.0)
    
    # Place buy order
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.2,
        price=50000.0
    )
    portfolio.place_order(buy_order)
    
    # Simulate fill
    buy_fill = Fill(
        order_id=buy_order.order_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.2,
        price=50000.0,
        fee=-10.0  # Maker rebate
    )
    portfolio.record_fill(buy_fill)
    
    print(f"✓ Bought 0.2 BTC @ 50000: cash={portfolio.cash:.2f}, position={portfolio.position}")
    assert portfolio.position == 0.2
    # For maker rebate (negative fee): cost = 0.2 * 50000 + (-10.0) = 9990.0
    assert portfolio.cash == 20000.0 - (0.2 * 50000.0) - (-10.0)  # 10010.0
    assert buy_order.status == OrderStatus.FILLED
    assert buy_order.order_id not in portfolio.open_orders
    
    # Place sell order
    sell_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.2,
        price=51000.0
    )
    portfolio.place_order(sell_order)
    
    # Simulate fill
    sell_fill = Fill(
        order_id=sell_order.order_id,
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.2,
        price=51000.0,
        fee=10.2  # Taker fee
    )
    portfolio.record_fill(sell_fill)
    
    print(f"✓ Sold 0.2 BTC @ 51000: cash={portfolio.cash:.2f}, position={portfolio.position}")
    assert portfolio.position == 0.0
    # Sell revenue: 0.2 * 51000 - 10.2 = 10189.8
    # New cash: 10010.0 + 10189.8 = 20199.8
    assert abs(portfolio.cash - (20000.0 - (0.2 * 50000.0) - (-10.0) + (0.2 * 51000.0) - 10.2)) < 0.01
    assert sell_order.status == OrderStatus.FILLED
    
    # Check PnL
    realized_pnl = portfolio.get_realized_pnl()
    print(f"✓ Realized PnL: {realized_pnl:.2f}")
    assert realized_pnl > 0  # Should have made profit
    
    print("✓ Buy and sell scenario test passed\n")


def test_partial_fills():
    """Test partial fill scenarios"""
    print("\n=== Test 4: Partial Fills ===")
    
    portfolio = Portfolio(initial_cash=50000.0)
    
    # Place large buy order
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=50000.0
    )
    portfolio.place_order(buy_order)
    
    # First partial fill
    fill1 = Fill(
        order_id=buy_order.order_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.2,
        price=50000.0,
        fee=-10.0
    )
    portfolio.record_fill(fill1)
    
    print(f"✓ First partial fill: 0.2/0.5, status={buy_order.status}")
    assert buy_order.status == OrderStatus.PARTIALLY_FILLED
    assert buy_order.filled_quantity == 0.2
    assert buy_order.order_id in portfolio.open_orders
    assert portfolio.position == 0.2
    
    # Second partial fill
    fill2 = Fill(
        order_id=buy_order.order_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.2,
        price=50000.0,
        fee=-10.0
    )
    portfolio.record_fill(fill2)
    
    print(f"✓ Second partial fill: 0.4/0.5, status={buy_order.status}")
    assert buy_order.status == OrderStatus.PARTIALLY_FILLED
    assert buy_order.filled_quantity == 0.4
    assert portfolio.position == 0.4
    
    # Final fill to complete order
    fill3 = Fill(
        order_id=buy_order.order_id,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        fee=-5.0
    )
    portfolio.record_fill(fill3)
    
    print(f"✓ Order completed: {buy_order.filled_quantity}/0.5, status={buy_order.status}")
    assert buy_order.status == OrderStatus.FILLED
    assert buy_order.filled_quantity == 0.5
    assert buy_order.order_id not in portfolio.open_orders
    assert portfolio.position == 0.5
    
    print("✓ Partial fills test passed\n")


def test_pnl_calculations():
    """Test PnL calculation accuracy"""
    print("\n=== Test 5: PnL Calculations ===")
    
    portfolio = Portfolio(initial_cash=30000.0)
    current_price = 52000.0
    
    # Buy at 50000
    buy_fill1 = Fill(
        order_id="buy1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        fee=-5.0
    )
    portfolio.record_fill(buy_fill1)
    
    # Buy more at 51000
    buy_fill2 = Fill(
        order_id="buy2",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=51000.0,
        fee=-5.1
    )
    portfolio.record_fill(buy_fill2)
    
    # Should have 0.2 position
    assert portfolio.position == 0.2
    
    # Check unrealized PnL
    unrealized_pnl = portfolio.get_unrealized_pnl(current_price)
    print(f"✓ Unrealized PnL @ {current_price}: {unrealized_pnl:.2f}")
    assert unrealized_pnl > 0  # Price is above average cost
    
    # Sell half
    sell_fill = Fill(
        order_id="sell1",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        price=51500.0,
        fee=5.15
    )
    portfolio.record_fill(sell_fill)
    
    # Should have 0.1 position left
    assert portfolio.position == 0.1
    
    # Check realized PnL
    realized_pnl = portfolio.get_realized_pnl()
    print(f"✓ Realized PnL: {realized_pnl:.2f}")
    
    # Check total value and PnL
    total_value = portfolio.get_total_value(current_price)
    unrealized_pnl_after = portfolio.get_unrealized_pnl(current_price)
    total_pnl = realized_pnl + unrealized_pnl_after
    
    print(f"✓ Total value @ {current_price}: {total_value:.2f}")
    print(f"✓ Total PnL (realized + unrealized): {total_pnl:.2f}")
    
    # Verify total value matches initial cash + total PnL
    expected_value = portfolio.initial_cash + total_pnl
    assert abs(total_value - expected_value) < 1.0  # Allow small rounding differences
    
    print("✓ PnL calculations test passed\n")


def test_portfolio_stats():
    """Test portfolio statistics generation"""
    print("\n=== Test 6: Portfolio Statistics ===")
    
    portfolio = Portfolio(initial_cash=10000.0)
    
    # Make some trades
    buy_fill = Fill(
        order_id="buy1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        fee=-5.0
    )
    portfolio.record_fill(buy_fill)
    
    # Place an order (not filled)
    order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.05,
        price=51000.0
    )
    portfolio.place_order(order)
    
    # Get stats without current price
    stats = portfolio.get_stats()
    print(f"✓ Stats without price: {stats}")
    assert 'cash' in stats
    assert 'position' in stats
    assert 'open_orders_count' in stats
    assert stats['open_orders_count'] == 1
    assert 'realized_pnl' in stats
    assert 'total_value' not in stats  # Not included without current_price
    
    # Get stats with current price
    stats_with_price = portfolio.get_stats(current_price=52000.0)
    print(f"✓ Stats with price @ 52000: {stats_with_price}")
    assert 'total_value' in stats_with_price
    assert 'unrealized_pnl' in stats_with_price
    assert 'total_pnl' in stats_with_price
    assert 'return_pct' in stats_with_price
    
    print(f"  - Cash: {stats_with_price['cash']:.2f}")
    print(f"  - Position: {stats_with_price['position']}")
    print(f"  - Total Value: {stats_with_price['total_value']:.2f}")
    print(f"  - Realized PnL: {stats_with_price['realized_pnl']:.2f}")
    print(f"  - Unrealized PnL: {stats_with_price['unrealized_pnl']:.2f}")
    print(f"  - Total PnL: {stats_with_price['total_pnl']:.2f}")
    print(f"  - Return %: {stats_with_price['return_pct']:.2f}%")
    
    print("✓ Portfolio statistics test passed\n")


def test_insufficient_funds_scenarios():
    """Test scenarios with insufficient cash or position"""
    print("\n=== Test 7: Insufficient Funds Scenarios ===")
    
    portfolio = Portfolio(initial_cash=1000.0)  # Low cash
    
    # Try to place order requiring more cash
    try:
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0  # Requires 5000.0
        )
        portfolio.place_order(order)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected order with insufficient cash: {e}")
    
    # Try to place sell order without position
    try:
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            price=51000.0
        )
        portfolio.place_order(order)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected sell order without position: {e}")
    
    # Add some position
    fill = Fill(
        order_id="buy1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.05,  # Buy with available cash
        price=20000.0,
        fee=-1.0
    )
    portfolio.record_fill(fill)
    
    # Try to sell more than we have
    try:
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=0.1,  # Want to sell 0.1, but only have 0.05
            price=21000.0
        )
        portfolio.place_order(order)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected sell order exceeding position: {e}")
    
    print("✓ Insufficient funds scenarios test passed\n")


def test_market_making_scenario():
    """Test a realistic market-making scenario"""
    print("\n=== Test 8: Market-Making Scenario ===")
    
    portfolio = Portfolio(initial_cash=50000.0)
    print(f"Initial portfolio: cash={portfolio.cash:.2f}, position={portfolio.position}")
    
    # Market maker buys at bid
    buy_fill1 = Fill(
        order_id="mm_buy1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=49900.0,
        fee=-4.99  # Maker rebate
    )
    portfolio.record_fill(buy_fill1)
    print(f"✓ MM bought 0.1 @ 49900: cash={portfolio.cash:.2f}, position={portfolio.position}")
    
    # Market maker sells at ask
    sell_fill1 = Fill(
        order_id="mm_sell1",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.1,
        price=50100.0,
        fee=5.01  # Taker fee (if crossed spread)
    )
    portfolio.record_fill(sell_fill1)
    print(f"✓ MM sold 0.1 @ 50100: cash={portfolio.cash:.2f}, position={portfolio.position}")
    assert portfolio.position == 0.0
    
    # Made profit from spread + fees
    realized_pnl = portfolio.get_realized_pnl()
    print(f"✓ Realized PnL from round trip: {realized_pnl:.2f}")
    assert realized_pnl > 0
    
    # Another round trip
    buy_fill2 = Fill(
        order_id="mm_buy2",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.15,
        price=49950.0,
        fee=-7.49
    )
    portfolio.record_fill(buy_fill2)
    
    sell_fill2 = Fill(
        order_id="mm_sell2",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=0.15,
        price=50050.0,
        fee=7.51
    )
    portfolio.record_fill(sell_fill2)
    
    assert portfolio.position == 0.0
    
    # Final stats
    stats = portfolio.get_stats(current_price=50000.0)
    print(f"\n✓ Final Portfolio Stats:")
    print(f"  - Cash: {stats['cash']:.2f}")
    print(f"  - Realized PnL: {stats['realized_pnl']:.2f}")
    print(f"  - Total Fees Paid: {stats['total_fees_paid']:.2f}")
    print(f"  - Total Fills: {stats['total_fills']}")
    print(f"  - Return %: {stats['return_pct']:.4f}%")
    
    # Portfolio should be profitable from market making
    assert stats['realized_pnl'] > 0
    assert stats['return_pct'] > 0
    
    print("✓ Market-making scenario test passed\n")


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Portfolio Integration Tests")
    print("=" * 60)
    
    try:
        test_basic_portfolio_operations()
        test_limit_order_placement()
        test_buy_and_sell_scenario()
        test_partial_fills()
        test_pnl_calculations()
        test_portfolio_stats()
        test_insufficient_funds_scenarios()
        test_market_making_scenario()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ Test failed with assertion error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

