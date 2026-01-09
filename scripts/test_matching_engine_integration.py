#!/usr/bin/env python3
"""
Integration test script for MatchingEngine
Tests realistic trading scenarios and end-to-end order matching flows
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sim.matching_engine import MatchingEngine
from app.sim.orderbook import OrderBook
from app.sim.orders import LimitOrder, MarketOrder, OrderSide, OrderStatus
from app.sim.exceptions import InsufficientLiquidityError


def test_basic_limit_order_matching():
    """Test basic limit order matching scenarios"""
    print("\n=== Test 1: Basic Limit Order Matching ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    # Set up orderbook with realistic levels
    snapshot = {
        'bids': [
            [49900.0, 2.0],
            [49890.0, 3.0],
            [49880.0, 1.5]
        ],
        'asks': [
            [50000.0, 1.5],
            [50010.0, 2.0],
            [50020.0, 1.0]
        ],
        'datetime': '2024-01-01T12:00:00+00:00'
    }
    order_book.update_from_snapshot(snapshot)
    
    print(f"✓ Orderbook initialized: best bid={order_book.get_best_bid()}, best ask={order_book.get_best_ask()}")
    
    # Test limit buy that matches
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0
    )
    
    fills = engine.match_limit_order(buy_order, order_book)
    print(f"✓ Limit buy order: {len(fills)} fills, status={buy_order.status}, filled={buy_order.filled_quantity}")
    assert len(fills) == 1
    assert buy_order.status == OrderStatus.FILLED
    assert fills[0].price == 50000.0
    assert fills[0].quantity == 1.0
    
    # Test limit sell that matches
    sell_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=1.0,
        price=49900.0
    )
    
    fills = engine.match_limit_order(sell_order, order_book)
    print(f"✓ Limit sell order: {len(fills)} fills, status={sell_order.status}, filled={sell_order.filled_quantity}")
    assert len(fills) == 1
    assert sell_order.status == OrderStatus.FILLED
    assert fills[0].price == 49900.0
    
    print("✓ Basic limit order matching test passed\n")


def test_market_order_execution():
    """Test market order execution scenarios"""
    print("\n=== Test 2: Market Order Execution ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [
            [49900.0, 1.0],
            [49890.0, 2.0]
        ],
        'asks': [
            [50000.0, 0.5],
            [50010.0, 0.8],
            [50020.0, 1.2]
        ],
    }
    order_book.update_from_snapshot(snapshot)
    
    # Test market buy that fills through multiple levels
    market_buy = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0
    )
    
    fills = engine.match_market_order(market_buy, order_book)
    print(f"✓ Market buy: {len(fills)} fills across price levels")
    assert len(fills) == 2  # 0.5 at 50000.0, 0.5 at 50010.0
    assert market_buy.status == OrderStatus.FILLED
    assert sum(f.quantity for f in fills) == 1.0
    assert fills[0].price == 50000.0
    assert fills[1].price == 50010.0
    
    # Rebuild orderbook for sell test
    order_book.update_from_snapshot(snapshot)
    
    # Test market sell
    market_sell = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=1.0
    )
    
    fills = engine.match_market_order(market_sell, order_book)
    print(f"✓ Market sell: {len(fills)} fills, status={market_sell.status}")
    assert len(fills) == 1
    assert market_sell.status == OrderStatus.FILLED
    assert fills[0].price == 49900.0
    
    print("✓ Market order execution test passed\n")


def test_fee_calculations():
    """Test fee calculations for maker and taker orders"""
    print("\n=== Test 3: Fee Calculations ===")
    
    engine = MatchingEngine(maker_fee_rate=-0.0001, taker_fee_rate=0.0001)
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [[49900.0, 1.0]],
        'asks': [[50000.0, 1.0]],
    }
    order_book.update_from_snapshot(snapshot)
    
    # Test maker fee (limit order - should be negative rebate)
    limit_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0
    )
    
    limit_fills = engine.match_limit_order(limit_order, order_book)
    maker_fee = limit_fills[0].fee
    print(f"✓ Maker fee (limit order): {maker_fee:.4f} (should be negative)")
    assert maker_fee < 0
    assert abs(abs(maker_fee) - 5.0) < 0.01  # 1.0 * 0.0001 * 50000.0
    
    # Rebuild for taker test
    order_book.update_from_snapshot(snapshot)
    
    # Test taker fee (market order - should be positive cost)
    market_order = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0
    )
    
    market_fills = engine.match_market_order(market_order, order_book)
    taker_fee = market_fills[0].fee
    print(f"✓ Taker fee (market order): {taker_fee:.4f} (should be positive)")
    assert taker_fee > 0
    assert abs(taker_fee - 5.0) < 0.01  # 1.0 * 0.0001 * 50000.0
    
    print("✓ Fee calculations test passed\n")


def test_slippage_calculations():
    """Test slippage calculations for different scenarios"""
    print("\n=== Test 4: Slippage Calculations ===")
    
    engine = MatchingEngine()
    from app.sim.orders import Fill
    
    # Test buy order with positive slippage
    fills = [
        Fill(
            order_id="test1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0,
            fee=0.0
        ),
        Fill(
            order_id="test1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50050.0,
            fee=0.0
        )
    ]
    
    expected_price = 50000.0
    slippage = engine.calculate_slippage(fills, expected_price, OrderSide.BUY)
    avg_price = (0.5 * 50000.0 + 0.5 * 50050.0) / 1.0
    print(f"✓ Buy slippage: {slippage:.6f} (avg price={avg_price:.2f}, expected={expected_price})")
    assert slippage > 0  # Positive slippage means worse execution
    assert abs(slippage - 0.0005) < 0.0001  # (50025 - 50000) / 50000
    
    # Test sell order with slippage
    fills = [
        Fill(
            order_id="test2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=49950.0,  # Lower than expected (worse for sell)
            fee=0.0
        )
    ]
    
    expected_price = 50000.0
    slippage = engine.calculate_slippage(fills, expected_price, OrderSide.SELL)
    print(f"✓ Sell slippage: {slippage:.6f} (fill price={fills[0].price}, expected={expected_price})")
    assert slippage > 0
    assert abs(slippage - 0.001) < 0.0001  # (50000 - 49950) / 50000
    
    print("✓ Slippage calculations test passed\n")


def test_partial_fills():
    """Test partial fill scenarios"""
    print("\n=== Test 5: Partial Fills ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [[49900.0, 0.3]],
        'asks': [[50000.0, 0.5]],
    }
    order_book.update_from_snapshot(snapshot)
    
    # Limit buy that partially fills
    order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0
    )
    
    fills = engine.match_limit_order(order, order_book)
    print(f"✓ Partial fill: {len(fills)} fills, filled={order.filled_quantity}/{order.quantity}")
    assert len(fills) == 1
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == 0.5
    assert order.remaining_quantity == 0.5
    
    print("✓ Partial fills test passed\n")


def test_insufficient_liquidity():
    """Test insufficient liquidity scenarios"""
    print("\n=== Test 6: Insufficient Liquidity ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [[49900.0, 0.5]],
        'asks': [[50000.0, 0.3]],
    }
    order_book.update_from_snapshot(snapshot)
    
    # Market order that can't be fully filled
    market_order = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0
    )
    
    try:
        engine.match_market_order(market_order, order_book)
        assert False, "Should have raised InsufficientLiquidityError"
    except InsufficientLiquidityError as e:
        print(f"✓ Caught InsufficientLiquidityError: {e}")
        assert "partially filled" in str(e).lower() or "insufficient" in str(e).lower()
    
    # Empty orderbook
    empty_orderbook = OrderBook("BTC/USDT")
    market_order2 = MarketOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0
    )
    
    try:
        engine.match_market_order(market_order2, empty_orderbook)
        assert False, "Should have raised InsufficientLiquidityError"
    except InsufficientLiquidityError as e:
        print(f"✓ Caught InsufficientLiquidityError for empty orderbook: {e}")
    
    print("✓ Insufficient liquidity test passed\n")


def test_multiple_orders_sequence():
    """Test sequence of multiple orders"""
    print("\n=== Test 7: Multiple Orders Sequence ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [
            [49900.0, 2.0],
            [49890.0, 3.0]
        ],
        'asks': [
            [50000.0, 2.0],
            [50010.0, 3.0]
        ],
    }
    order_book.update_from_snapshot(snapshot)
    
    # Execute multiple orders
    orders = [
        LimitOrder(symbol="BTC/USDT", side=OrderSide.BUY, quantity=1.0, price=50000.0),
        LimitOrder(symbol="BTC/USDT", side=OrderSide.SELL, quantity=0.5, price=49900.0),
        MarketOrder(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.5),
    ]
    
    total_fills = 0
    for i, order in enumerate(orders):
        if isinstance(order, LimitOrder):
            fills = engine.match_limit_order(order, order_book)
        else:
            fills = engine.match_market_order(order, order_book)
        total_fills += len(fills)
        print(f"✓ Order {i+1}: {len(fills)} fills, status={order.status}")
    
    print(f"✓ Total fills across all orders: {total_fills}")
    assert total_fills >= 3
    
    print("✓ Multiple orders sequence test passed\n")


def test_orderbook_state_consistency():
    """Test that orderbook state remains consistent after matching"""
    print("\n=== Test 8: Orderbook State Consistency ===")
    
    engine = MatchingEngine()
    order_book = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [
            [49900.0, 2.0],
            [49890.0, 3.0]
        ],
        'asks': [
            [50000.0, 1.5],
            [50010.0, 2.0]
        ],
    }
    order_book.update_from_snapshot(snapshot)
    
    initial_bid_count = len(order_book.bids)
    initial_ask_count = len(order_book.asks)
    
    # Execute order that consumes a level
    order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.5,
        price=50000.0
    )
    
    fills = engine.match_limit_order(order, order_book)
    
    # Check that consumed level is removed
    assert 50000.0 not in order_book.asks
    assert len(order_book.asks) == initial_ask_count - 1
    
    # Execute order that partially consumes a level
    order2 = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=1.0,
        price=49900.0
    )
    
    fills2 = engine.match_limit_order(order2, order_book)
    
    # Check that level quantity is reduced but not removed
    assert 49900.0 in order_book.bids
    assert order_book.bids[49900.0] == 1.0  # 2.0 - 1.0
    
    print(f"✓ Orderbook state consistent: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
    print("✓ Orderbook state consistency test passed\n")


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("MatchingEngine Integration Tests")
    print("=" * 60)
    
    try:
        test_basic_limit_order_matching()
        test_market_order_execution()
        test_fee_calculations()
        test_slippage_calculations()
        test_partial_fills()
        test_insufficient_liquidity()
        test_multiple_orders_sequence()
        test_orderbook_state_consistency()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

