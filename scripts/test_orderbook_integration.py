#!/usr/bin/env python3
"""
Integration test script for OrderBook
Tests realistic scenarios and edge cases in a single execution flow
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sim.orderbook import OrderBook


def test_basic_functionality():
    """Test basic orderbook operations"""
    print("\n=== Test 1: Basic Functionality ===")
    ob = OrderBook("BTC/USDT")
    
    # Update from snapshot
    snapshot = {
        'bids': [
            [50000.0, 1.5],
            [49999.0, 2.0],
            [49998.0, 3.0],
            [49997.0, 1.0]
        ],
        'asks': [
            [50001.0, 1.2],
            [50002.0, 2.5],
            [50003.0, 1.8],
            [50004.0, 0.9]
        ],
        'datetime': '2024-01-01T12:00:00+00:00'
    }
    
    ob.update_from_snapshot(snapshot)
    print(f"✓ Updated from snapshot: {len(ob.bids)} bids, {len(ob.asks)} asks")
    
    # Check best prices
    best_bid = ob.get_best_bid()
    best_ask = ob.get_best_ask()
    print(f"✓ Best bid: {best_bid}, Best ask: {best_ask}")
    assert best_bid == 50000.0, f"Expected best bid 50000.0, got {best_bid}"
    assert best_ask == 50001.0, f"Expected best ask 50001.0, got {best_ask}"
    
    # Check mid price and spread
    mid_price = ob.get_mid_price()
    spread = ob.get_spread()
    spread_pct = ob.get_spread_percentage()
    print(f"✓ Mid price: {mid_price}, Spread: {spread}, Spread %: {spread_pct:.4f}")
    assert mid_price == 50000.5
    assert spread == 1.0
    
    print("✓ Basic functionality test passed\n")


def test_agent_orders():
    """Test agent order management"""
    print("\n=== Test 2: Agent Orders ===")
    ob = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [[50000.0, 1.5]],
        'asks': [[50001.0, 1.2]],
    }
    ob.update_from_snapshot(snapshot)
    
    # Add buy orders
    ob.add_agent_order("buy_order_1", 50000.5, 2.0, "BUY")
    ob.add_agent_order("buy_order_2", 49999.5, 1.5, "BUY")
    print(f"✓ Added 2 buy orders: {len(ob.agent_orders)} total agent orders")
    
    # Add sell orders
    ob.add_agent_order("sell_order_1", 50001.5, 1.8, "SELL")
    ob.add_agent_order("sell_order_2", 50002.0, 0.9, "SELL")
    print(f"✓ Added 2 sell orders: {len(ob.agent_orders)} total agent orders")
    
    # Check that orders are tracked
    assert len(ob.agent_orders) == 4
    assert ob.bids[50000.5] == 2.0
    assert ob.bids[49999.5] == 1.5
    assert ob.asks[50001.5] == 1.8
    assert ob.asks[50002.0] == 0.9
    
    # Remove orders
    result1 = ob.remove_agent_order("buy_order_1")
    result2 = ob.remove_agent_order("sell_order_1")
    print(f"✓ Removed 2 orders: {len(ob.agent_orders)} remaining")
    assert result1 is True
    assert result2 is True
    assert len(ob.agent_orders) == 2
    assert 50000.5 not in ob.bids
    assert 50001.5 not in ob.asks
    
    # Try removing non-existent order
    result3 = ob.remove_agent_order("nonexistent")
    assert result3 is False
    print("✓ Non-existent order removal handled correctly")
    
    print("✓ Agent orders test passed\n")


def test_depth_calculations():
    """Test market depth calculations"""
    print("\n=== Test 3: Market Depth ===")
    ob = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [
            [50000.0, 1.5],
            [49999.0, 2.0],
            [49998.0, 3.0],
            [49997.0, 1.0]
        ],
        'asks': [
            [50001.0, 1.2],
            [50002.0, 2.5],
            [50003.0, 1.8],
            [50004.0, 0.9]
        ],
    }
    ob.update_from_snapshot(snapshot)
    
    # Test buy depth (asks at or below price)
    depth_buy = ob.get_depth(50002.0, "BUY")
    print(f"✓ Buy depth at 50002.0: {depth_buy}")
    assert depth_buy == 3.7  # 1.2 + 2.5
    
    depth_buy_high = ob.get_depth(50005.0, "BUY")
    print(f"✓ Buy depth at 50005.0 (above all asks): {depth_buy_high}")
    assert depth_buy_high == 6.4  # All asks: 1.2 + 2.5 + 1.8 + 0.9
    
    # Test sell depth (bids at or above price)
    depth_sell = ob.get_depth(49999.0, "SELL")
    print(f"✓ Sell depth at 49999.0: {depth_sell}")
    assert depth_sell == 3.5  # 1.5 + 2.0
    
    depth_sell_low = ob.get_depth(49996.0, "SELL")
    print(f"✓ Sell depth at 49996.0 (below all bids): {depth_sell_low}")
    assert depth_sell_low == 7.5  # All bids: 1.5 + 2.0 + 3.0 + 1.0
    
    print("✓ Market depth test passed\n")


def test_imbalance_calculation():
    """Test orderbook imbalance"""
    print("\n=== Test 4: Orderbook Imbalance ===")
    
    # Balanced orderbook
    ob1 = OrderBook("BTC/USDT")
    snapshot1 = {
        'bids': [[50000.0, 5.0], [49999.0, 3.0], [49998.0, 2.0]],
        'asks': [[50001.0, 5.0], [50002.0, 3.0], [50003.0, 2.0]],
    }
    ob1.update_from_snapshot(snapshot1)
    imbalance1 = ob1.get_imbalance()
    print(f"✓ Balanced orderbook imbalance: {imbalance1:.4f}")
    assert abs(imbalance1) < 0.01  # Should be close to 0
    
    # Bullish orderbook (more buy volume)
    ob2 = OrderBook("BTC/USDT")
    snapshot2 = {
        'bids': [[50000.0, 10.0], [49999.0, 5.0]],
        'asks': [[50001.0, 2.0], [50002.0, 1.0]],
    }
    ob2.update_from_snapshot(snapshot2)
    imbalance2 = ob2.get_imbalance()
    print(f"✓ Bullish orderbook imbalance: {imbalance2:.4f}")
    assert imbalance2 > 0
    
    # Bearish orderbook (more sell volume)
    ob3 = OrderBook("BTC/USDT")
    snapshot3 = {
        'bids': [[50000.0, 2.0], [49999.0, 1.0]],
        'asks': [[50001.0, 10.0], [50002.0, 5.0]],
    }
    ob3.update_from_snapshot(snapshot3)
    imbalance3 = ob3.get_imbalance()
    print(f"✓ Bearish orderbook imbalance: {imbalance3:.4f}")
    assert imbalance3 < 0
    
    print("✓ Imbalance calculation test passed\n")


def test_snapshot_updates():
    """Test multiple snapshot updates"""
    print("\n=== Test 5: Snapshot Updates ===")
    ob = OrderBook("BTC/USDT")
    
    # First snapshot
    snapshot1 = {
        'bids': [[50000.0, 1.5], [49999.0, 2.0]],
        'asks': [[50001.0, 1.2], [50002.0, 2.5]],
        'datetime': '2024-01-01T10:00:00+00:00'
    }
    ob.update_from_snapshot(snapshot1)
    print(f"✓ Snapshot 1: {len(ob.bids)} bids, {len(ob.asks)} asks")
    
    # Add agent orders
    ob.add_agent_order("agent1", 50000.5, 1.0, "BUY")
    ob.add_agent_order("agent2", 50001.5, 0.8, "SELL")
    assert len(ob.agent_orders) == 2
    print(f"✓ Added agent orders: {len(ob.agent_orders)}")
    
    # Second snapshot (should clear snapshot data, keep agent orders)
    snapshot2 = {
        'bids': [[51000.0, 3.0], [50999.0, 2.5]],
        'asks': [[51001.0, 2.0], [51002.0, 1.5]],
        'datetime': '2024-01-01T11:00:00+00:00'
    }
    ob.update_from_snapshot(snapshot2)
    print(f"✓ Snapshot 2: {len(ob.bids)} bids, {len(ob.asks)} asks")
    
    # Agent orders should still exist
    assert len(ob.agent_orders) == 2
    # But old snapshot prices should be gone
    assert 50000.0 not in ob.bids
    assert 50001.0 not in ob.asks
    # New snapshot prices should exist
    assert 51000.0 in ob.bids
    assert 51001.0 in ob.asks
    print("✓ Agent orders preserved after snapshot update")
    
    print("✓ Snapshot updates test passed\n")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 6: Edge Cases ===")
    
    # Empty orderbook
    ob1 = OrderBook("BTC/USDT")
    snapshot_empty = {
        'bids': [],
        'asks': [],
    }
    ob1.update_from_snapshot(snapshot_empty)
    assert ob1.get_best_bid() is None
    assert ob1.get_best_ask() is None
    assert ob1.get_mid_price() is None
    assert ob1.get_spread() is None
    assert ob1.get_imbalance() == 0.0
    print("✓ Empty orderbook handled correctly")
    
    # Only bids
    ob2 = OrderBook("BTC/USDT")
    snapshot_bids_only = {
        'bids': [[50000.0, 1.5]],
        'asks': [],
    }
    ob2.update_from_snapshot(snapshot_bids_only)
    assert ob2.get_best_bid() == 50000.0
    assert ob2.get_best_ask() is None
    assert ob2.get_mid_price() is None
    print("✓ Bids-only orderbook handled correctly")
    
    # Only asks
    ob3 = OrderBook("BTC/USDT")
    snapshot_asks_only = {
        'bids': [],
        'asks': [[50001.0, 1.2]],
    }
    ob3.update_from_snapshot(snapshot_asks_only)
    assert ob3.get_best_bid() is None
    assert ob3.get_best_ask() == 50001.0
    assert ob3.get_mid_price() is None
    print("✓ Asks-only orderbook handled correctly")
    
    # String prices and quantities
    ob4 = OrderBook("BTC/USDT")
    snapshot_strings = {
        'bids': [['50000.0', '1.5'], ['49999.0', '2.0']],
        'asks': [['50001.0', '1.2'], ['50002.0', '2.5']],
    }
    ob4.update_from_snapshot(snapshot_strings)
    assert ob4.bids[50000.0] == 1.5
    assert ob4.asks[50001.0] == 1.2
    print("✓ String prices/quantities handled correctly")
    
    # Duplicate prices in snapshot
    ob5 = OrderBook("BTC/USDT")
    snapshot_duplicates = {
        'bids': [[50000.0, 1.5], [50000.0, 2.0], [50000.0, 0.5]],
        'asks': [[50001.0, 1.2], [50001.0, 0.8]],
    }
    ob5.update_from_snapshot(snapshot_duplicates)
    assert ob5.bids[50000.0] == 4.0  # 1.5 + 2.0 + 0.5
    assert ob5.asks[50001.0] == 2.0  # 1.2 + 0.8
    print("✓ Duplicate prices aggregated correctly")
    
    print("✓ Edge cases test passed\n")


def test_depth_levels():
    """Test depth levels retrieval"""
    print("\n=== Test 7: Depth Levels ===")
    ob = OrderBook("BTC/USDT")
    
    snapshot = {
        'bids': [
            [49997.0, 1.0],
            [49998.0, 3.0],
            [49999.0, 2.0],
            [50000.0, 1.5]
        ],
        'asks': [
            [50001.0, 1.2],
            [50002.0, 2.5],
            [50003.0, 1.8],
            [50004.0, 0.9]
        ],
    }
    ob.update_from_snapshot(snapshot)
    
    bid_levels, ask_levels = ob.get_depth_levels(2)
    print(f"✓ Top 2 bid levels: {bid_levels}")
    print(f"✓ Top 2 ask levels: {ask_levels}")
    
    assert len(bid_levels) == 2
    assert len(ask_levels) == 2
    # Bids should be highest first
    assert bid_levels[0] == [50000.0, 1.5]
    assert bid_levels[1] == [49999.0, 2.0]
    # Asks should be lowest first
    assert ask_levels[0] == [50001.0, 1.2]
    assert ask_levels[1] == [50002.0, 2.5]
    
    print("✓ Depth levels test passed\n")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("OrderBook Integration Test Suite")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_agent_orders()
        test_depth_calculations()
        test_imbalance_calculation()
        test_snapshot_updates()
        test_edge_cases()
        test_depth_levels()
        
        print("=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

