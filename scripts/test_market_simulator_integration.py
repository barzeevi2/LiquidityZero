#!/usr/bin/env python3
"""
Integration test script for MarketSimulator
Tests realistic simulation scenarios, historical data replay, and end-to-end trading flows
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sim.market_simulator import MarketSimulator
from app.sim.orders import LimitOrder, OrderSide, OrderStatus
from app.sim.exceptions import SimulationError
import asyncio


def create_mock_snapshots():
    """Create mock orderbook snapshots for testing"""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    snapshots = []
    
    # Create 10 snapshots with increasing prices (bullish trend)
    for i in range(10):
        snapshot = {
            'datetime': (base_time + timedelta(seconds=i)).isoformat(),
            'timestamp': (base_time + timedelta(seconds=i)).timestamp(),
            'symbol': 'BTC/USDT',
            'bids': [
                [50000.0 + i * 10, 1.5 + i * 0.1],
                [49999.0 + i * 10, 2.0 + i * 0.1],
                [49998.0 + i * 10, 3.0 + i * 0.1]
            ],
            'asks': [
                [50001.0 + i * 10, 1.2 + i * 0.1],
                [50002.0 + i * 10, 2.5 + i * 0.1],
                [50003.0 + i * 10, 1.8 + i * 0.1]
            ],
            'best_bid': 50000.0 + i * 10,
            'best_ask': 50001.0 + i * 10,
            'spread': 1.0,
            'spread_pct': 0.002
        }
        snapshots.append(snapshot)
    
    return snapshots


async def test_basic_simulation_flow():
    """Test basic simulation flow"""
    print("\n=== Test 1: Basic Simulation Flow ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
    
    # Create mock snapshots
    sim.snapshots = create_mock_snapshots()
    
    # Reset to start
    sim.reset(0)
    print(f"✓ Simulator reset: current_index={sim.current_index}, time={sim.current_time}")
    assert sim.current_index == 0
    assert sim.current_time == datetime.fromisoformat(sim.snapshots[0]['datetime'])
    
    # Get initial state
    initial_state = sim.get_current_state()
    print(f"✓ Initial state: best_bid={initial_state['best_bid']}, best_ask={initial_state['best_ask']}")
    assert initial_state['best_bid'] == 50000.0
    assert initial_state['best_ask'] == 50001.0
    
    # Step through a few snapshots
    for i in range(3):
        has_more = await sim.step()
        state = sim.get_current_state()
        print(f"✓ Step {i+1}: index={sim.current_index}, bid={state['best_bid']}, ask={state['best_ask']}")
        assert has_more is True
        assert state['best_bid'] == 50000.0 + (i + 1) * 10
    
    print("✓ Basic simulation flow test passed\n")


async def test_order_placement_and_matching():
    """Test placing orders and matching against market"""
    print("\n=== Test 2: Order Placement and Matching ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=100000.0)
    
    # Create snapshots with price that will match our order
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
        },
        {
            'datetime': '2024-01-01T00:00:02+00:00',
            'bids': [[50010.0, 1.6]],
            'asks': [[50020.0, 1.7]]  # Price went up, our sell order should match
        }
    ]
    
    sim.reset(0)
    
    # Place a buy limit order at 50000
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.3,
        price=50000.0
    )
    sim.place_agent_order(buy_order)
    print(f"✓ Placed buy order: {buy_order.quantity} @ {buy_order.price}")
    assert buy_order.order_id in sim.portfolio.open_orders
    assert buy_order.order_id in sim.orderbook.agent_orders
    
    # Step - should match the order
    has_more = await sim.step()
    state = sim.get_current_state()
    print(f"✓ After step 1: order status={buy_order.status}, filled={buy_order.filled_quantity}")
    
    # Order should be filled or partially filled
    assert buy_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
    assert buy_order.filled_quantity > 0
    assert sim.portfolio.position > 0
    
    # Place a sell order if we have position
    if sim.portfolio.position > 0:
        sell_order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=min(sim.portfolio.position, 0.2),
            price=50010.0
        )
        sim.place_agent_order(sell_order)
        print(f"✓ Placed sell order: {sell_order.quantity} @ {sell_order.price}")
        
        # Step again - should match the sell order
        await sim.step()
        print(f"✓ After step 2: sell order status={sell_order.status}, filled={sell_order.filled_quantity}")
        assert sell_order.filled_quantity > 0
    
    print("✓ Order placement and matching test passed\n")


async def test_order_cancellation():
    """Test canceling orders"""
    print("\n=== Test 3: Order Cancellation ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
    
    sim.snapshots = create_mock_snapshots()[:5]  # Use 5 snapshots
    sim.reset(0)
    
    # Place an order
    order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=49900.0  # Price that won't match
    )
    sim.place_agent_order(order)
    
    print(f"✓ Placed order: {order.order_id}")
    assert order.order_id in sim.portfolio.open_orders
    assert order.order_id in sim.orderbook.agent_orders
    
    # Cancel the order
    result = sim.cancel_agent_order(order.order_id)
    print(f"✓ Cancelled order: result={result}")
    
    assert result is True
    assert order.order_id not in sim.portfolio.open_orders
    assert order.order_id not in sim.orderbook.agent_orders
    assert order.status == OrderStatus.CANCELLED
    
    # Try to cancel non-existent order
    result = sim.cancel_agent_order("nonexistent")
    assert result is False
    
    print("✓ Order cancellation test passed\n")


async def test_reset_functionality():
    """Test reset functionality"""
    print("\n=== Test 4: Reset Functionality ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=30000.0)
    
    sim.snapshots = create_mock_snapshots()[:5]
    sim.reset(0)
    
    # Step through a few snapshots
    initial_cash = sim.portfolio.cash
    for i in range(3):
        await sim.step()
    
    assert sim.current_index == 3
    print(f"✓ Stepped to index {sim.current_index}")
    
    # Modify portfolio
    order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0
    )
    sim.place_agent_order(order)
    
    # Reset to start
    sim.reset(0)
    print(f"✓ Reset to index 0")
    
    assert sim.current_index == 0
    assert sim.portfolio.cash == 30000.0
    assert sim.portfolio.position == 0.0
    assert len(sim.portfolio.open_orders) == 0
    assert sim.current_time == datetime.fromisoformat(sim.snapshots[0]['datetime'])
    
    # Reset to specific index
    sim.reset(2)
    print(f"✓ Reset to index 2")
    
    assert sim.current_index == 2
    assert sim.current_time == datetime.fromisoformat(sim.snapshots[2]['datetime'])
    
    print("✓ Reset functionality test passed\n")


async def test_full_simulation_cycle():
    """Test a complete simulation cycle"""
    print("\n=== Test 5: Full Simulation Cycle ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=100000.0)
    
    sim.snapshots = create_mock_snapshots()
    sim.reset(0)
    
    initial_cash = sim.portfolio.initial_cash
    initial_state = sim.get_current_state()
    print(f"✓ Initial: cash={initial_cash}, bid={initial_state['best_bid']}, ask={initial_state['best_ask']}")
    
    # Place a buy order early
    buy_order = LimitOrder(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=50050.0  # Will likely match as price goes up
    )
    sim.place_agent_order(buy_order)
    
    step_count = 0
    while True:
        has_more = await sim.step()
        
        if not has_more:
            break
        
        step_count += 1
        state = sim.get_current_state()
        
        if step_count % 3 == 0:
            print(f"✓ Step {step_count}: index={sim.current_index}, bid={state['best_bid']:.1f}, "
                  f"position={sim.portfolio.position:.3f}, cash={sim.portfolio.cash:.2f}")
    
    print(f"✓ Completed {step_count} steps")
    
    final_state = sim.get_current_state()
    final_stats = final_state['portfolio']
    
    print(f"✓ Final stats:")
    print(f"  - Cash: {final_stats['cash']:.2f}")
    print(f"  - Position: {final_stats['position']:.3f}")
    print(f"  - Total Value: {final_stats['total_value']:.2f}")
    print(f"  - Realized PnL: {final_stats['realized_pnl']:.2f}")
    print(f"  - Unrealized PnL: {final_stats.get('unrealized_pnl', 0):.2f}")
    
    assert step_count == len(sim.snapshots) - 1  # Should process all but initial snapshot
    assert sim.current_index == len(sim.snapshots) - 1
    
    print("✓ Full simulation cycle test passed\n")


async def test_market_making_strategy():
    """Test a simple market-making strategy"""
    print("\n=== Test 6: Market-Making Strategy ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=200000.0)
    
    # Create volatile snapshots
    sim.snapshots = []
    base_price = 50000.0
    for i in range(20):
        price_offset = (i % 4 - 1.5) * 10  # Oscillating price
        snapshot = {
            'datetime': (datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=i)).isoformat(),
            'timestamp': (datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=i)).timestamp(),
            'symbol': 'BTC/USDT',
            'bids': [[base_price + price_offset - 1, 2.0], [base_price + price_offset - 2, 3.0]],
            'asks': [[base_price + price_offset + 1, 2.0], [base_price + price_offset + 2, 3.0]],
            'best_bid': base_price + price_offset - 1,
            'best_ask': base_price + price_offset + 1,
            'spread': 2.0,
            'spread_pct': 0.004
        }
        sim.snapshots.append(snapshot)
    
    sim.reset(0)
    
    print(f"✓ Starting market-making simulation")
    print(f"  Initial cash: {sim.portfolio.initial_cash:.2f}")
    
    # Simple market-making: place buy at bid-1, sell at ask+1
    for step_num in range(min(15, len(sim.snapshots) - 1)):
        state = sim.get_current_state()
        mid_price = state['mid_price']
        
        if mid_price:
            # Place buy order below mid
            if len(sim.portfolio.open_orders) < 2:  # Max 2 orders
                buy_order = LimitOrder(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    quantity=0.1,
                    price=mid_price - 50
                )
                try:
                    sim.place_agent_order(buy_order)
                except ValueError:
                    pass  # Insufficient funds or position
        
        # Step
        has_more = await sim.step()
        
        if step_num % 5 == 0:
            stats = state['portfolio']
            print(f"  Step {step_num}: cash={stats['cash']:.2f}, position={stats['position']:.3f}, "
                  f"open_orders={stats['open_orders_count']}")
        
        if not has_more:
            break
    
    final_state = sim.get_current_state()
    final_stats = final_state['portfolio']
    
    print(f"✓ Market-making simulation complete:")
    print(f"  - Cash: {final_stats['cash']:.2f}")
    print(f"  - Position: {final_stats['position']:.3f}")
    print(f"  - Total Value: {final_stats['total_value']:.2f}")
    print(f"  - Total Fills: {final_stats['total_fills']}")
    print(f"  - Open Orders: {final_stats['open_orders_count']}")
    
    print("✓ Market-making strategy test passed\n")


async def test_error_handling():
    """Test error handling"""
    print("\n=== Test 7: Error Handling ===")
    
    sim = MarketSimulator(symbol="BTC/USDT")
    
    # Test reset without snapshots
    try:
        sim.reset(0)
        assert False, "Should have raised SimulationError"
    except SimulationError as e:
        print(f"✓ Correctly raised error for reset without snapshots: {e}")
    
    # Test reset with invalid index
    sim.snapshots = create_mock_snapshots()[:3]
    try:
        sim.reset(10)
        assert False, "Should have raised SimulationError"
    except SimulationError as e:
        print(f"✓ Correctly raised error for invalid index: {e}")
    
    try:
        sim.reset(-1)
        assert False, "Should have raised SimulationError"
    except SimulationError as e:
        print(f"✓ Correctly raised error for negative index: {e}")
    
    print("✓ Error handling test passed\n")


async def test_state_observations():
    """Test state observation functionality"""
    print("\n=== Test 8: State Observations ===")
    
    sim = MarketSimulator(symbol="BTC/USDT", initial_cash=50000.0)
    
    sim.snapshots = create_mock_snapshots()[:5]
    sim.reset(0)
    
    # Check initial state structure
    state = sim.get_current_state()
    
    required_keys = [
        'timestamp', 'best_bid', 'best_ask', 'mid_price',
        'spread', 'spread_pct', 'imbalance', 'depth', 'portfolio'
    ]
    
    for key in required_keys:
        assert key in state, f"Missing key: {key}"
        print(f"✓ State contains '{key}'")
    
    # Verify state values
    assert state['best_bid'] == 50000.0
    assert state['best_ask'] == 50001.0
    assert state['mid_price'] == 50000.5
    assert state['spread'] == 1.0
    assert isinstance(state['depth'], tuple)
    assert len(state['depth']) == 2  # (bids, asks)
    
    # Verify portfolio stats
    portfolio_stats = state['portfolio']
    assert 'cash' in portfolio_stats
    assert 'position' in portfolio_stats
    assert 'total_value' in portfolio_stats
    
    # Step and check state changes
    await sim.step()
    new_state = sim.get_current_state()
    
    assert new_state['best_bid'] == 50010.0
    assert new_state['best_ask'] == 50011.0
    assert new_state['mid_price'] == 50010.5
    print(f"✓ State updated correctly after step")
    
    print("✓ State observations test passed\n")


async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("MarketSimulator Integration Tests")
    print("=" * 60)
    
    try:
        await test_basic_simulation_flow()
        await test_order_placement_and_matching()
        await test_order_cancellation()
        await test_reset_functionality()
        await test_full_simulation_cycle()
        await test_market_making_strategy()
        await test_error_handling()
        await test_state_observations()
        
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
    sys.exit(asyncio.run(main()))

