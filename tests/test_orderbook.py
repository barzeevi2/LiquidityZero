"""
Unit tests for OrderBook
Tests orderbook state management, price calculations, and agent order handling
"""

import pytest
from datetime import datetime
from app.sim.orderbook import OrderBook


class TestOrderBookInitialization:
    """Test OrderBook initialization"""
    
    def test_initialization_with_symbol(self):
        """OrderBook should initialize with symbol"""
        ob = OrderBook("BTC/USDT")
        assert ob.symbol == "BTC/USDT"
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0
        assert len(ob.agent_orders) == 0
        assert ob.last_update_time is None
    
    def test_initialization_empty_orderbook(self):
        """Empty orderbook should have no best prices"""
        ob = OrderBook("ETH/USDT")
        assert ob.get_best_bid() is None
        assert ob.get_best_ask() is None
        assert ob.get_mid_price() is None
        assert ob.get_spread() is None


class TestOrderBookUpdateFromSnapshot:
    """Test updating orderbook from snapshot"""
    
    def test_update_from_valid_snapshot(self):
        """Should update orderbook from valid snapshot"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 3.0]],
            'asks': [[50001.0, 1.2], [50002.0, 2.5], [50003.0, 1.8]],
            'datetime': '2024-01-01T00:00:00+00:00'
        }
        
        ob.update_from_snapshot(snapshot)
        
        assert len(ob.bids) == 3
        assert len(ob.asks) == 3
        assert ob.bids[50000.0] == 1.5
        assert ob.bids[49999.0] == 2.0
        assert ob.asks[50001.0] == 1.2
        assert ob.asks[50002.0] == 2.5
        assert ob.last_update_time is not None
    
    def test_update_from_snapshot_string_prices(self):
        """Should handle string prices in snapshot"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [['50000.0', '1.5'], ['49999.0', '2.0']],
            'asks': [['50001.0', '1.2'], ['50002.0', '2.5']],
        }
        
        ob.update_from_snapshot(snapshot)
        
        assert ob.bids[50000.0] == 1.5
        assert ob.bids[49999.0] == 2.0
        assert ob.asks[50001.0] == 1.2
    
    def test_update_from_snapshot_duplicate_prices(self):
        """Should aggregate quantities for duplicate prices"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5], [50000.0, 2.0]],
            'asks': [[50001.0, 1.2], [50001.0, 0.8]],
        }
        
        ob.update_from_snapshot(snapshot)
        
        assert ob.bids[50000.0] == 3.5
        assert ob.asks[50001.0] == 2.0
    
    def test_update_from_snapshot_without_datetime(self):
        """Should handle snapshot without datetime"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
        }
        
        ob.update_from_snapshot(snapshot)
        
        assert ob.last_update_time is None
    
    def test_update_from_snapshot_clears_existing(self):
        """Should clear existing orderbook data before updating"""
        ob = OrderBook("BTC/USDT")
        
        # Add initial data
        snapshot1 = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot1)
        
        # Update with new snapshot
        snapshot2 = {
            'bids': [[51000.0, 2.0]],
            'asks': [[51100.0, 3.0]],
        }
        ob.update_from_snapshot(snapshot2)
        
        assert len(ob.bids) == 1
        assert len(ob.asks) == 1
        assert 50000.0 not in ob.bids
        assert ob.bids[51000.0] == 2.0


class TestOrderBookBestPrices:
    """Test best bid/ask price retrieval"""
    
    def test_get_best_bid(self):
        """Should return highest bid price"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[49998.0, 3.0], [50000.0, 1.5], [49999.0, 2.0]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        best_bid = ob.get_best_bid()
        assert best_bid == 50000.0
    
    def test_get_best_ask(self):
        """Should return lowest ask price"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50003.0, 1.8], [50001.0, 1.2], [50002.0, 2.5]],
        }
        ob.update_from_snapshot(snapshot)
        
        best_ask = ob.get_best_ask()
        assert best_ask == 50001.0
    
    def test_get_best_bid_empty(self):
        """Should return None when no bids"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_best_bid() is None
    
    def test_get_best_ask_empty(self):
        """Should return None when no asks"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_best_ask() is None


class TestOrderBookPriceCalculations:
    """Test price calculation methods"""
    
    def test_get_mid_price(self):
        """Should calculate mid price correctly"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        mid_price = ob.get_mid_price()
        assert mid_price == 50001.0
    
    def test_get_mid_price_missing_bid(self):
        """Should return None when bid is missing"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_mid_price() is None
    
    def test_get_mid_price_missing_ask(self):
        """Should return None when ask is missing"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_mid_price() is None
    
    def test_get_spread(self):
        """Should calculate spread correctly"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        spread = ob.get_spread()
        assert spread == 2.0
    
    def test_get_spread_missing_prices(self):
        """Should return None when prices are missing"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_spread() is None
    
    def test_get_spread_percentage(self):
        """Should calculate spread percentage correctly"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        spread_pct = ob.get_spread_percentage()
        # spread = 2.0, mid = 50001.0, so spread% = (2.0 / 50001.0) * 100
        expected = (2.0 / 50001.0) * 100
        assert abs(spread_pct - expected) < 0.0001
    
    def test_get_spread_percentage_missing_data(self):
        """Should return None when data is missing"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        assert ob.get_spread_percentage() is None


class TestOrderBookAgentOrders:
    """Test agent order management"""
    
    def test_add_agent_order_buy(self):
        """Should add buy order to bids"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50001.0, 2.0, "BUY")
        
        assert "order1" in ob.agent_orders
        assert ob.agent_orders["order1"] == (50001.0, 2.0, "BUY")
        assert ob.bids[50001.0] == 2.0
    
    def test_add_agent_order_sell(self):
        """Should add sell order to asks"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50003.0, 1.5, "SELL")
        
        assert "order1" in ob.agent_orders
        assert ob.agent_orders["order1"] == (50003.0, 1.5, "SELL")
        assert ob.asks[50003.0] == 1.5
    
    def test_add_agent_order_existing_price(self):
        """Should aggregate quantity at existing price level"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50000.0, 2.0, "BUY")
        
        assert ob.bids[50000.0] == 3.5  # 1.5 + 2.0
    
    def test_add_agent_order_case_insensitive(self):
        """Should handle case-insensitive side"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50001.0, 1.0, "buy")
        ob.add_agent_order("order2", 50003.0, 1.0, "sell")
        
        assert ob.agent_orders["order1"][2] == "buy"
        assert ob.agent_orders["order2"][2] == "sell"
    
    def test_remove_agent_order_buy(self):
        """Should remove buy order from bids"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50001.0, 2.0, "BUY")
        assert 50001.0 in ob.bids
        
        result = ob.remove_agent_order("order1")
        
        assert result is True
        assert "order1" not in ob.agent_orders
        assert 50001.0 not in ob.bids
    
    def test_remove_agent_order_sell(self):
        """Should remove sell order from asks"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50003.0, 1.5, "SELL")
        assert 50003.0 in ob.asks
        
        result = ob.remove_agent_order("order1")
        
        assert result is True
        assert "order1" not in ob.agent_orders
        assert 50003.0 not in ob.asks
    
    def test_remove_agent_order_partial_quantity(self):
        """Should reduce quantity when removing partial order"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50000.0, 2.0, "BUY")
        assert ob.bids[50000.0] == 3.5
        
        result = ob.remove_agent_order("order1")
        
        assert result is True
        assert ob.bids[50000.0] == 1.5  # Back to original snapshot quantity
    
    def test_remove_agent_order_zero_quantity(self):
        """Should remove price level when quantity reaches zero"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50002.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50001.0, 2.0, "BUY")
        
        result = ob.remove_agent_order("order1")
        
        assert result is True
        assert 50001.0 not in ob.bids
    
    def test_remove_agent_order_nonexistent(self):
        """Should return False when order doesn't exist"""
        ob = OrderBook("BTC/USDT")
        
        result = ob.remove_agent_order("nonexistent")
        
        assert result is False


class TestOrderBookDepth:
    """Test market depth calculations"""
    
    def test_get_depth_buy_side(self):
        """Should calculate depth for buy orders (asks at or below price)"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2], [50002.0, 2.5], [50003.0, 1.8]],
        }
        ob.update_from_snapshot(snapshot)
        
        # For buy at 50002.0, should include asks at 50001.0 and 50002.0
        depth = ob.get_depth(50002.0, "BUY")
        assert depth == 3.7  # 1.2 + 2.5
    
    def test_get_depth_sell_side(self):
        """Should calculate depth for sell orders (bids at or above price)"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 3.0]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        # For sell at 49999.0, should include bids at 49999.0 and 50000.0
        depth = ob.get_depth(49999.0, "SELL")
        assert depth == 3.5  # 1.5 + 2.0
    
    def test_get_depth_no_matching_prices(self):
        """Should return 0.0 when no prices match"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        # Buy at price below all asks
        depth = ob.get_depth(40000.0, "BUY")
        assert depth == 0.0
        
        # Sell at price above all bids
        depth = ob.get_depth(60000.0, "SELL")
        assert depth == 0.0


class TestOrderBookImbalance:
    """Test order book imbalance calculation"""
    
    def test_get_imbalance_balanced(self):
        """Should return 0.0 for balanced orderbook"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 5.0]],
            'asks': [[50001.0, 5.0]],
        }
        ob.update_from_snapshot(snapshot)
        
        imbalance = ob.get_imbalance()
        assert imbalance == 0.0
    
    def test_get_imbalance_bullish(self):
        """Should return positive value when more buy volume"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 10.0]],
            'asks': [[50001.0, 2.0]],
        }
        ob.update_from_snapshot(snapshot)
        
        imbalance = ob.get_imbalance()
        # (10 - 2) / (10 + 2) = 8/12 = 0.666...
        expected = 8.0 / 12.0
        assert abs(imbalance - expected) < 0.0001
        assert imbalance > 0
    
    def test_get_imbalance_bearish(self):
        """Should return negative value when more sell volume"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 2.0]],
            'asks': [[50001.0, 10.0]],
        }
        ob.update_from_snapshot(snapshot)
        
        imbalance = ob.get_imbalance()
        # (2 - 10) / (2 + 10) = -8/12 = -0.666...
        expected = -8.0 / 12.0
        assert abs(imbalance - expected) < 0.0001
        assert imbalance < 0
    
    def test_get_imbalance_empty_orderbook(self):
        """Should return 0.0 for empty orderbook"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        imbalance = ob.get_imbalance()
        assert imbalance == 0.0
    
    def test_get_imbalance_top_10_levels(self):
        """Should only consider top 10 levels"""
        ob = OrderBook("BTC/USDT")
        # Create orderbook with more than 10 levels on each side
        bids = [[50000.0 - i, 1.0] for i in range(15)]
        asks = [[50001.0 + i, 1.0] for i in range(15)]
        snapshot = {
            'bids': bids,
            'asks': asks,
        }
        ob.update_from_snapshot(snapshot)
        
        imbalance = ob.get_imbalance()
        # Should only use top 10 levels (last 10 for bids, first 10 for asks)
        # Each side has 10.0 volume, so imbalance should be ~0
        assert abs(imbalance) < 0.1


class TestOrderBookDepthLevels:
    """Test depth levels retrieval"""
    
    def test_get_depth_levels(self):
        """Should return top N levels for each side"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[49998.0, 3.0], [49999.0, 2.0], [50000.0, 1.5]],
            'asks': [[50001.0, 1.2], [50002.0, 2.5], [50003.0, 1.8]],
        }
        ob.update_from_snapshot(snapshot)
        
        bid_levels, ask_levels = ob.get_depth_levels(2)
        
        assert len(bid_levels) == 2
        assert len(ask_levels) == 2
        # Bids should be highest first (descending)
        assert bid_levels[0] == [50000.0, 1.5]
        assert bid_levels[1] == [49999.0, 2.0]
        # Asks should be lowest first (ascending)
        assert ask_levels[0] == [50001.0, 1.2]
        assert ask_levels[1] == [50002.0, 2.5]
    
    def test_get_depth_levels_less_than_n(self):
        """Should return available levels when less than N exist"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        bid_levels, ask_levels = ob.get_depth_levels(5)
        
        assert len(bid_levels) == 1
        assert len(ask_levels) == 1
    
    def test_get_depth_levels_empty_orderbook(self):
        """Should return empty lists for empty orderbook"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [],
            'asks': [],
        }
        ob.update_from_snapshot(snapshot)
        
        bid_levels, ask_levels = ob.get_depth_levels(5)
        
        assert bid_levels == []
        assert ask_levels == []


class TestOrderBookIntegration:
    """Integration tests for OrderBook"""
    
    def test_full_workflow(self):
        """Test complete orderbook workflow"""
        ob = OrderBook("BTC/USDT")
        
        # Initial snapshot
        snapshot1 = {
            'bids': [[50000.0, 1.5], [49999.0, 2.0]],
            'asks': [[50001.0, 1.2], [50002.0, 2.5]],
            'datetime': '2024-01-01T00:00:00+00:00'
        }
        ob.update_from_snapshot(snapshot1)
        
        assert ob.get_best_bid() == 50000.0
        assert ob.get_best_ask() == 50001.0
        
        # Add agent orders
        ob.add_agent_order("buy1", 50000.5, 1.0, "BUY")
        ob.add_agent_order("sell1", 50000.5, 0.8, "SELL")
        
        assert len(ob.agent_orders) == 2
        
        # Update snapshot (agent orders remain)
        snapshot2 = {
            'bids': [[50010.0, 2.0], [50009.0, 1.0]],
            'asks': [[50011.0, 1.5], [50012.0, 2.0]],
        }
        ob.update_from_snapshot(snapshot2)
        
        # Agent orders should still exist
        assert len(ob.agent_orders) == 2
        # But snapshot data should be replaced
        assert 50000.0 not in ob.bids
        assert ob.get_best_bid() == 50010.0
        
        # Remove agent orders
        result1 = ob.remove_agent_order("buy1")
        result2 = ob.remove_agent_order("sell1")
        
        assert result1 is True
        assert result2 is True
        assert len(ob.agent_orders) == 0
    
    def test_multiple_agent_orders_same_price(self):
        """Test multiple agent orders at same price level"""
        ob = OrderBook("BTC/USDT")
        snapshot = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
        }
        ob.update_from_snapshot(snapshot)
        
        ob.add_agent_order("order1", 50000.0, 1.0, "BUY")
        ob.add_agent_order("order2", 50000.0, 2.0, "BUY")
        ob.add_agent_order("order3", 50001.0, 1.5, "SELL")
        
        assert ob.bids[50000.0] == 4.5  # 1.5 + 1.0 + 2.0
        assert ob.asks[50001.0] == 2.7  # 1.2 + 1.5
        
        ob.remove_agent_order("order1")
        assert ob.bids[50000.0] == 3.5  # 1.5 + 2.0
        
        ob.remove_agent_order("order2")
        assert ob.bids[50000.0] == 1.5  # Back to snapshot value
        
        ob.remove_agent_order("order3")
        assert abs(ob.asks[50001.0] - 1.2) < 0.0001  # Back to snapshot value

