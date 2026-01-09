"""
Unit tests for MatchingEngine
Tests order matching, fee calculations, and slippage calculations
"""

import pytest
from app.sim.matching_engine import MatchingEngine
from app.sim.orderbook import OrderBook
from app.sim.orders import LimitOrder, MarketOrder, OrderSide, OrderStatus
from app.sim.exceptions import InsufficientLiquidityError


class TestMatchingEngineInitialization:
    """Test MatchingEngine initialization"""
    
    def test_initialization_default_fees(self):
        """Should initialize with default fee rates"""
        engine = MatchingEngine()
        assert engine.maker_fee_rate == -0.0001
        assert engine.taker_fee_rate == 0.0001
    
    def test_initialization_custom_fees(self):
        """Should initialize with custom fee rates"""
        engine = MatchingEngine(maker_fee_rate=-0.0002, taker_fee_rate=0.0002)
        assert engine.maker_fee_rate == -0.0002
        assert engine.taker_fee_rate == 0.0002


class TestLimitOrderMatching:
    """Test limit order matching"""
    
    def test_limit_buy_no_match_price_too_low(self):
        """Limit buy should not match if price is below best ask"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0], [50010.0, 2.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=49900.0  # Below best ask
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
    
    def test_limit_buy_full_fill_single_level(self):
        """Limit buy should fully fill at single price level"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0], [50010.0, 2.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].quantity == 1.0
        assert fills[0].price == 50000.0
        assert fills[0].fee == 1.0 * -0.0001 * 50000.0  # Maker fee (rebate)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert 50000.0 not in order_book.asks  # Level should be removed
    
    def test_limit_buy_partial_fill(self):
        """Limit buy should partially fill if orderbook quantity is less"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 0.5], [50010.0, 2.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].quantity == 0.5
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 0.5
        assert 50000.0 not in order_book.asks  # Should be removed when quantity reaches 0
    
    def test_limit_buy_fill_multiple_levels(self):
        """Limit buy should fill through multiple price levels"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 0.5], [50005.0, 0.3], [50010.0, 2.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50010.0  # Can fill through first two levels
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        # Order can fill through all levels up to price limit (50010.0)
        assert len(fills) == 3
        assert fills[0].quantity == 0.5
        assert fills[0].price == 50000.0
        assert fills[1].quantity == 0.3
        assert fills[1].price == 50005.0
        assert fills[2].quantity == 0.2
        assert fills[2].price == 50010.0
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert 50000.0 not in order_book.asks
        assert 50005.0 not in order_book.asks
        assert order_book.asks[50010.0] == 1.8  # 2.0 - 0.2
    
    def test_limit_sell_no_match_price_too_high(self):
        """Limit sell should not match if price is above best bid"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 1.0], [49990.0, 2.0]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=50100.0  # Above best bid
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING
    
    def test_limit_sell_full_fill_single_level(self):
        """Limit sell should fully fill at single price level"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 1.0], [49990.0, 2.0]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=50000.0
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].quantity == 1.0
        assert fills[0].price == 50000.0
        assert fills[0].fee == 1.0 * -0.0001 * 50000.0  # Maker fee (rebate)
        assert order.status == OrderStatus.FILLED
        assert 50000.0 not in order_book.bids
    
    def test_limit_sell_fill_multiple_levels(self):
        """Limit sell should fill through multiple price levels"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 0.5], [49995.0, 0.3], [49990.0, 2.0]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            price=49990.0  # Can fill through first two levels
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        # Order can fill through all levels down to price limit (49990.0)
        assert len(fills) == 3
        assert fills[0].quantity == 0.5
        assert fills[0].price == 50000.0
        assert fills[1].quantity == 0.3
        assert fills[1].price == 49995.0
        assert fills[2].quantity == 0.2
        assert fills[2].price == 49990.0
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order_book.bids[49990.0] == 1.8  # 2.0 - 0.2
    
    def test_limit_order_empty_orderbook(self):
        """Limit order should return empty fills if orderbook is empty"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING


class TestMarketOrderMatching:
    """Test market order matching"""
    
    def test_market_buy_full_fill_single_level(self):
        """Market buy should fully fill at best available price"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0], [50010.0, 2.0]],
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        fills = engine.match_market_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].quantity == 1.0
        assert fills[0].price == 50000.0
        assert fills[0].fee == 1.0 * 50000.0 * 0.0001  # Taker fee
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
    
    def test_market_buy_fill_multiple_levels(self):
        """Market buy should fill through multiple price levels"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 0.5], [50005.0, 0.3], [50010.0, 0.3]],  # Total 1.1 to ensure enough liquidity
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        fills = engine.match_market_order(order, order_book)
        
        assert len(fills) == 3
        assert fills[0].quantity == 0.5
        assert fills[0].price == 50000.0
        assert fills[1].quantity == 0.3
        assert fills[1].price == 50005.0
        assert fills[2].quantity == 0.2
        assert fills[2].price == 50010.0
        assert order.status == OrderStatus.FILLED
        assert abs(order_book.asks[50010.0] - 0.1) < 0.0001  # 0.3 - 0.2 (floating point tolerance)
    
    def test_market_buy_insufficient_liquidity(self):
        """Market buy should raise error if insufficient liquidity"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 0.5]],  # Less than order quantity
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        with pytest.raises(InsufficientLiquidityError):
            engine.match_market_order(order, order_book)
    
    def test_market_buy_empty_orderbook(self):
        """Market buy should raise error if orderbook is empty"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        with pytest.raises(InsufficientLiquidityError):
            engine.match_market_order(order, order_book)
    
    def test_market_sell_full_fill_single_level(self):
        """Market sell should fully fill at best available price"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 1.0], [49990.0, 2.0]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0
        )
        
        fills = engine.match_market_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].quantity == 1.0
        assert fills[0].price == 50000.0
        assert fills[0].fee == 1.0 * 50000.0 * 0.0001  # Taker fee
        assert order.status == OrderStatus.FILLED
    
    def test_market_sell_fill_multiple_levels(self):
        """Market sell should fill through multiple price levels"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 0.5], [49995.0, 0.3], [49990.0, 0.2]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0
        )
        
        fills = engine.match_market_order(order, order_book)
        
        assert len(fills) == 3
        assert fills[0].quantity == 0.5
        assert fills[0].price == 50000.0
        assert fills[1].quantity == 0.3
        assert fills[1].price == 49995.0
        assert fills[2].quantity == 0.2
        assert fills[2].price == 49990.0
        assert order.status == OrderStatus.FILLED
    
    def test_market_sell_insufficient_liquidity(self):
        """Market sell should raise error if insufficient liquidity"""
        engine = MatchingEngine()
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[50000.0, 0.5]],
            'asks': [[50100.0, 1.0]],
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0
        )
        
        with pytest.raises(InsufficientLiquidityError):
            engine.match_market_order(order, order_book)


class TestFeeCalculations:
    """Test fee calculations"""
    
    def test_maker_fee_negative_rebate(self):
        """Maker fee should be negative (rebate)"""
        engine = MatchingEngine(maker_fee_rate=-0.0001, taker_fee_rate=0.0001)
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0]],
        })
        
        order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )
        
        fills = engine.match_limit_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].fee < 0  # Negative fee (rebate)
        assert fills[0].fee == pytest.approx(-5.0)  # 1.0 * -0.0001 * 50000.0
    
    def test_taker_fee_positive_cost(self):
        """Taker fee should be positive (cost)"""
        engine = MatchingEngine(maker_fee_rate=-0.0001, taker_fee_rate=0.0001)
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0]],
        })
        
        order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        fills = engine.match_market_order(order, order_book)
        
        assert len(fills) == 1
        assert fills[0].fee > 0  # Positive fee (cost)
        assert fills[0].fee == pytest.approx(5.0)  # 1.0 * 50000.0 * 0.0001
    
    def test_custom_fee_rates(self):
        """Should use custom fee rates when provided"""
        engine = MatchingEngine(maker_fee_rate=-0.0002, taker_fee_rate=0.0002)
        order_book = OrderBook("BTC/USDT")
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0]],
        })
        
        limit_order = LimitOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )
        
        market_order = MarketOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
        
        limit_fills = engine.match_limit_order(limit_order, order_book)
        order_book.update_from_snapshot({
            'bids': [[49900.0, 1.0]],
            'asks': [[50000.0, 1.0]],
        })
        market_fills = engine.match_market_order(market_order, order_book)
        
        assert limit_fills[0].fee == pytest.approx(-10.0)  # 1.0 * -0.0002 * 50000.0
        assert market_fills[0].fee == pytest.approx(10.0)  # 1.0 * 50000.0 * 0.0002


class TestSlippageCalculations:
    """Test slippage calculations"""
    
    def test_slippage_no_fills(self):
        """Slippage should be 0.0 if no fills"""
        engine = MatchingEngine()
        slippage = engine.calculate_slippage([], 50000.0, OrderSide.BUY)
        assert slippage == 0.0
    
    def test_slippage_buy_no_slippage(self):
        """Slippage should be 0.0 if fill price equals expected price"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                price=50000.0,
                fee=0.0
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.BUY)
        assert slippage == pytest.approx(0.0)
    
    def test_slippage_buy_positive_slippage(self):
        """Buy slippage should be positive if fill price is higher"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                price=50050.0,  # Higher than expected
                fee=0.0
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.BUY)
        assert slippage == pytest.approx(0.001)  # (50050 - 50000) / 50000
    
    def test_slippage_buy_negative_slippage(self):
        """Buy slippage should be negative if fill price is lower (better)"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                price=49950.0,  # Lower than expected (better)
                fee=0.0
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.BUY)
        assert slippage == pytest.approx(-0.001)  # (49950 - 50000) / 50000
    
    def test_slippage_sell_positive_slippage(self):
        """Sell slippage should be positive if fill price is lower (worse)"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                quantity=1.0,
                price=49950.0,  # Lower than expected (worse for sell)
                fee=0.0
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.SELL)
        assert slippage == pytest.approx(0.001)  # (50000 - 49950) / 50000
    
    def test_slippage_multiple_fills(self):
        """Slippage should calculate average fill price across multiple fills"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.5,
                price=50000.0,
                fee=0.0
            ),
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.5,
                price=50050.0,
                fee=0.0
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.BUY)
        # Average price: (0.5 * 50000 + 0.5 * 50050) / 1.0 = 50025
        # Slippage: (50025 - 50000) / 50000 = 0.0005
        assert slippage == pytest.approx(0.0005)
    
    def test_slippage_ignores_fees(self):
        """Slippage calculation should ignore fees (use price only)"""
        engine = MatchingEngine()
        from app.sim.orders import Fill
        
        fills = [
            Fill(
                order_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                price=50000.0,
                fee=100.0  # Fee should not affect slippage
            )
        ]
        
        slippage = engine.calculate_slippage(fills, 50000.0, OrderSide.BUY)
        assert slippage == pytest.approx(0.0)  # Should be 0, not affected by fee

