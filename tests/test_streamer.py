"""
unit tests for OrderBookStreamer
tests validation, normalization, backoff calculation and error handling
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.data.streamer import OrderBookStreamer
from app.data.exceptions import StreamReconnectionError, DataValidationError


class TestOrderBookValidation:
    """Test orderbook validation logic"""
    
    def test_valid_orderbook_passes(self):
        """Valid orderbook with proper structure should pass"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [[50000.0, 1.5], [49999.0, 2.0]],
            'asks': [[50001.0, 1.2], [50002.0, 3.0]],
            'timestamp': 1234567890
        }
        assert streamer._validate_orderbook(orderbook) is True
    
    def test_missing_required_keys_fails(self):
        """Orderbook missing required keys should fail"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {'bids': [[50000.0, 1.5]]}  # missing 'asks' and 'timestamp'
        assert streamer._validate_orderbook(orderbook) is False
    
    def test_empty_bids_fails(self):
        """Orderbook with empty bids should fail"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [],
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        assert streamer._validate_orderbook(orderbook) is False
    
    def test_invalid_bid_structure_fails(self):
        """Bids with invalid structure should fail"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [[50000.0]],  # missing amount
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        assert streamer._validate_orderbook(orderbook) is False
    
    def test_non_numeric_price_fails(self):
        """Non-numeric prices should fail validation"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [["invalid", 1.5]],
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        assert streamer._validate_orderbook(orderbook) is False
    
    def test_string_amount_fails(self):
        """String amounts should fail validation"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [[50000.0, "invalid"]],
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        assert streamer._validate_orderbook(orderbook) is False


class TestOrderBookNormalization:
    """Test orderbook normalization logic"""
    
    def test_normalization_adds_computed_fields(self):
        """Normalization should add best_bid, best_ask, spread, etc."""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [[50000.0, 1.5], [49999.0, 2.0]],
            'asks': [[50001.0, 1.2], [50002.0, 3.0]],
            'timestamp': 1234567890
        }
        
        normalized = streamer._normalize_orderbook(orderbook)
        
        assert normalized['symbol'] == 'BTC/USDT'
        assert normalized['best_bid'] == 50000.0
        assert normalized['best_ask'] == 50001.0
        assert normalized['spread'] == 1.0
        assert normalized['spread_pct'] == pytest.approx(0.002, rel=1e-3)  # 1/50000 * 100
        assert normalized['bids'] == orderbook['bids']
        assert normalized['asks'] == orderbook['asks']
    
    def test_normalization_handles_missing_datetime(self):
        """Should add datetime if missing from orderbook"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        
        normalized = streamer._normalize_orderbook(orderbook)
        assert 'datetime' in normalized
        assert normalized['datetime'] is not None
    
    def test_normalization_handles_empty_orderbook(self):
        """Should handle edge case of empty bids/asks"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        orderbook = {
            'bids': [],
            'asks': [],
            'timestamp': 1234567890
        }
        
        # This should not crash, but will have None values
        # Note: validation would catch this first, but testing normalization robustness
        normalized = streamer._normalize_orderbook(orderbook)
        assert normalized['best_bid'] is None
        assert normalized['best_ask'] is None
        assert normalized['spread'] is None


class TestBackoffCalculation:
    """Test exponential backoff calculation"""
    
    @pytest.mark.asyncio
    async def test_backoff_exponential_growth(self):
        """Backoff should grow exponentially"""
        streamer = OrderBookStreamer(
            symbol="BTC/USDT",
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=60.0
        )
        
        assert await streamer._calculate_backoff(0) == 1.0
        assert await streamer._calculate_backoff(1) == 2.0
        assert await streamer._calculate_backoff(2) == 4.0
        assert await streamer._calculate_backoff(3) == 8.0
    
    @pytest.mark.asyncio
    async def test_backoff_respects_max(self):
        """Backoff should cap at max_backoff"""
        streamer = OrderBookStreamer(
            symbol="BTC/USDT",
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=10.0
        )
        
        # 2^4 = 16, but should cap at 10
        assert await streamer._calculate_backoff(4) == 10.0
        assert await streamer._calculate_backoff(10) == 10.0


class TestStreamerLifecycle:
    """Test streamer initialization and lifecycle"""
    
    def test_initialization_with_defaults(self):
        """Should initialize with default values"""
        streamer = OrderBookStreamer()
        assert streamer.symbol is not None
        assert streamer.max_reconnect_attempts == 10
        assert streamer.is_running is False
        assert streamer.reconnect_count == 0
    
    def test_initialization_with_custom_symbol(self):
        """Should use provided symbol"""
        streamer = OrderBookStreamer(symbol="ETH/USDT")
        assert streamer.symbol == "ETH/USDT"
    
    @pytest.mark.asyncio
    async def test_stop_sets_is_running_false(self):
        """Stop should set is_running to False"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        streamer.is_running = True
        streamer.exchange = AsyncMock()
        streamer.exchange.close = AsyncMock()
        
        await streamer.stop()
        
        assert streamer.is_running is False


class TestStreamerIntegration:
    """Integration tests with mocked exchange"""
    
    @pytest.mark.asyncio
    async def test_stream_yields_normalized_data(self):
        """Stream should yield normalized orderbook data"""
        streamer = OrderBookStreamer(symbol="BTC/USDT", max_reconnect_attempts=1)
        
        # Mock exchange and orderbook data
        mock_exchange = AsyncMock()
        mock_orderbook = {
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]],
            'timestamp': 1234567890
        }
        
        # Create async generator that yields one orderbook then stops
        async def mock_watch_orderbook(symbol):
            yield mock_orderbook
            streamer.is_running = False  # Stop after first yield
        
        mock_exchange.watch_order_book = mock_watch_orderbook
        mock_exchange.close = AsyncMock()
        
        with patch.object(streamer, '_create_exchange', return_value=mock_exchange):
            results = []
            async for data in streamer.stream():
                results.append(data)
                break  # Only take first result
        
        assert len(results) == 1
        assert results[0]['symbol'] == 'BTC/USDT'
        assert results[0]['best_bid'] == 50000.0
    
    @pytest.mark.asyncio
    async def test_stream_skips_invalid_data(self):
        """Stream should skip invalid orderbook data"""
        streamer = OrderBookStreamer(symbol="BTC/USDT", max_reconnect_attempts=1)
        
        mock_exchange = AsyncMock()
        
        # First invalid, then valid
        async def mock_watch_orderbook(symbol):
            yield {'invalid': 'data'}  # Invalid
            yield {
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]],
                'timestamp': 1234567890
            }  # Valid
            streamer.is_running = False
        
        mock_exchange.watch_order_book = mock_watch_orderbook
        mock_exchange.close = AsyncMock()
        
        with patch.object(streamer, '_create_exchange', return_value=mock_exchange):
            results = []
            async for data in streamer.stream():
                results.append(data)
                break
        
        # Should only have the valid one
        assert len(results) == 1
        assert results[0]['best_bid'] == 50000.0
    
    @pytest.mark.asyncio
    async def test_stream_reconnects_on_error(self):
        """Stream should attempt reconnection on error"""
        streamer = OrderBookStreamer(
            symbol="BTC/USDT",
            max_reconnect_attempts=2,
            initial_backoff=0.1  # Fast for testing
        )
        
        mock_exchange = AsyncMock()
        call_count = 0
        
        async def mock_watch_orderbook(symbol):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection lost")
            # Second attempt succeeds
            yield {
                'bids': [[50000.0, 1.5]],
                'asks': [[50001.0, 1.2]],
                'timestamp': 1234567890
            }
            streamer.is_running = False
        
        mock_exchange.watch_order_book = mock_watch_orderbook
        mock_exchange.close = AsyncMock()
        
        with patch.object(streamer, '_create_exchange', return_value=mock_exchange):
            with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
                results = []
                async for data in streamer.stream():
                    results.append(data)
                    break
        
        assert call_count == 2  # Should have reconnected
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_stream_raises_after_max_reconnects(self):
        """Stream should raise StreamReconnectionError after max attempts"""
        streamer = OrderBookStreamer(
            symbol="BTC/USDT",
            max_reconnect_attempts=2,
            initial_backoff=0.1
        )
        
        mock_exchange = AsyncMock()
        mock_exchange.watch_order_book = AsyncMock(side_effect=ConnectionError("Persistent error"))
        mock_exchange.close = AsyncMock()
        
        with patch.object(streamer, '_create_exchange', return_value=mock_exchange):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(StreamReconnectionError):
                    async for _ in streamer.stream():
                        pass
    
    #wrote the test with AI
    @pytest.mark.asyncio
    async def test_stream_handles_cancellation(self):
        """Stream should handle CancelledError gracefully"""
        streamer = OrderBookStreamer(symbol="BTC/USDT")
        
        mock_exchange = AsyncMock()
        cancellation_occurred = False
        
        # watch_order_book must return an async iterator
        #create a custom async iterator class that raises CancelledError
        class CancellingAsyncIterator:
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                nonlocal cancellation_occurred
                cancellation_occurred = True
                raise asyncio.CancelledError()
        
        mock_exchange.watch_order_book = lambda symbol: CancellingAsyncIterator()
        mock_exchange.close = AsyncMock()
        
        with patch.object(streamer, '_create_exchange', return_value=mock_exchange):
            results = []
            
            # Collect with timeout as safety net
            try:
                async def collect_stream():
                    async for data in streamer.stream():
                        results.append(data)
                
                # If cancellation is handled properly, this should complete quickly (< 0.5s)
                # If it times out, cancellation wasn't handled
                await asyncio.wait_for(collect_stream(), timeout=0.5)
            except asyncio.TimeoutError:
                pytest.fail("Stream didn't handle CancelledError - generator didn't close")
            except asyncio.CancelledError:
                pytest.fail("CancelledError propagated - streamer didn't catch it")
        
        # Verify cancellation actually occurred
        assert cancellation_occurred, "CancelledError was never raised"
        
        # Verify streamer handled it gracefully (no results yielded)
        assert len(results) == 0
        
        # Verify cleanup happened
        assert mock_exchange.close.called, "Exchange should be closed after cancellation"