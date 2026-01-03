"""
Unit tests for RedisCache
Tests initialization, caching operations, TTL, error handling, and statistics
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone

from app.data.cache import RedisCache
from app.data.exceptions import CacheError


class TestRedisCacheInitialization:
    """Test RedisCache initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default values"""
        cache = RedisCache()
        assert cache.ttl == 60
        assert cache.max_retries == 3
        assert cache.client is None
        assert cache.total_updates == 0
        assert cache.total_failed == 0
    
    def test_custom_initialization(self):
        """Should accept custom parameters"""
        cache = RedisCache(
            redis_url="redis://localhost:6379/1",
            ttl=120,
            max_retries=5
        )
        assert cache.redis_url == "redis://localhost:6379/1"
        assert cache.ttl == 120
        assert cache.max_retries == 5
    
    def test_initialization_uses_settings_default(self):
        """Should use settings.REDIS_URL if redis_url not provided"""
        with patch('app.data.cache.settings') as mock_settings:
            mock_settings.REDIS_URL = "redis://default:6379/0"
            cache = RedisCache()
            assert cache.redis_url == "redis://default:6379/0"
    
    def test_initialization_prefers_provided_redis_url(self):
        """Should prefer provided redis_url over settings"""
        with patch('app.data.cache.settings') as mock_settings:
            mock_settings.REDIS_URL = "redis://default:6379/0"
            cache = RedisCache(redis_url="redis://custom:6379/0")
            assert cache.redis_url == "redis://custom:6379/0"


class TestRedisCacheInitialize:
    """Test RedisCache.initialize() method"""
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should successfully initialize Redis client"""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        
        mock_client = AsyncMock()
        async def mock_from_url(*args, **kwargs):
            return mock_client
        
        with patch('app.data.cache.redis.from_url', side_effect=mock_from_url):
            await cache.initialize()
            
            assert cache.client == mock_client
    
    @pytest.mark.asyncio
    async def test_initialize_raises_cache_error_on_failure(self):
        """Should raise CacheError when initialization fails"""
        cache = RedisCache()
        
        with patch('app.data.cache.redis.from_url', side_effect=Exception("Connection refused")):
            with pytest.raises(CacheError) as exc_info:
                await cache.initialize()
            
            assert "Failed to initialize Redis cache" in str(exc_info.value)
            assert cache.client is None


class TestRedisCacheUpdateLatest:
    """Test RedisCache.update_latest() method"""
    
    @pytest.fixture
    def sample_snapshot(self):
        """Sample orderbook snapshot for testing"""
        return {
            'symbol': 'BTC/USDT',
            'timestamp': 1234567890,
            'bids': [[50000.0, 1.5], [49999.0, 2.0]],
            'asks': [[50001.0, 1.2], [50002.0, 2.5]],
            'best_bid': 50000.0,
            'best_ask': 50001.0,
            'spread': 1.0,
            'spread_pct': 0.00002
        }
    
    @pytest.mark.asyncio
    async def test_update_latest_success(self, sample_snapshot):
        """Should successfully update cache with orderbook snapshot"""
        cache = RedisCache(ttl=60)
        mock_client = AsyncMock()
        cache.client = mock_client
        
        result = await cache.update_latest(sample_snapshot)
        
        assert result is True
        assert cache.total_updates == 1
        assert cache.total_failed == 0
        
        # Verify setex was called with correct parameters
        expected_key = "orderbook:BTC/USDT"
        expected_value = json.dumps(sample_snapshot).encode('utf-8')
        mock_client.setex.assert_called_once_with(expected_key, 60, expected_value)
    
    @pytest.mark.asyncio
    async def test_update_latest_without_initialization(self, sample_snapshot):
        """Should return False if client not initialized"""
        cache = RedisCache()
        # client is None by default
        
        result = await cache.update_latest(sample_snapshot)
        
        assert result is False
        assert cache.total_updates == 0
        assert cache.total_failed == 0
    
    @pytest.mark.asyncio
    async def test_update_latest_handles_exception(self, sample_snapshot):
        """Should handle exceptions gracefully and increment total_failed"""
        cache = RedisCache(ttl=60)
        mock_client = AsyncMock()
        mock_client.setex.side_effect = Exception("Redis connection error")
        cache.client = mock_client
        
        result = await cache.update_latest(sample_snapshot)
        
        assert result is False
        assert cache.total_updates == 0
        assert cache.total_failed == 1
    
    @pytest.mark.asyncio
    async def test_update_latest_uses_custom_ttl(self, sample_snapshot):
        """Should use the configured TTL value"""
        cache = RedisCache(ttl=120)
        mock_client = AsyncMock()
        cache.client = mock_client
        
        await cache.update_latest(sample_snapshot)
        
        # Verify TTL was passed correctly
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 120  # TTL is the second argument
    
    @pytest.mark.asyncio
    async def test_update_latest_multiple_symbols(self):
        """Should handle multiple different symbols"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        
        snapshot1 = {'symbol': 'BTC/USDT', 'timestamp': 1}
        snapshot2 = {'symbol': 'ETH/USDT', 'timestamp': 2}
        
        await cache.update_latest(snapshot1)
        await cache.update_latest(snapshot2)
        
        assert cache.total_updates == 2
        assert mock_client.setex.call_count == 2
        
        # Verify different keys were used
        calls = mock_client.setex.call_args_list
        assert "orderbook:BTC/USDT" in str(calls[0])
        assert "orderbook:ETH/USDT" in str(calls[1])


class TestRedisCacheGetLatest:
    """Test RedisCache.get_latest() method"""
    
    @pytest.mark.asyncio
    async def test_get_latest_success(self):
        """Should successfully retrieve cached orderbook snapshot"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        
        sample_data = {
            'symbol': 'BTC/USDT',
            'timestamp': 1234567890,
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]]
        }
        encoded_value = json.dumps(sample_data).encode('utf-8')
        mock_client.get.return_value = encoded_value
        
        result = await cache.get_latest('BTC/USDT')
        
        assert result == sample_data
        mock_client.get.assert_called_once_with("orderbook:BTC/USDT")
    
    @pytest.mark.asyncio
    async def test_get_latest_not_found(self):
        """Should return None when key doesn't exist"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        mock_client.get.return_value = None
        
        result = await cache.get_latest('BTC/USDT')
        
        assert result is None
        mock_client.get.assert_called_once_with("orderbook:BTC/USDT")
    
    @pytest.mark.asyncio
    async def test_get_latest_without_initialization(self):
        """Should return None if client not initialized"""
        cache = RedisCache()
        # client is None by default
        
        result = await cache.get_latest('BTC/USDT')
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_latest_handles_exception(self):
        """Should handle exceptions gracefully and return None"""
        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Redis connection error")
        cache.client = mock_client
        
        result = await cache.get_latest('BTC/USDT')
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_latest_invalid_json(self):
        """Should handle invalid JSON gracefully"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        mock_client.get.return_value = b"invalid json data"
        
        result = await cache.get_latest('BTC/USDT')
        
        # Should return None due to JSON decode error
        assert result is None


class TestRedisCacheClose:
    """Test RedisCache.close() method"""
    
    @pytest.mark.asyncio
    async def test_close_success(self):
        """Should successfully close Redis client"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        
        await cache.close()
        
        mock_client.close.assert_called_once()
        assert cache.client is None
    
    @pytest.mark.asyncio
    async def test_close_without_initialization(self):
        """Should handle close when client is None"""
        cache = RedisCache()
        # client is None by default
        
        # Should not raise an exception
        await cache.close()
        assert cache.client is None


class TestRedisCacheGetStats:
    """Test RedisCache.get_stats() method"""
    
    def test_get_stats_initial(self):
        """Should return initial statistics"""
        cache = RedisCache(ttl=60)
        stats = cache.get_stats()
        
        assert stats == {
            'total_updates': 0,
            'total_failed': 0,
            'ttl': 60
        }
    
    def test_get_stats_after_operations(self):
        """Should return updated statistics after operations"""
        cache = RedisCache(ttl=120)
        cache.total_updates = 10
        cache.total_failed = 2
        
        stats = cache.get_stats()
        
        assert stats == {
            'total_updates': 10,
            'total_failed': 2,
            'ttl': 120
        }


class TestRedisCacheIntegration:
    """Integration tests for RedisCache workflow"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow: initialize -> update -> get -> close"""
        cache = RedisCache(ttl=60)
        mock_client = AsyncMock()
        
        sample_snapshot = {
            'symbol': 'BTC/USDT',
            'timestamp': 1234567890,
            'bids': [[50000.0, 1.5]],
            'asks': [[50001.0, 1.2]]
        }
        encoded_value = json.dumps(sample_snapshot).encode('utf-8')
        mock_client.get.return_value = encoded_value
        
        async def mock_from_url(*args, **kwargs):
            return mock_client
        
        with patch('app.data.cache.redis.from_url', side_effect=mock_from_url):
            # Initialize
            await cache.initialize()
            assert cache.client is not None
            
            # Update
            result = await cache.update_latest(sample_snapshot)
            assert result is True
            assert cache.total_updates == 1
            
            # Get
            retrieved = await cache.get_latest('BTC/USDT')
            assert retrieved == sample_snapshot
            
            # Close
            await cache.close()
            assert cache.client is None
    
    @pytest.mark.asyncio
    async def test_update_and_retrieve_multiple_symbols(self):
        """Test caching multiple symbols and retrieving them"""
        cache = RedisCache()
        mock_client = AsyncMock()
        cache.client = mock_client
        
        snapshots = [
            {'symbol': 'BTC/USDT', 'price': 50000},
            {'symbol': 'ETH/USDT', 'price': 3000},
            {'symbol': 'SOL/USDT', 'price': 100}
        ]
        
        # Update all
        for snapshot in snapshots:
            await cache.update_latest(snapshot)
        
        assert cache.total_updates == 3
        
        # Mock get to return different values
        def mock_get(key):
            if key == "orderbook:BTC/USDT":
                return json.dumps(snapshots[0]).encode('utf-8')
            elif key == "orderbook:ETH/USDT":
                return json.dumps(snapshots[1]).encode('utf-8')
            elif key == "orderbook:SOL/USDT":
                return json.dumps(snapshots[2]).encode('utf-8')
            return None
        
        mock_client.get.side_effect = mock_get
        
        # Retrieve all
        for snapshot in snapshots:
            result = await cache.get_latest(snapshot['symbol'])
            assert result == snapshot
    
    @pytest.mark.asyncio
    async def test_ttl_expiration_simulation(self):
        """Test that TTL is properly set (simulated by checking call arguments)"""
        cache = RedisCache(ttl=30)
        mock_client = AsyncMock()
        cache.client = mock_client
        
        snapshot = {'symbol': 'BTC/USDT', 'timestamp': 1}
        await cache.update_latest(snapshot)
        
        # Verify TTL was set to 30 seconds
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 30  # TTL is second argument

