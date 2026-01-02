"""
Unit tests for StorageEngine
Tests initialization, schema creation, batch writes, retry logic, and error handling
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timezone
import json

from app.data.storage import StorageEngine
from app.data.exceptions import StorageError


class TestStorageEngineInitialization:
    """Test StorageEngine initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default values"""
        engine = StorageEngine()
        assert engine.pool_size == 5
        assert engine.max_retries == 3
        assert engine.initial_backoff == 1.0
        assert engine.pool is None
        assert engine.total_written == 0
        assert engine.total_failed == 0
    
    def test_custom_initialization(self):
        """Should accept custom parameters"""
        engine = StorageEngine(
            connection_string="postgresql://test",
            pool_size=10,
            max_retries=5,
            initial_backoff=2.0
        )
        assert engine.connection_string == "postgresql://test"
        assert engine.pool_size == 10
        assert engine.max_retries == 5
        assert engine.initial_backoff == 2.0
    
    def test_initialization_uses_settings_default(self):
        """Should use settings.TIMESCALE_URL if connection_string not provided"""
        with patch('app.data.storage.settings') as mock_settings:
            mock_settings.TIMESCALE_URL = "postgresql://default"
            engine = StorageEngine()
            assert engine.connection_string == "postgresql://default"
    
    def test_initialization_prefers_provided_connection_string(self):
        """Should prefer provided connection_string over settings"""
        with patch('app.data.storage.settings') as mock_settings:
            mock_settings.TIMESCALE_URL = "postgresql://default"
            engine = StorageEngine(connection_string="postgresql://custom")
            assert engine.connection_string == "postgresql://custom"


class TestStorageEngineInitialize:
    """Test StorageEngine.initialize()"""
    
    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self):
        """Should create connection pool on initialize"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool.return_value = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.return_value.acquire = AsyncMock(return_value=mock_conn.__aenter__())
            mock_pool.return_value.acquire.return_value = mock_conn
            
            # Mock the connection context manager
            async def acquire():
                return mock_conn
            mock_pool.return_value.acquire = MagicMock(return_value=mock_conn)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute = AsyncMock()
            
            await engine.initialize()
            
            assert engine.pool is not None
            mock_pool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self):
        """Should create schema on initialize"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            await engine.initialize()
            
            # Should call execute twice (table creation and hypertable creation)
            assert mock_conn.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_initialize_raises_on_failure(self):
        """Should raise StorageError on initialization failure"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool.side_effect = Exception("Connection failed")
            
            with pytest.raises(StorageError) as exc_info:
                await engine.initialize()
            
            assert "Failed to initialize" in str(exc_info.value)


class TestBackoffCalculation:
    """Test exponential backoff calculation"""
    
    @pytest.mark.asyncio
    async def test_backoff_exponential_growth(self):
        """Backoff should grow exponentially"""
        engine = StorageEngine(initial_backoff=1.0)
        
        assert await engine._calculate_backoff(0) == 1.0
        assert await engine._calculate_backoff(1) == 2.0
        assert await engine._calculate_backoff(2) == 4.0
        assert await engine._calculate_backoff(3) == 8.0
    
    @pytest.mark.asyncio
    async def test_backoff_custom_initial(self):
        """Backoff should respect custom initial_backoff"""
        engine = StorageEngine(initial_backoff=2.0)
        
        assert await engine._calculate_backoff(0) == 2.0
        assert await engine._calculate_backoff(1) == 4.0
        assert await engine._calculate_backoff(2) == 8.0


class TestWriteBatch:
    """Test _write_batch method"""
    
    def create_sample_snapshot(self, index: int = 0) -> dict:
        """Create a sample orderbook snapshot"""
        base_price = 50000.0 + (index * 0.01)
        return {
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000) + index,
            'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0]],
            'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 3.0]],
            'best_bid': base_price - 0.01,
            'best_ask': base_price + 0.01,
            'spread': 0.02,
            'spread_pct': 0.00004
        }
    
    @pytest.mark.asyncio
    async def test_write_batch_empty_list(self):
        """Should return 0 for empty snapshot list"""
        engine = StorageEngine()
        result = await engine._write_batch([])
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_write_batch_not_initialized(self):
        """Should raise StorageError if not initialized"""
        engine = StorageEngine()
        snapshots = [self.create_sample_snapshot()]
        
        with pytest.raises(StorageError) as exc_info:
            await engine._write_batch(snapshots)
        
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_write_batch_single_snapshot(self):
        """Should write a single snapshot successfully"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value="INSERT 0 1")
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        result = await engine._write_batch([snapshot])
        
        assert result == 1
        assert engine.total_written == 1
        assert mock_conn.executemany.call_count == 1
    
    @pytest.mark.asyncio
    async def test_write_batch_multiple_snapshots(self):
        """Should write multiple snapshots in a single batch"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value="INSERT 0 3")
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshots = [self.create_sample_snapshot(i) for i in range(3)]
        result = await engine._write_batch(snapshots)
        
        assert result == 3
        assert engine.total_written == 3
        # Should call executemany once with all rows
        assert mock_conn.executemany.call_count == 1
        # Check that it was called with 3 rows
        call_args = mock_conn.executemany.call_args
        assert len(call_args[0][1]) == 3
    
    @pytest.mark.asyncio
    async def test_write_batch_timestamp_parsing_milliseconds(self):
        """Should parse timestamp in milliseconds correctly"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        # Timestamp in milliseconds (> 1e10)
        snapshot = self.create_sample_snapshot()
        snapshot['timestamp'] = 1234567890123  # milliseconds
        
        await engine._write_batch([snapshot])
        
        # Check that timestamp was parsed correctly
        call_args = mock_conn.executemany.call_args
        rows = call_args[0][1]
        assert len(rows) == 1
        ts = rows[0][0]
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc
    
    @pytest.mark.asyncio
    async def test_write_batch_timestamp_parsing_seconds(self):
        """Should parse timestamp in seconds correctly"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        # Timestamp in seconds (< 1e10)
        snapshot = self.create_sample_snapshot()
        snapshot['timestamp'] = 1234567890  # seconds
        
        await engine._write_batch([snapshot])
        
        call_args = mock_conn.executemany.call_args
        rows = call_args[0][1]
        ts = rows[0][0]
        assert isinstance(ts, datetime)
    
    @pytest.mark.asyncio
    async def test_write_batch_datetime_string(self):
        """Should parse datetime string correctly"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        del snapshot['timestamp']
        snapshot['datetime'] = '2024-01-01T00:00:00Z'
        
        await engine._write_batch([snapshot])
        
        call_args = mock_conn.executemany.call_args
        rows = call_args[0][1]
        ts = rows[0][0]
        assert isinstance(ts, datetime)
    
    @pytest.mark.asyncio
    async def test_write_batch_no_timestamp_uses_current_time(self):
        """Should use current time if no timestamp or datetime provided"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        del snapshot['timestamp']
        # No datetime either
        
        before = datetime.now(timezone.utc)
        await engine._write_batch([snapshot])
        after = datetime.now(timezone.utc)
        
        call_args = mock_conn.executemany.call_args
        rows = call_args[0][1]
        ts = rows[0][0]
        assert isinstance(ts, datetime)
        assert before <= ts <= after
    
    @pytest.mark.asyncio
    async def test_write_batch_json_serialization(self):
        """Should serialize bids and asks as JSON"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        await engine._write_batch([snapshot])
        
        call_args = mock_conn.executemany.call_args
        rows = call_args[0][1]
        bids_json = rows[0][2]
        asks_json = rows[0][3]
        
        assert isinstance(bids_json, str)
        assert isinstance(asks_json, str)
        # Verify it's valid JSON
        assert json.loads(bids_json) == snapshot['bids']
        assert json.loads(asks_json) == snapshot['asks']
    
    @pytest.mark.asyncio
    async def test_write_batch_retries_on_failure(self):
        """Should retry on failure with exponential backoff"""
        engine = StorageEngine(
            connection_string="postgresql://test",
            max_retries=3,
            initial_backoff=0.1
        )
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        # First two attempts fail, third succeeds
        call_count = 0
        async def mock_executemany(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Database error")
            return "INSERT 0 1"
        
        mock_conn.executemany = mock_executemany
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await engine._write_batch([snapshot])
        
        assert result == 1
        assert call_count == 3  # Should have retried
    
    @pytest.mark.asyncio
    async def test_write_batch_raises_after_max_retries(self):
        """Should raise StorageError after max retries"""
        engine = StorageEngine(
            connection_string="postgresql://test",
            max_retries=2,
            initial_backoff=0.1
        )
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(side_effect=Exception("Persistent error"))
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshot = self.create_sample_snapshot()
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(StorageError) as exc_info:
                await engine._write_batch([snapshot])
        
        assert "Failed to write" in str(exc_info.value)
        # total_failed is incremented on each retry attempt, so with max_retries=2 it will be 2
        assert engine.total_failed == 2
    
    @pytest.mark.asyncio
    async def test_write_batch_tracks_failed_count(self):
        """Should track failed writes correctly"""
        engine = StorageEngine(
            connection_string="postgresql://test",
            max_retries=1,
            initial_backoff=0.1
        )
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(side_effect=Exception("Error"))
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        snapshots = [self.create_sample_snapshot(i) for i in range(3)]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            try:
                await engine._write_batch(snapshots)
            except StorageError:
                pass
        
        assert engine.total_failed == 3  # All 3 snapshots failed


class TestStorageEngineClose:
    """Test StorageEngine.close()"""
    
    @pytest.mark.asyncio
    async def test_close_with_pool(self):
        """Should close pool if it exists"""
        engine = StorageEngine()
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        engine.pool = mock_pool
        
        await engine.close()
        
        mock_pool.close.assert_called_once()
        assert engine.pool is None
    
    @pytest.mark.asyncio
    async def test_close_without_pool(self):
        """Should not raise error if pool doesn't exist"""
        engine = StorageEngine()
        engine.pool = None
        
        # Should not raise
        await engine.close()
        assert engine.pool is None


class TestStorageEngineStats:
    """Test StorageEngine.get_stats()"""
    
    def test_get_stats_not_initialized(self):
        """Should return stats with pool_size 0 if not initialized"""
        engine = StorageEngine()
        stats = engine.get_stats()
        
        assert stats['total_written'] == 0
        assert stats['total_failed'] == 0
        assert stats['pool_size'] == 0
    
    def test_get_stats_initialized(self):
        """Should return correct stats when initialized"""
        engine = StorageEngine(pool_size=10)
        mock_pool = AsyncMock()
        engine.pool = mock_pool
        engine.total_written = 100
        engine.total_failed = 5
        
        stats = engine.get_stats()
        
        assert stats['total_written'] == 100
        assert stats['total_failed'] == 5
        assert stats['pool_size'] == 10
    
    def test_get_stats_after_operations(self):
        """Should reflect operations in stats"""
        engine = StorageEngine()
        engine.total_written = 50
        engine.total_failed = 2
        
        stats = engine.get_stats()
        
        assert stats['total_written'] == 50
        assert stats['total_failed'] == 2


class TestStorageEngineIntegration:
    """Integration tests for StorageEngine"""
    
    def create_sample_snapshot(self, index: int = 0) -> dict:
        """Create a sample orderbook snapshot"""
        base_price = 50000.0 + (index * 0.01)
        return {
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000) + index,
            'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0]],
            'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 3.0]],
            'best_bid': base_price - 0.01,
            'best_ask': base_price + 0.01,
            'spread': 0.02,
            'spread_pct': 0.00004
        }
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full lifecycle: initialize, write, get stats, close"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value="INSERT 0 1")
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            
            # Initialize
            await engine.initialize()
            assert engine.pool is not None
            
            # Write
            snapshot = self.create_sample_snapshot()
            result = await engine._write_batch([snapshot])
            assert result == 1
            
            # Get stats
            stats = engine.get_stats()
            assert stats['total_written'] == 1
            
            # Close
            await engine.close()
            assert engine.pool is None
    
    @pytest.mark.asyncio
    async def test_multiple_batch_writes(self):
        """Test writing multiple batches"""
        engine = StorageEngine(connection_string="postgresql://test")
        
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock(return_value="INSERT 0 2")
        
        async def acquire():
            return mock_conn
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        engine.pool = mock_pool
        
        # Write first batch
        snapshots1 = [self.create_sample_snapshot(i) for i in range(2)]
        result1 = await engine._write_batch(snapshots1)
        assert result1 == 2
        
        # Write second batch
        snapshots2 = [self.create_sample_snapshot(i + 2) for i in range(2)]
        result2 = await engine._write_batch(snapshots2)
        assert result2 == 2
        
        assert engine.total_written == 4
        assert mock_conn.executemany.call_count == 2

