"""
Unit tests for IngestorService
Tests initialization, start/stop, task management, error handling, and statistics
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

from app.data.ingestor import IngestorService
from app.data.exceptions import IngestorException


class TestIngestorServiceInitialization:
    """Test IngestorService initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default values"""
        ingestor = IngestorService()
        
        assert ingestor.is_running is False
        assert ingestor.start_time is None
        assert ingestor.total_received == 0
        assert ingestor.streamer is not None
        assert ingestor.buffer is not None
        assert ingestor.storage is not None
        assert ingestor.cache is not None
        assert len(ingestor.tasks) == 0
    
    def test_custom_initialization(self):
        """Should accept custom parameters"""
        ingestor = IngestorService(
            buffer_size=500,
            buffer_interval=2.0,
            enable_storage=False,
            enable_cache=False
        )
        
        assert ingestor.buffer.flush_size == 500
        assert ingestor.buffer.flush_interval == 2.0
        assert ingestor.storage is None
        assert ingestor.cache is None
    
    def test_initialization_without_storage(self):
        """Should work without storage enabled"""
        ingestor = IngestorService(enable_storage=False)
        assert ingestor.storage is None
        assert ingestor.cache is not None
    
    def test_initialization_without_cache(self):
        """Should work without cache enabled"""
        ingestor = IngestorService(enable_cache=False)
        assert ingestor.storage is not None
        assert ingestor.cache is None


class TestIngestorServiceStart:
    """Test starting the ingestor service"""
    
    @pytest.mark.asyncio
    async def test_start_initializes_components(self):
        """Should initialize storage and cache when starting"""
        ingestor = IngestorService()
        ingestor.storage = AsyncMock()
        ingestor.cache = AsyncMock()
        
        # Mock the shutdown_event.wait to return immediately
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.wait = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=True)
        
        # Mock create_task to avoid actually running tasks
        with patch('asyncio.create_task') as mock_create_task:
            await ingestor.start()
        
        ingestor.storage.initialize.assert_called_once()
        ingestor.cache.initialize.assert_called_once()
        assert ingestor.is_running is True
        assert ingestor.start_time is not None
    
    @pytest.mark.asyncio
    async def test_start_creates_tasks(self):
        """Should create three background tasks"""
        ingestor = IngestorService()
        ingestor.storage = AsyncMock()
        ingestor.cache = AsyncMock()
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.wait = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=True)
        
        mock_tasks = []
        with patch('asyncio.create_task') as mock_create_task:
            def create_task_side_effect(coro, name=None):
                task = AsyncMock()
                task.done = MagicMock(return_value=True)
                mock_tasks.append(task)
                return task
            
            mock_create_task.side_effect = create_task_side_effect
            await ingestor.start()
        
        assert mock_create_task.call_count == 3
        assert len(ingestor.tasks) == 3
    
    @pytest.mark.asyncio
    async def test_start_without_storage(self):
        """Should start successfully without storage"""
        ingestor = IngestorService(enable_storage=False)
        ingestor.cache = AsyncMock()
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.wait = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=True)
        
        with patch('asyncio.create_task'):
            await ingestor.start()
        
        assert ingestor.is_running is True
        assert ingestor.storage is None
    
    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Should not start if already running"""
        ingestor = IngestorService()
        ingestor.is_running = True
        
        await ingestor.start()
        
        # Should return early without initializing
        assert ingestor.is_running is True
    
    @pytest.mark.asyncio
    async def test_start_handles_initialization_error(self):
        """Should raise IngestorException if initialization fails"""
        ingestor = IngestorService()
        ingestor.storage = AsyncMock()
        ingestor.storage.initialize = AsyncMock(side_effect=Exception("DB connection failed"))
        ingestor.cache = AsyncMock()
        
        with pytest.raises(IngestorException) as exc_info:
            await ingestor.start()
        
        assert "Error starting ingestor service" in str(exc_info.value)


class TestIngestorServiceStreamConsumer:
    """Test the stream consumer task"""
    
    @pytest.mark.asyncio
    async def test_stream_consumer_processes_snapshots(self):
        """Should process snapshots from streamer and add to buffer"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=False)
        
        # Mock streamer to yield snapshots
        mock_snapshots = [
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
            {'symbol': 'BTC/USDT', 'timestamp': 1234567891},
        ]
        
        async def mock_stream():
            for snapshot in mock_snapshots:
                yield snapshot
                if not ingestor.is_running:
                    break
        
        ingestor.streamer.stream = mock_stream
        ingestor.buffer.add = AsyncMock(return_value=True)
        ingestor.cache = AsyncMock()
        ingestor.cache.update_latest = AsyncMock()
        
        # Run the consumer
        await ingestor._stream_consumer()
        
        assert ingestor.buffer.add.call_count == 2
        assert ingestor.total_received == 2
        assert ingestor.cache.update_latest.call_count == 2
    
    @pytest.mark.asyncio
    async def test_stream_consumer_stops_on_shutdown(self):
        """Should stop when shutdown event is set"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        
        call_count = 0
        async def mock_stream():
            nonlocal call_count
            while True:
                call_count += 1
                if call_count > 5:
                    ingestor.shutdown_event.is_set = MagicMock(return_value=True)
                yield {'symbol': 'BTC/USDT', 'timestamp': 1234567890}
        
        ingestor.streamer.stream = mock_stream
        ingestor.buffer.add = AsyncMock(return_value=True)
        ingestor.cache = None
        
        await ingestor._stream_consumer()
        
        assert ingestor.buffer.add.call_count <= 6
    
    @pytest.mark.asyncio
    async def test_stream_consumer_without_cache(self):
        """Should work without cache enabled"""
        ingestor = IngestorService(enable_cache=False)
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=False)
        
        async def mock_stream():
            yield {'symbol': 'BTC/USDT', 'timestamp': 1234567890}
            ingestor.shutdown_event.is_set = MagicMock(return_value=True)
            yield {'symbol': 'BTC/USDT', 'timestamp': 1234567891}
        
        ingestor.streamer.stream = mock_stream
        ingestor.buffer.add = AsyncMock(return_value=True)
        
        await ingestor._stream_consumer()
        
        assert ingestor.buffer.add.call_count >= 1
        assert ingestor.cache is None


class TestIngestorServiceStorageWriter:
    """Test the storage writer task"""
    
    @pytest.mark.asyncio
    async def test_storage_writer_flushes_on_size(self):
        """Should flush buffer when size threshold is reached"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=False)
        
        ingestor.buffer.should_flush_by_size = MagicMock(return_value=True)
        ingestor.buffer.should_flush_by_time = MagicMock(return_value=False)
        ingestor.buffer.get_batch = AsyncMock(return_value=[
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
            {'symbol': 'BTC/USDT', 'timestamp': 1234567891},
        ])
        
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock()
        
        # Run for a short time then stop
        async def run_and_stop():
            await asyncio.sleep(0.2)
            ingestor.is_running = False
        
        await asyncio.gather(
            ingestor._storage_writer(),
            run_and_stop()
        )
        
        assert ingestor.storage.write_batch.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_storage_writer_flushes_on_time(self):
        """Should flush buffer when time threshold is reached"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=False)
        
        ingestor.buffer.should_flush_by_size = MagicMock(return_value=False)
        ingestor.buffer.should_flush_by_time = MagicMock(return_value=True)
        ingestor.buffer.get_batch = AsyncMock(return_value=[
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
        ])
        
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock()
        
        async def run_and_stop():
            await asyncio.sleep(0.2)
            ingestor.is_running = False
        
        await asyncio.gather(
            ingestor._storage_writer(),
            run_and_stop()
        )
        
        assert ingestor.storage.write_batch.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_storage_writer_without_storage(self):
        """Should return early if storage is not enabled"""
        ingestor = IngestorService(enable_storage=False)
        ingestor.is_running = True
        
        # Should return immediately
        await ingestor._storage_writer()
        
        assert ingestor.storage is None
    
    @pytest.mark.asyncio
    async def test_storage_writer_handles_write_errors(self):
        """Should handle errors when writing to storage"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.shutdown_event = AsyncMock()
        ingestor.shutdown_event.is_set = MagicMock(return_value=False)
        
        ingestor.buffer.should_flush_by_size = MagicMock(return_value=True)
        ingestor.buffer.should_flush_by_time = MagicMock(return_value=False)
        ingestor.buffer.get_batch = AsyncMock(return_value=[
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
        ])
        
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock(side_effect=Exception("DB error"))
        
        async def run_and_stop():
            await asyncio.sleep(0.2)
            ingestor.is_running = False
        
        # Should not raise, just log error
        await asyncio.gather(
            ingestor._storage_writer(),
            run_and_stop()
        )
        
        assert ingestor.storage.write_batch.call_count >= 1


class TestIngestorServiceStop:
    """Test stopping the ingestor service"""
    
    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Should return early if not running"""
        ingestor = IngestorService()
        ingestor.is_running = False
        
        await ingestor.stop()
        
        # Should complete without error
        assert ingestor.is_running is False
    
    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Should cancel all running tasks"""
        ingestor = IngestorService()
        ingestor.is_running = True
        
        # Create mock tasks
        mock_task1 = AsyncMock()
        mock_task1.done = MagicMock(return_value=False)
        mock_task2 = AsyncMock()
        mock_task2.done = MagicMock(return_value=False)
        mock_task3 = AsyncMock()
        mock_task3.done = MagicMock(return_value=False)
        
        ingestor.tasks = [mock_task1, mock_task2, mock_task3]
        ingestor.streamer.stop = AsyncMock()
        ingestor.buffer.flush = AsyncMock(return_value=[])
        ingestor.storage = AsyncMock()
        ingestor.storage.close = AsyncMock()
        ingestor.cache = AsyncMock()
        ingestor.cache.close = AsyncMock()
        
        with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
            await ingestor.stop()
        
        assert ingestor.is_running is False
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()
        mock_task3.cancel.assert_called_once()
        ingestor.streamer.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_flushes_remaining_data(self):
        """Should flush remaining buffer data before stopping"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.tasks = []
        ingestor.streamer.stop = AsyncMock()
        
        remaining_data = [
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
            {'symbol': 'BTC/USDT', 'timestamp': 1234567891},
        ]
        ingestor.buffer.flush = AsyncMock(return_value=remaining_data)
        
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock()
        ingestor.storage.close = AsyncMock()
        ingestor.cache = AsyncMock()
        ingestor.cache.close = AsyncMock()
        
        await ingestor.stop()
        
        ingestor.buffer.flush.assert_called_once()
        ingestor.storage.write_batch.assert_called_once_with(remaining_data)
    
    @pytest.mark.asyncio
    async def test_stop_closes_connections(self):
        """Should close storage and cache connections"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.tasks = []
        ingestor.streamer.stop = AsyncMock()
        ingestor.buffer.flush = AsyncMock(return_value=[])
        
        ingestor.storage = AsyncMock()
        ingestor.storage.close = AsyncMock()
        ingestor.cache = AsyncMock()
        ingestor.cache.close = AsyncMock()
        
        await ingestor.stop()
        
        ingestor.storage.close.assert_called_once()
        ingestor.cache.close.assert_called_once()


class TestIngestorServiceFlushRemaining:
    """Test flushing remaining data"""
    
    @pytest.mark.asyncio
    async def test_flush_remaining_with_data(self):
        """Should flush remaining data to storage"""
        ingestor = IngestorService()
        remaining_data = [
            {'symbol': 'BTC/USDT', 'timestamp': 1234567890},
        ]
        ingestor.buffer.flush = AsyncMock(return_value=remaining_data)
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock()
        
        await ingestor._flush_remaining()
        
        ingestor.buffer.flush.assert_called_once()
        ingestor.storage.write_batch.assert_called_once_with(remaining_data)
    
    @pytest.mark.asyncio
    async def test_flush_remaining_without_storage(self):
        """Should not write if storage is disabled"""
        ingestor = IngestorService(enable_storage=False)
        remaining_data = [{'symbol': 'BTC/USDT', 'timestamp': 1234567890}]
        ingestor.buffer.flush = AsyncMock(return_value=remaining_data)
        
        await ingestor._flush_remaining()
        
        ingestor.buffer.flush.assert_called_once()
        assert ingestor.storage is None
    
    @pytest.mark.asyncio
    async def test_flush_remaining_handles_errors(self):
        """Should handle errors when flushing remaining data"""
        ingestor = IngestorService()
        remaining_data = [{'symbol': 'BTC/USDT', 'timestamp': 1234567890}]
        ingestor.buffer.flush = AsyncMock(return_value=remaining_data)
        ingestor.storage = AsyncMock()
        ingestor.storage.write_batch = AsyncMock(side_effect=Exception("DB error"))
        
        # Should not raise, just log error
        await ingestor._flush_remaining()
        
        ingestor.storage.write_batch.assert_called_once()


class TestIngestorServiceGetStats:
    """Test getting service statistics"""
    
    def test_get_stats_when_not_started(self):
        """Should return stats with zero uptime when not started"""
        ingestor = IngestorService()
        ingestor.start_time = None
        
        stats = ingestor.get_stats()
        
        assert stats['is_running'] is False
        assert stats['uptime_seconds'] == 0
        assert stats['total_received'] == 0
        assert 'buffer' in stats
    
    def test_get_stats_includes_all_components(self):
        """Should include stats from all components"""
        ingestor = IngestorService()
        ingestor.is_running = True
        ingestor.start_time = datetime.now(timezone.utc)
        ingestor.total_received = 100
        
        ingestor.buffer.get_stats = MagicMock(return_value={'current_size': 10})
        ingestor.storage.get_stats = MagicMock(return_value={'total_written': 50})
        ingestor.cache.get_stats = MagicMock(return_value={'total_updates': 30})
        
        stats = ingestor.get_stats()
        
        assert stats['is_running'] is True
        assert stats['total_received'] == 100
        assert 'buffer' in stats
        assert 'storage' in stats
        assert 'cache' in stats
    
    def test_get_stats_without_storage(self):
        """Should work without storage stats"""
        ingestor = IngestorService(enable_storage=False)
        ingestor.buffer.get_stats = MagicMock(return_value={'current_size': 10})
        ingestor.cache.get_stats = MagicMock(return_value={'total_updates': 30})
        
        stats = ingestor.get_stats()
        
        assert 'buffer' in stats
        assert 'storage' not in stats
        assert 'cache' in stats
    
    def test_get_stats_without_cache(self):
        """Should work without cache stats"""
        ingestor = IngestorService(enable_cache=False)
        ingestor.buffer.get_stats = MagicMock(return_value={'current_size': 10})
        ingestor.storage.get_stats = MagicMock(return_value={'total_written': 50})
        
        stats = ingestor.get_stats()
        
        assert 'buffer' in stats
        assert 'storage' in stats
        assert 'cache' not in stats

