"""
Integration test script for IngestorService
Tests the full data ingestion pipeline with real components
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.ingestor import IngestorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def create_sample_orderbook(index: int) -> dict:
    """Create a sample orderbook snapshot for testing"""
    base_price = 50000.0 + (index * 0.01)
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {
        'symbol': 'BTC/USDT',
        'timestamp': timestamp,
        'datetime': datetime.now(timezone.utc).isoformat(),
        'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0], [base_price - 0.03, 1.0]],
        'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 3.0], [base_price + 0.03, 2.0]],
        'best_bid': base_price - 0.01,
        'best_ask': base_price + 0.01,
        'spread': 0.02,
        'spread_pct': 0.00004
    }


async def test_ingestor_without_storage_cache(duration: int = 5):
    """
    Test ingestor with storage and cache disabled
    This tests the basic streaming and buffering functionality
    """
    print("=" * 70)
    print("TEST 1: IngestorService without Storage and Cache")
    print("=" * 70)
    print(f"Will run for {duration} seconds...")
    print("-" * 70)
    
    ingestor = IngestorService(
        buffer_size=10,  # Small buffer for testing
        buffer_interval=2.0,
        enable_storage=False,
        enable_cache=False
    )
    
    # Mock the streamer to yield sample data
    async def mock_stream():
        count = 0
        while count < 20:  # Yield 20 snapshots
            await asyncio.sleep(0.25)  # Simulate real-time updates
            yield create_sample_orderbook(count)
            count += 1
    
    ingestor.streamer.stream = mock_stream
    
    try:
        # Start the service in a background task
        start_task = asyncio.create_task(ingestor.start())
        
        # Wait for the specified duration
        await asyncio.sleep(duration)
        
        # Stop the service
        print("\nStopping ingestor service...")
        await ingestor.stop()
        
        # Wait for start task to complete
        await start_task
        
        # Print final stats
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = ingestor.get_stats()
        print(f"Total Received: {stats['total_received']}")
        print(f"Buffer Stats: {stats['buffer']}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        await ingestor.stop()


async def test_ingestor_with_storage(duration: int = 10):
    """
    Test ingestor with storage enabled (requires TimescaleDB)
    This tests the full pipeline including database writes
    """
    print("\n" + "=" * 70)
    print("TEST 2: IngestorService with Storage Enabled")
    print("=" * 70)
    print(f"Will run for {duration} seconds...")
    print("Note: This requires TimescaleDB to be running")
    print("-" * 70)
    
    ingestor = IngestorService(
        buffer_size=5,  # Small buffer to trigger frequent flushes
        buffer_interval=3.0,
        enable_storage=True,
        enable_cache=False
    )
    
    # Mock the streamer to yield sample data
    async def mock_stream():
        count = 0
        while count < 30:  # Yield 30 snapshots
            await asyncio.sleep(0.3)
            yield create_sample_orderbook(count)
            count += 1
    
    ingestor.streamer.stream = mock_stream
    
    try:
        # Start the service in a background task
        start_task = asyncio.create_task(ingestor.start())
        
        # Wait for the specified duration
        await asyncio.sleep(duration)
        
        # Stop the service
        print("\nStopping ingestor service...")
        await ingestor.stop()
        
        # Wait for start task to complete
        await start_task
        
        # Print final stats
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = ingestor.get_stats()
        print(f"Total Received: {stats['total_received']}")
        print(f"Buffer Stats: {stats['buffer']}")
        if 'storage' in stats:
            print(f"Storage Stats: {stats['storage']}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        await ingestor.stop()


async def test_ingestor_with_cache(duration: int = 5):
    """
    Test ingestor with cache enabled (requires Redis)
    This tests the caching functionality
    """
    print("\n" + "=" * 70)
    print("TEST 3: IngestorService with Cache Enabled")
    print("=" * 70)
    print(f"Will run for {duration} seconds...")
    print("Note: This requires Redis to be running")
    print("-" * 70)
    
    ingestor = IngestorService(
        buffer_size=10,
        buffer_interval=2.0,
        enable_storage=False,
        enable_cache=True
    )
    
    # Mock the streamer to yield sample data
    async def mock_stream():
        count = 0
        while count < 15:  # Yield 15 snapshots
            await asyncio.sleep(0.3)
            yield create_sample_orderbook(count)
            count += 1
    
    ingestor.streamer.stream = mock_stream
    
    try:
        # Start the service in a background task
        start_task = asyncio.create_task(ingestor.start())
        
        # Wait for the specified duration
        await asyncio.sleep(duration)
        
        # Stop the service
        print("\nStopping ingestor service...")
        await ingestor.stop()
        
        # Wait for start task to complete
        await start_task
        
        # Print final stats
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = ingestor.get_stats()
        print(f"Total Received: {stats['total_received']}")
        print(f"Buffer Stats: {stats['buffer']}")
        if 'cache' in stats:
            print(f"Cache Stats: {stats['cache']}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        await ingestor.stop()


async def test_ingestor_full_pipeline(duration: int = 15):
    """
    Test ingestor with all components enabled (requires TimescaleDB and Redis)
    This tests the complete data ingestion pipeline
    """
    print("\n" + "=" * 70)
    print("TEST 4: IngestorService - Full Pipeline (Storage + Cache)")
    print("=" * 70)
    print(f"Will run for {duration} seconds...")
    print("Note: This requires both TimescaleDB and Redis to be running")
    print("-" * 70)
    
    ingestor = IngestorService(
        buffer_size=5,  # Small buffer to trigger frequent flushes
        buffer_interval=3.0,
        enable_storage=True,
        enable_cache=True
    )
    
    # Mock the streamer to yield sample data
    async def mock_stream():
        count = 0
        while count < 50:  # Yield 50 snapshots
            await asyncio.sleep(0.3)
            yield create_sample_orderbook(count)
            count += 1
    
    ingestor.streamer.stream = mock_stream
    
    try:
        # Start the service in a background task
        start_task = asyncio.create_task(ingestor.start())
        
        # Monitor progress
        monitor_count = 0
        while monitor_count < duration:
            await asyncio.sleep(5)
            monitor_count += 5
            stats = ingestor.get_stats()
            print(f"\n[Progress Check] Received: {stats['total_received']}, "
                  f"Buffer: {stats['buffer']['current_size']}, "
                  f"Uptime: {stats['uptime_seconds']:.0f}s")
        
        # Stop the service
        print("\nStopping ingestor service...")
        await ingestor.stop()
        
        # Wait for start task to complete
        await start_task
        
        # Print final stats
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = ingestor.get_stats()
        print(f"Service Running: {stats['is_running']}")
        print(f"Uptime: {stats['uptime_seconds']:.2f} seconds")
        print(f"Total Received: {stats['total_received']}")
        print(f"\nBuffer Stats:")
        for key, value in stats['buffer'].items():
            print(f"  {key}: {value}")
        if 'storage' in stats:
            print(f"\nStorage Stats:")
            for key, value in stats['storage'].items():
                print(f"  {key}: {value}")
        if 'cache' in stats:
            print(f"\nCache Stats:")
            for key, value in stats['cache'].items():
                print(f"  {key}: {value}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        await ingestor.stop()


async def test_ingestor_graceful_shutdown():
    """
    Test that ingestor shuts down gracefully and flushes remaining data
    """
    print("\n" + "=" * 70)
    print("TEST 5: Graceful Shutdown Test")
    print("=" * 70)
    print("Testing that remaining data is flushed on shutdown...")
    print("-" * 70)
    
    ingestor = IngestorService(
        buffer_size=10,
        buffer_interval=5.0,
        enable_storage=False,
        enable_cache=False
    )
    
    # Add some data directly to buffer to test flush on shutdown
    print("\nAdding 5 items directly to buffer...")
    for i in range(5):
        snapshot = create_sample_orderbook(i)
        await ingestor.buffer.add(snapshot)
        print(f"  Added item {i+1}")
    
    print(f"\nBuffer size before shutdown: {ingestor.buffer.size()}")
    
    # Mock streamer that stops quickly
    async def mock_stream():
        yield create_sample_orderbook(0)
        await asyncio.sleep(0.1)
        ingestor.shutdown_event.set()
    
    ingestor.streamer.stream = mock_stream
    
    try:
        # Start and immediately trigger shutdown
        start_task = asyncio.create_task(ingestor.start())
        await asyncio.sleep(0.5)  # Give it a moment
        await ingestor.stop()
        await start_task
        
        print(f"\nBuffer size after shutdown: {ingestor.buffer.size()}")
        print(f"Total processed: {ingestor.buffer.get_stats()['total_processed']}")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        await ingestor.stop()


async def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("INGESTOR SERVICE INTEGRATION TESTS")
    print("=" * 70)
    print("\nThis script tests the IngestorService with various configurations.")
    print("Some tests require TimescaleDB and/or Redis to be running.")
    print("\nPress Ctrl+C at any time to stop the tests.")
    print("=" * 70)
    
    try:
        # Test 1: Basic functionality without storage/cache
        await test_ingestor_without_storage_cache(duration=5)
        
        # Test 2: With storage (comment out if DB not available)
        # await test_ingestor_with_storage(duration=10)
        
        # Test 3: With cache (comment out if Redis not available)
        # await test_ingestor_with_cache(duration=5)
        
        # Test 4: Full pipeline (comment out if DB/Redis not available)
        # await test_ingestor_full_pipeline(duration=15)
        
        # Test 5: Graceful shutdown
        await test_ingestor_graceful_shutdown()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

