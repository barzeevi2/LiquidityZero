"""
test script for data buffer
"""

import asyncio
import sys
import logging

from pathlib import Path

from datetime import datetime, timezone

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.buffer import DataBuffer
from app.data.exceptions import BufferOverflowError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_sample_orderbook(index: int) -> dict:
    """Create a sample orderbook snapshot for testing"""
    base_price = 50000.0 + (index * 0.01)
    return {
        'symbol': 'BTC/USDT',
        'timestamp': datetime.now(timezone.utc).timestamp() * 1000,
        'datetime': datetime.now(timezone.utc).isoformat(),
        'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0]],
        'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 3.0]],
        'best_bid': base_price - 0.01,
        'best_ask': base_price + 0.01,
        'spread': 0.02,
        'spread_pct': 0.00004
    }

async def test_basic_operations():
    """Test basic add flush and batch operations"""
    print("-"*50)
    print("testing basic operations")
    print("-"*50)

    buffer = DataBuffer(max_size=100, flush_size = 5, flush_interval = 2.0)

    #add items
    print(f"\nAdding 10 items to buffer")
    for i in range(10):
        item = create_sample_orderbook(i)
        result = await buffer.add(item)
        if result:
            print(f"Added item {i+1} to buffer")
    
    print(f"\nBuffer stats: {buffer.get_stats()}")

    #get a batch
    print(f"\nGetting a batch (flush_size={buffer.flush_size})...")
    batch = await buffer.get_batch(block=False)
    print(f"  Retrieved {len(batch)} items")
    print(f"  Buffer size after batch: {buffer.size()}")

    #flush remaining
    print(f"\nFlushing remaining items...")
    flushed = await buffer.flush()
    print(f"  Flushed {len(flushed)} items")
    print(f"  Buffer size after flush: {buffer.size()}")
    print(f"  Total processed: {buffer.get_stats()['total_processed']}")


async def test_size_based_flushing():
    """Test size based flushing"""
    print("-"*50)
    print("testing size based flushing")
    print("-"*50)

    buffer = DataBuffer(max_size=100, flush_size = 5, flush_interval = 2.0)

    #add items
    print(f"\nAdding 10 items to buffer")
    for i in range(10):
        item = create_sample_orderbook(i)
        await buffer.add(item)
        size = buffer.size()
        should_flush = buffer.should_flush_by_size()
        print(f"  Item {i+1}: size={size}, should_flush_by_size={should_flush}")

        # Get batch to simulate auto-flush
    print(f"\nGetting batch (should get {buffer.flush_size} items)...")
    batch = await buffer.get_batch(block=False)
    print(f"  Retrieved {len(batch)} items")
    print(f"  Remaining in buffer: {buffer.size()}")


async def test_time_based_flushing():
    """Test time-based flushing conditions"""
    print("\n" + "=" * 70)
    print("TEST 3: Time-Based Flushing")
    print("=" * 70)
    
    buffer = DataBuffer(max_size=100, flush_size=10, flush_interval=1.0)
    
    print(f"\nAdding 3 items...")
    for i in range(3):
        await buffer.add(create_sample_orderbook(i))
    
    print(f"\nInitial state:")
    print(f"  Buffer size: {buffer.size()}")
    print(f"  Last flush time: {buffer.last_flush_time}")
    print(f"  should_flush_by_time: {buffer.should_flush_by_time()}")
    
    # Flush to set last_flush_time
    await buffer.flush()
    print(f"\nAfter flush:")
    print(f"  Last flush time: {buffer.last_flush_time}")
    print(f"  should_flush_by_time (immediately): {buffer.should_flush_by_time()}")
    
    # Wait for interval to pass
    print(f"\nWaiting {buffer.flush_interval} seconds for interval to pass...")
    await asyncio.sleep(buffer.flush_interval + 0.1)
    print(f"  should_flush_by_time (after wait): {buffer.should_flush_by_time()}")

async def test_concurrent_operations():
    """Test concurrent adding operations"""
    print("\n" + "=" * 70)
    print("TEST 4: Concurrent Operations")
    print("=" * 70)
    
    buffer = DataBuffer(max_size=100, flush_size=5, flush_interval=2.0)
    
    async def add_items(start_index: int, count: int):
        """Helper to add items concurrently"""
        for i in range(count):
            item = create_sample_orderbook(start_index + i)
            await buffer.add(item)
    
    print(f"\nAdding 15 items concurrently (3 tasks × 5 items each)...")
    await asyncio.gather(
        add_items(0, 5),
        add_items(5, 5),
        add_items(10, 5)
    )
    
    print(f"  Buffer size: {buffer.size()}")
    print(f"  Total processed: {buffer.get_stats()['total_processed']}")
    
    # Get batches
    batches_retrieved = 0
    while buffer.size() > 0:
        batch = await buffer.get_batch(block=False)
        if batch:
            batches_retrieved += 1
            print(f"  Batch {batches_retrieved}: {len(batch)} items")
    
    print(f"\n  Retrieved {batches_retrieved} batches")
    print(f"  Final buffer size: {buffer.size()}")

async def test_overflow_protection():
    """Test buffer overflow protection"""
    print("\n" + "=" * 70)
    print("TEST 5: Overflow Protection")
    print("=" * 70)
    
    buffer = DataBuffer(max_size=5, flush_size=2, flush_interval=2.0)
    
    print(f"\nFilling buffer to max_size ({buffer.max_size})...")
    for i in range(5):
        await buffer.add(create_sample_orderbook(i))
        print(f"  Added item {i+1}, buffer size: {buffer.size()}")
    
    print(f"\nAttempting to add item when buffer is full...")
    try:
        await buffer.add(create_sample_orderbook(6))
        print("  ERROR: Should have raised BufferOverflowError!")
    except BufferOverflowError as e:
        print(f"  ✓ Correctly raised BufferOverflowError: {e}")
    
    print(f"\nBuffer stats:")
    stats = buffer.get_stats()
    print(f"  Current size: {stats['current_size']}")
    print(f"  Total dropped: {stats['total_dropped']}")


async def test_get_batch_blocking():
    """Test blocking batch retrieval"""
    print("\n" + "=" * 70)
    print("TEST 6: Blocking Batch Retrieval")
    print("=" * 70)
    
    buffer = DataBuffer(max_size=100, flush_size=3, flush_interval=1.0)
    
    async def delayed_add():
        """Add items after a delay"""
        await asyncio.sleep(0.5)
        print("  Adding 2 items after delay...")
        await buffer.add(create_sample_orderbook(1))
        await buffer.add(create_sample_orderbook(2))
    
    print(f"\nStarting blocking get_batch (will wait up to {buffer.flush_interval}s)...")
    print("  (Items will be added after 0.5s)")
    
    # Start adding items in background
    add_task = asyncio.create_task(delayed_add())
    
    # Try to get batch (should wait for items)
    start_time = asyncio.get_event_loop().time()
    batch = await buffer.get_batch(block=True)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    await add_task  # Ensure task completes
    
    print(f"  Retrieved {len(batch)} items after {elapsed:.2f}s")
    print(f"  Buffer size: {buffer.size()}")


async def run_all_tests():
    """Run all buffer tests"""
    print("\n" + "=" * 70)
    print("DataBuffer Test Script")
    print("=" * 70)
    
    try:
        await test_basic_operations()
        await test_size_based_flushing()
        await test_time_based_flushing()
        await test_concurrent_operations()
        await test_overflow_protection()
        await test_get_batch_blocking()
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())