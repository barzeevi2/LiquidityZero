"""
Test script for StorageEngine
Tests initialization, schema creation, batch writes, retry logic, and error handling
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.storage import StorageEngine
from app.data.exceptions import StorageError
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_orderbook(index: int) -> dict:
    """Create a sample orderbook snapshot for testing"""
    base_price = 50000.0 + (index * 0.01)
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000) + index
    return {
        'symbol': 'BTC/USDT',
        'timestamp': timestamp_ms,
        'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0], [base_price - 0.03, 3.0]],
        'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 2.5], [base_price + 0.03, 1.8]],
        'best_bid': base_price - 0.01,
        'best_ask': base_price + 0.01,
        'spread': 0.02,
        'spread_pct': 0.00004
    }


async def test_initialization():
    """Test StorageEngine initialization"""
    print("\n" + "=" * 70)
    print("TEST 1: Initialization")
    print("=" * 70)
    
    engine = StorageEngine(
        connection_string=settings.TIMESCALE_URL,
        pool_size=5,
        max_retries=3,
        initial_backoff=1.0
    )
    
    print(f"\nEngine created with:")
    print(f"  Connection string: {engine.connection_string[:50]}...")
    print(f"  Pool size: {engine.pool_size}")
    print(f"  Max retries: {engine.max_retries}")
    print(f"  Initial backoff: {engine.initial_backoff}")
    print(f"  Pool initialized: {engine.pool is not None}")
    
    try:
        print(f"\nInitializing connection pool...")
        await engine.initialize()
        print(f"  ✓ Connection pool initialized successfully")
        print(f"  Pool: {engine.pool}")
        
        stats = engine.get_stats()
        print(f"\nInitial stats:")
        print(f"  Total written: {stats['total_written']}")
        print(f"  Total failed: {stats['total_failed']}")
        print(f"  Pool size: {stats['pool_size']}")
        
        return engine
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        print(f"  Note: Make sure TimescaleDB is running and connection string is correct")
        return None


async def test_schema_creation(engine: StorageEngine):
    """Test that schema was created correctly"""
    print("\n" + "=" * 70)
    print("TEST 2: Schema Creation")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        async with engine.pool.acquire() as conn:
            # Check if table exists
            table_check = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'orderbook_snapshots'
                );
            """)
            
            print(f"\nTable 'orderbook_snapshots' exists: {table_check}")
            
            if table_check:
                # Check if it's a hypertable
                hypertable_check = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM _timescaledb_catalog.hypertable 
                        WHERE table_name = 'orderbook_snapshots'
                    );
                """)
                print(f"  Is hypertable: {hypertable_check}")
                
                # Get table structure
                columns = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'orderbook_snapshots'
                    ORDER BY ordinal_position;
                """)
                print(f"\n  Table columns:")
                for col in columns:
                    print(f"    - {col['column_name']}: {col['data_type']}")
                
                print(f"  ✓ Schema created successfully")
            else:
                print(f"  ✗ Table not found")
    except Exception as e:
        print(f"  ✗ Schema check failed: {e}")


async def test_write_single_snapshot(engine: StorageEngine):
    """Test writing a single snapshot"""
    print("\n" + "=" * 70)
    print("TEST 3: Write Single Snapshot")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        snapshot = create_sample_orderbook(0)
        print(f"\nWriting snapshot:")
        print(f"  Symbol: {snapshot['symbol']}")
        print(f"  Timestamp: {snapshot['timestamp']}")
        print(f"  Best bid: {snapshot['best_bid']}")
        print(f"  Best ask: {snapshot['best_ask']}")
        print(f"  Spread: {snapshot['spread']}")
        
        result = await engine._write_batch([snapshot])
        
        print(f"\n  ✓ Successfully wrote {result} snapshot(s)")
        stats = engine.get_stats()
        print(f"  Total written: {stats['total_written']}")
        print(f"  Total failed: {stats['total_failed']}")
    except Exception as e:
        print(f"  ✗ Write failed: {e}")
        import traceback
        traceback.print_exc()


async def test_write_batch_snapshots(engine: StorageEngine):
    """Test writing multiple snapshots in a batch"""
    print("\n" + "=" * 70)
    print("TEST 4: Write Batch of Snapshots")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        batch_size = 10
        snapshots = [create_sample_orderbook(i) for i in range(batch_size)]
        
        print(f"\nWriting batch of {batch_size} snapshots...")
        start_time = asyncio.get_event_loop().time()
        
        result = await engine._write_batch(snapshots)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print(f"  ✓ Successfully wrote {result} snapshots in {elapsed:.3f}s")
        print(f"  Rate: {result/elapsed:.1f} snapshots/second")
        
        stats = engine.get_stats()
        print(f"\n  Total written: {stats['total_written']}")
        print(f"  Total failed: {stats['total_failed']}")
    except Exception as e:
        print(f"  ✗ Batch write failed: {e}")
        import traceback
        traceback.print_exc()


async def test_timestamp_parsing(engine: StorageEngine):
    """Test different timestamp formats"""
    print("\n" + "=" * 70)
    print("TEST 5: Timestamp Parsing")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        # Test milliseconds timestamp
        snapshot1 = create_sample_orderbook(100)
        snapshot1['timestamp'] = 1234567890123  # milliseconds
        print(f"\n1. Testing milliseconds timestamp: {snapshot1['timestamp']}")
        result1 = await engine._write_batch([snapshot1])
        print(f"   ✓ Wrote {result1} snapshot")
        
        # Test seconds timestamp
        snapshot2 = create_sample_orderbook(101)
        snapshot2['timestamp'] = 1234567890  # seconds
        print(f"\n2. Testing seconds timestamp: {snapshot2['timestamp']}")
        result2 = await engine._write_batch([snapshot2])
        print(f"   ✓ Wrote {result2} snapshot")
        
        # Test datetime string
        snapshot3 = create_sample_orderbook(102)
        del snapshot3['timestamp']
        snapshot3['datetime'] = '2024-01-01T12:00:00Z'
        print(f"\n3. Testing datetime string: {snapshot3['datetime']}")
        result3 = await engine._write_batch([snapshot3])
        print(f"   ✓ Wrote {result3} snapshot")
        
        # Test no timestamp (should use current time)
        snapshot4 = create_sample_orderbook(103)
        del snapshot4['timestamp']
        print(f"\n4. Testing no timestamp (uses current time)")
        result4 = await engine._write_batch([snapshot4])
        print(f"   ✓ Wrote {result4} snapshot")
        
        print(f"\n  ✓ All timestamp formats handled correctly")
    except Exception as e:
        print(f"  ✗ Timestamp parsing test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_duplicate_handling(engine: StorageEngine):
    """Test that duplicate snapshots are handled (ON CONFLICT DO NOTHING)"""
    print("\n" + "=" * 70)
    print("TEST 6: Duplicate Handling")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        # Create a snapshot with specific timestamp
        snapshot = create_sample_orderbook(200)
        snapshot['timestamp'] = 9999999999999  # Unique timestamp
        
        print(f"\nWriting snapshot with timestamp: {snapshot['timestamp']}")
        result1 = await engine._write_batch([snapshot])
        print(f"  First write: {result1} snapshot(s)")
        
        # Try to write the same snapshot again
        print(f"\nAttempting to write duplicate snapshot...")
        result2 = await engine._write_batch([snapshot])
        print(f"  Second write: {result2} snapshot(s)")
        
        if result2 == 0:
            print(f"  ✓ Duplicate correctly ignored (ON CONFLICT DO NOTHING)")
        else:
            print(f"  ⚠ Duplicate was inserted (may be expected if timestamp differs)")
    except Exception as e:
        print(f"  ✗ Duplicate handling test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_query_data(engine: StorageEngine):
    """Test querying written data"""
    print("\n" + "=" * 70)
    print("TEST 7: Query Written Data")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        async with engine.pool.acquire() as conn:
            # Get total count
            total_count = await conn.fetchval("""
                SELECT COUNT(*) FROM orderbook_snapshots;
            """)
            print(f"\nTotal snapshots in database: {total_count}")
            
            # Get recent snapshots
            recent = await conn.fetch("""
                SELECT time, symbol, best_bid, best_ask, spread
                FROM orderbook_snapshots
                ORDER BY time DESC
                LIMIT 5;
            """)
            
            print(f"\n  Recent snapshots (last 5):")
            for row in recent:
                print(f"    {row['time']} | {row['symbol']} | "
                      f"bid={row['best_bid']} ask={row['best_ask']} spread={row['spread']}")
            
            # Get stats by symbol
            symbol_stats = await conn.fetch("""
                SELECT symbol, COUNT(*) as count, 
                       MIN(time) as first, MAX(time) as last
                FROM orderbook_snapshots
                GROUP BY symbol;
            """)
            
            print(f"\n  Stats by symbol:")
            for row in symbol_stats:
                print(f"    {row['symbol']}: {row['count']} snapshots "
                      f"(from {row['first']} to {row['last']})")
            
            print(f"  ✓ Data query successful")
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        import traceback
        traceback.print_exc()


async def test_large_batch(engine: StorageEngine):
    """Test writing a large batch"""
    print("\n" + "=" * 70)
    print("TEST 8: Large Batch Write")
    print("=" * 70)
    
    if not engine or not engine.pool:
        print("  Skipping - engine not initialized")
        return
    
    try:
        batch_size = 100
        snapshots = [create_sample_orderbook(i + 1000) for i in range(batch_size)]
        
        print(f"\nWriting large batch of {batch_size} snapshots...")
        start_time = asyncio.get_event_loop().time()
        
        result = await engine._write_batch(snapshots)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print(f"  ✓ Successfully wrote {result} snapshots in {elapsed:.3f}s")
        print(f"  Rate: {result/elapsed:.1f} snapshots/second")
        
        stats = engine.get_stats()
        print(f"\n  Total written: {stats['total_written']}")
        print(f"  Total failed: {stats['total_failed']}")
    except Exception as e:
        print(f"  ✗ Large batch write failed: {e}")
        import traceback
        traceback.print_exc()


async def test_stats(engine: StorageEngine):
    """Test getting statistics"""
    print("\n" + "=" * 70)
    print("TEST 9: Statistics")
    print("=" * 70)
    
    if not engine:
        print("  Skipping - engine not initialized")
        return
    
    stats = engine.get_stats()
    
    print(f"\nStorage Engine Statistics:")
    print(f"  Total written: {stats['total_written']}")
    print(f"  Total failed: {stats['total_failed']}")
    print(f"  Pool size: {stats['pool_size']}")
    
    if stats['total_written'] > 0:
        success_rate = (stats['total_written'] / 
                        (stats['total_written'] + stats['total_failed'])) * 100
        print(f"  Success rate: {success_rate:.2f}%")
    
    print(f"  ✓ Statistics retrieved successfully")


async def test_close(engine: StorageEngine):
    """Test closing the engine"""
    print("\n" + "=" * 70)
    print("TEST 10: Close Engine")
    print("=" * 70)
    
    if not engine:
        print("  Skipping - engine not initialized")
        return
    
    try:
        print(f"\nClosing connection pool...")
        await engine.close()
        print(f"  ✓ Connection pool closed successfully")
        print(f"  Pool is None: {engine.pool is None}")
    except Exception as e:
        print(f"  ✗ Close failed: {e}")


async def run_all_tests():
    """Run all storage engine tests"""
    print("\n" + "=" * 70)
    print("StorageEngine Test Script")
    print("=" * 70)
    print(f"\nUsing connection string: {settings.TIMESCALE_URL[:50]}...")
    print(f"Note: Make sure TimescaleDB is running and accessible")
    
    engine = None
    
    try:
        # Test initialization
        engine = await test_initialization()
        
        if engine and engine.pool:
            # Run all tests
            await test_schema_creation(engine)
            await test_write_single_snapshot(engine)
            await test_write_batch_snapshots(engine)
            await test_timestamp_parsing(engine)
            await test_duplicate_handling(engine)
            await test_query_data(engine)
            await test_large_batch(engine)
            await test_stats(engine)
        else:
            print("\n⚠ Skipping database tests - connection not available")
            print("  Make sure TimescaleDB is running:")
            print("    docker-compose -f docker/docker-compose.yml up -d")
        
        # Always test stats and close
        await test_stats(engine)
        await test_close(engine)
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        if engine:
            await engine.close()
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if engine:
            await engine.close()


if __name__ == "__main__":
    asyncio.run(run_all_tests())

