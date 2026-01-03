"""
Integration test script for RedisCache
Tests real Redis connection, caching operations, TTL expiration, and error handling
Requires Redis server to be running (via docker-compose or standalone)
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.cache import RedisCache
from app.data.exceptions import CacheError
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_orderbook(symbol: str, index: int = 0) -> dict:
    """Create a sample orderbook snapshot for testing"""
    base_price = 50000.0 + (index * 0.01)
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000) + index
    return {
        'symbol': symbol,
        'timestamp': timestamp_ms,
        'bids': [[base_price - 0.01, 1.5], [base_price - 0.02, 2.0], [base_price - 0.03, 3.0]],
        'asks': [[base_price + 0.01, 1.2], [base_price + 0.02, 2.5], [base_price + 0.03, 1.8]],
        'best_bid': base_price - 0.01,
        'best_ask': base_price + 0.01,
        'spread': 0.02,
        'spread_pct': 0.00004
    }


async def test_initialization():
    """Test RedisCache initialization"""
    print("\n" + "=" * 70)
    print("TEST 1: Initialization")
    print("=" * 70)
    
    cache = RedisCache(
        redis_url=settings.REDIS_URL,
        ttl=60,
        max_retries=3
    )
    
    print(f"\nCache created with:")
    print(f"  Redis URL: {cache.redis_url}")
    print(f"  TTL: {cache.ttl} seconds")
    print(f"  Max retries: {cache.max_retries}")
    print(f"  Client initialized: {cache.client is not None}")
    
    try:
        print(f"\nInitializing Redis connection...")
        await cache.initialize()
        print(f"  ✓ Redis connection initialized successfully")
        print(f"  Client: {cache.client}")
        
        stats = cache.get_stats()
        print(f"\nInitial stats:")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Total failed: {stats['total_failed']}")
        print(f"  TTL: {stats['ttl']} seconds")
        
        return cache
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        print(f"  Note: Make sure Redis is running at {cache.redis_url}")
        print(f"  You can start it with: docker-compose up -d redis")
        return None


async def test_update_latest(cache: RedisCache):
    """Test updating cache with orderbook snapshots"""
    print("\n" + "=" * 70)
    print("TEST 2: Update Latest (SETEX)")
    print("=" * 70)
    
    if not cache or not cache.client:
        print("  Skipping - cache not initialized")
        return
    
    try:
        # Test single update
        snapshot1 = create_sample_orderbook('BTC/USDT', index=0)
        print(f"\nUpdating cache with snapshot:")
        print(f"  Symbol: {snapshot1['symbol']}")
        print(f"  Timestamp: {snapshot1['timestamp']}")
        print(f"  Best bid: {snapshot1['best_bid']}")
        print(f"  Best ask: {snapshot1['best_ask']}")
        
        result = await cache.update_latest(snapshot1)
        if result:
            print(f"  ✓ Successfully updated cache")
        else:
            print(f"  ✗ Failed to update cache")
            return
        
        stats = cache.get_stats()
        print(f"\nStats after update:")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Total failed: {stats['total_failed']}")
        
        # Test multiple updates
        print(f"\nUpdating multiple symbols...")
        symbols = ['ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        for i, symbol in enumerate(symbols):
            snapshot = create_sample_orderbook(symbol, index=i+1)
            result = await cache.update_latest(snapshot)
            if result:
                print(f"  ✓ Updated {symbol}")
            else:
                print(f"  ✗ Failed to update {symbol}")
        
        stats = cache.get_stats()
        print(f"\nFinal stats:")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Total failed: {stats['total_failed']}")
        
    except Exception as e:
        print(f"  ✗ Update test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_get_latest(cache: RedisCache):
    """Test retrieving cached orderbook snapshots"""
    print("\n" + "=" * 70)
    print("TEST 3: Get Latest (GET)")
    print("=" * 70)
    
    if not cache or not cache.client:
        print("  Skipping - cache not initialized")
        return
    
    try:
        # First, ensure we have data
        snapshot = create_sample_orderbook('BTC/USDT', index=0)
        await cache.update_latest(snapshot)
        
        print(f"\nRetrieving cached snapshot for BTC/USDT...")
        result = await cache.get_latest('BTC/USDT')
        
        if result:
            print(f"  ✓ Successfully retrieved snapshot")
            print(f"  Symbol: {result['symbol']}")
            print(f"  Timestamp: {result['timestamp']}")
            print(f"  Best bid: {result['best_bid']}")
            print(f"  Best ask: {result['best_ask']}")
            print(f"  Spread: {result['spread']}")
            print(f"  Number of bids: {len(result['bids'])}")
            print(f"  Number of asks: {len(result['asks'])}")
            
            # Verify data integrity
            assert result['symbol'] == snapshot['symbol']
            assert result['timestamp'] == snapshot['timestamp']
            print(f"  ✓ Data integrity verified")
        else:
            print(f"  ✗ Failed to retrieve snapshot (key may not exist)")
        
        # Test retrieving non-existent key
        print(f"\nRetrieving non-existent key (TEST/USDT)...")
        result = await cache.get_latest('TEST/USDT')
        if result is None:
            print(f"  ✓ Correctly returned None for non-existent key")
        else:
            print(f"  ✗ Unexpectedly returned data for non-existent key")
        
        # Test retrieving multiple symbols
        print(f"\nRetrieving multiple symbols...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'NONEXISTENT/USDT']
        for symbol in symbols:
            result = await cache.get_latest(symbol)
            if result:
                print(f"  ✓ {symbol}: Found (best_bid={result['best_bid']})")
            else:
                print(f"  - {symbol}: Not found")
        
    except Exception as e:
        print(f"  ✗ Get test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_ttl_expiration(cache: RedisCache):
    """Test TTL expiration (wait for key to expire)"""
    print("\n" + "=" * 70)
    print("TEST 4: TTL Expiration")
    print("=" * 70)
    
    if not cache or not cache.client:
        print("  Skipping - cache not initialized")
        return
    
    try:
        # Create a snapshot with short TTL
        short_ttl_cache = RedisCache(redis_url=settings.REDIS_URL, ttl=5)
        await short_ttl_cache.initialize()
        
        snapshot = create_sample_orderbook('TTL_TEST/USDT', index=0)
        print(f"\nSetting key with 5-second TTL...")
        result = await short_ttl_cache.update_latest(snapshot)
        
        if result:
            print(f"  ✓ Key set successfully")
            
            # Immediately retrieve (should exist)
            print(f"\nImmediately retrieving key...")
            result = await short_ttl_cache.get_latest('TTL_TEST/USDT')
            if result:
                print(f"  ✓ Key exists (as expected)")
            else:
                print(f"  ✗ Key not found (unexpected)")
            
            # Wait for TTL to expire
            print(f"\nWaiting 6 seconds for TTL to expire...")
            await asyncio.sleep(6)
            
            # Try to retrieve again (should be None)
            print(f"Retrieving key after TTL expiration...")
            result = await short_ttl_cache.get_latest('TTL_TEST/USDT')
            if result is None:
                print(f"  ✓ Key correctly expired (returned None)")
            else:
                print(f"  ✗ Key still exists (TTL may not have expired)")
        
        await short_ttl_cache.close()
        
    except Exception as e:
        print(f"  ✗ TTL test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_error_handling(cache: RedisCache):
    """Test error handling scenarios"""
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)
    
    if not cache or not cache.client:
        print("  Skipping - cache not initialized")
        return
    
    try:
        # Test update without initialization
        print(f"\nTesting update without initialization...")
        uninitialized_cache = RedisCache()
        snapshot = create_sample_orderbook('TEST/USDT')
        result = await uninitialized_cache.update_latest(snapshot)
        if result is False:
            print(f"  ✓ Correctly returned False when not initialized")
        else:
            print(f"  ✗ Unexpectedly succeeded without initialization")
        
        # Test get without initialization
        print(f"\nTesting get without initialization...")
        result = await uninitialized_cache.get_latest('TEST/USDT')
        if result is None:
            print(f"  ✓ Correctly returned None when not initialized")
        else:
            print(f"  ✗ Unexpectedly returned data without initialization")
        
        # Test with invalid data (should still work, but verify)
        print(f"\nTesting with valid data structure...")
        valid_snapshot = create_sample_orderbook('VALID/USDT')
        result = await cache.update_latest(valid_snapshot)
        if result:
            print(f"  ✓ Successfully handled valid snapshot")
        
        stats = cache.get_stats()
        print(f"\nFinal error handling stats:")
        print(f"  Total updates: {stats['total_updates']}")
        print(f"  Total failed: {stats['total_failed']}")
        
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_statistics(cache: RedisCache):
    """Test statistics tracking"""
    print("\n" + "=" * 70)
    print("TEST 6: Statistics")
    print("=" * 70)
    
    if not cache or not cache.client:
        print("  Skipping - cache not initialized")
        return
    
    try:
        initial_stats = cache.get_stats()
        print(f"\nInitial statistics:")
        for key, value in initial_stats.items():
            print(f"  {key}: {value}")
        
        # Perform operations
        print(f"\nPerforming 5 update operations...")
        for i in range(5):
            snapshot = create_sample_orderbook('STATS_TEST/USDT', index=i)
            await cache.update_latest(snapshot)
        
        final_stats = cache.get_stats()
        print(f"\nFinal statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
        
        assert final_stats['total_updates'] == initial_stats['total_updates'] + 5
        print(f"  ✓ Statistics correctly tracked")
        
    except Exception as e:
        print(f"  ✗ Statistics test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("REDIS CACHE INTEGRATION TESTS")
    print("=" * 70)
    print(f"\nRedis URL: {settings.REDIS_URL}")
    print(f"Make sure Redis is running before proceeding!")
    
    cache = None
    try:
        # Run tests
        cache = await test_initialization()
        await test_update_latest(cache)
        await test_get_latest(cache)
        await test_ttl_expiration(cache)
        await test_error_handling(cache)
        await test_statistics(cache)
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cache and cache.client:
            print("\nClosing Redis connection...")
            await cache.close()
            print("✓ Connection closed")


if __name__ == "__main__":
    asyncio.run(main())

