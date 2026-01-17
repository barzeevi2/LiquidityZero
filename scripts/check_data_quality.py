#!/usr/bin/env python3
"""Script to check database data quality"""

import asyncio
import asyncpg
from datetime import datetime, timedelta
from app.core.config import settings

async def check_data():
    try:
        conn = await asyncpg.connect(settings.TIMESCALE_URL)
        
        # Check data range
        range_query = '''
            SELECT MIN(time) as min_time, MAX(time) as max_time, COUNT(*) as total_count
            FROM orderbook_snapshots
            WHERE symbol = $1
        '''
        result = await conn.fetchrow(range_query, 'BTC/USDT')
        print(f'Data Range: {result["min_time"]} to {result["max_time"]}')
        print(f'Total snapshots: {result["total_count"]:,}')
        
        # Check recent 24h
        recent_query = '''
            SELECT COUNT(*) as count, AVG(spread_pct) as avg_spread, 
                   MIN(spread_pct) as min_spread, MAX(spread_pct) as max_spread,
                   AVG(spread) as avg_spread_abs
            FROM orderbook_snapshots
            WHERE symbol = $1 AND time >= NOW() - INTERVAL '24 hours'
        '''
        recent = await conn.fetchrow(recent_query, 'BTC/USDT')
        print(f'\nLast 24h snapshots: {recent["count"]:,}')
        if recent["avg_spread"]:
            print(f'Avg spread %: {recent["avg_spread"]:.6f}%')
            print(f'Spread range: {recent["min_spread"]:.6f}% to {recent["max_spread"]:.6f}%')
        if recent["avg_spread_abs"]:
            print(f'Avg absolute spread: ${recent["avg_spread_abs"]:.4f}')
        
        # Check data gaps
        gap_query = '''
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT DATE_TRUNC('hour', time)) as unique_hours,
                COUNT(DISTINCT DATE_TRUNC('day', time)) as unique_days
            FROM orderbook_snapshots
            WHERE symbol = $1 AND time >= NOW() - INTERVAL '7 days'
        '''
        gap_result = await conn.fetchrow(gap_query, 'BTC/USDT')
        print(f'\nLast 7 days:')
        print(f'  Total snapshots: {gap_result["total"]:,}')
        print(f'  Unique hours: {gap_result["unique_hours"]}')
        print(f'  Unique days: {gap_result["unique_days"]}')
        
        # Check for gaps in data
        gap_detection = '''
            WITH time_diffs AS (
                SELECT 
                    time,
                    LAG(time) OVER (ORDER BY time) as prev_time,
                    time - LAG(time) OVER (ORDER BY time) as diff
                FROM orderbook_snapshots
                WHERE symbol = $1 AND time >= NOW() - INTERVAL '7 days'
                ORDER BY time
            )
            SELECT 
                COUNT(*) as total_gaps,
                AVG(EXTRACT(EPOCH FROM diff)) as avg_gap_seconds,
                MAX(EXTRACT(EPOCH FROM diff)) as max_gap_seconds
            FROM time_diffs
            WHERE diff IS NOT NULL
        '''
        gap_check = await conn.fetchrow(gap_detection, 'BTC/USDT')
        if gap_check["total_gaps"]:
            print(f'\nData Gaps (last 7 days):')
            print(f'  Total gaps: {gap_check["total_gaps"]:,}')
            print(f'  Avg gap: {gap_check["avg_gap_seconds"]:.1f} seconds')
            print(f'  Max gap: {gap_check["max_gap_seconds"]:.1f} seconds')
        
        # Check spread distribution
        spread_query = '''
            SELECT 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY spread_pct) as median_spread,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY spread_pct) as p25_spread,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY spread_pct) as p75_spread
            FROM orderbook_snapshots
            WHERE symbol = $1 AND time >= NOW() - INTERVAL '7 days' AND spread_pct IS NOT NULL
        '''
        spread_stats = await conn.fetchrow(spread_query, 'BTC/USDT')
        if spread_stats["median_spread"]:
            print(f'\nSpread Distribution (last 7 days):')
            print(f'  Median: {spread_stats["median_spread"]:.6f}%')
            print(f'  25th percentile: {spread_stats["p25_spread"]:.6f}%')
            print(f'  75th percentile: {spread_stats["p75_spread"]:.6f}%')
        
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_data())



