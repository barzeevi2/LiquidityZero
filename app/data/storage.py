"""
storage engine, handles batch writes to timescledb 
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncpg
import json

from app.core.config import settings
from app.data.exceptions import StorageError

logger = logging.getLogger(__name__)

class StorageEngine:
    """
    manages writes to TimescaleDB for historical order book data 
    Async batch inserts using asynpg
    retry logic with exponential backoff
    connection pooling and error handling
    """

    def __init__(
        self,
        connection_string: str = None,
        pool_size: int = 5,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ):
        """
        Initialize the storage engine
        """
        self.connection_string = connection_string or settings.TIMESCALE_URL
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        self.pool: Optional[asyncpg.Pool] = None
        self.total_written = 0
        self.total_failed = 0
    


    async def initialize(self):
        """
        initialize database connection pool and create tables if needed
        """
        try:
            logger.info("Initializing TimescaleDB connection pool...")
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size = 1,
                max_size = self.pool_size,
                command_timeout = 60,
            )
            #create a table and hypertables if not exists
            await self._ensure_schema()
            logger.info("TimescaleDB connection pool initialized successfully")
        except Exception as e:
            raise StorageError(f"Failed to initialize TimescaleDB connection pool: {e}") from e
    
    async def _ensure_schema(self):
        """
        create table and hypertables if not exists
        """
        async with self.pool.acquire() as conn:
            #create table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    bids JSONB NOT NULL,
                    asks JSONB NOT NULL,
                    best_bid NUMERIC,
                    best_ask NUMERIC,
                    spread NUMERIC,
                    spread_pct NUMERIC,
                    PRIMARY KEY (time, symbol)
                );
            """)

            #create hypertables
            await conn.execute("""
                DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM _timescaledb_catalog.hypertable 
                            WHERE table_name = 'orderbook_snapshots'
                        ) THEN
                            PERFORM create_hypertable('orderbook_snapshots', 'time');
                        END IF;
                    END $$;
            """)

            logger.info("TimescaleDB schema created successfully")
    

    async def _calculate_backoff(self, attempt: int) -> float:
        return self.initial_backoff * 2 ** attempt
    

    async def _write_batch(self, snapshots: List[Dict[str, Any]]) -> int:
        """
        write a batch of order book snapshots to the db
        takes in a list of nornalized order book snapshots
        returns number of successful writes
        raises StorageError on failure
        """
        if not snapshots:
            return 0

        if not self.pool:
            raise StorageError("Storage engine not initialized")

        attempt = 0 

        while attempt < self.max_retries:
            try:
                async with self.pool.acquire() as conn:
                    #prepare batch data
                    rows = []
                    for snapshot in snapshots:
                        #parse timestamp
                        if 'timestamp' in snapshot and isinstance(snapshot['timestamp'], (int, float)):
                            ts = datetime.fromtimestamp(
                                snapshot['timestamp'] / 1000 if snapshot['timestamp'] > 1e10 else snapshot['timestamp'],
                                tz = timezone.utc
                            )
                        elif 'datetime' in snapshot and isinstance(snapshot['datetime'], str):
                            ts = datetime.fromisoformat(snapshot['datetime'].replace('Z', '+00:00'))
                        else:
                            ts = datetime.now(timezone.utc)
                        

                        rows.append((
                            ts,
                            snapshot['symbol'],
                            json.dumps(snapshot['bids']),
                            json.dumps(snapshot['asks']),
                            snapshot['best_bid'],
                            snapshot['best_ask'],
                            snapshot['spread'],
                            snapshot['spread_pct'],
                        ))
                    
                    #batch insert (outside loop - insert all rows at once)
                    await conn.executemany("""
                        INSERT INTO orderbook_snapshots 
                        (time, symbol, bids, asks, best_bid, best_ask, spread, spread_pct)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (time, symbol) DO NOTHING
                    """, rows)

                    written = len(rows)
                    self.total_written += written
                    logger.info(f"Successfully wrote {written} order book snapshots to TimescaleDB")
                    return written

            except Exception as e:
                attempt += 1
                self.total_failed += len(snapshots)

                if attempt >= self.max_retries:
                    logger.error(f"Failed to write order book snapshots to TimescaleDB after {self.max_retries} attempts: {e}")
                    raise StorageError(f"Failed to write order book snapshots to TimescaleDB after {self.max_retries} attempts: {e}") from e

                backoff = await self._calculate_backoff(attempt)
                logger.warning(f"Failed to write order book snapshots to TimescaleDB on attempt {attempt}: {e}. Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
        return 0

    async def write_batch(self, snapshots: List[Dict[str, Any]]) -> int:
        """
        Public method to write a batch of order book snapshots to the database.
        This is a wrapper around _write_batch for better API design.
        """
        return await self._write_batch(snapshots)

    
    async def close(self):
        """
        close the database connection pool
        """
        if self.pool:
            logger.info("Closing TimescaleDB connection pool...")
            await self.pool.close()
            self.pool = None
            logger.info("TimescaleDB connection pool closed successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        get statistics about the storage engine
        """
        return {
            "total_written": self.total_written,
            "total_failed": self.total_failed,
            "pool_size": self.pool_size if self.pool else 0
        }