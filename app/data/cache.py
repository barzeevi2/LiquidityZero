"""
redis cache, manages real time order book data in memory using redis
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
import redis.asyncio as redis

from app.core.config import settings
from app.data.exceptions import CacheError

logger = logging.getLogger(__name__)

class RedisCache:
    """
    manages redis cache for latest order book data
    async redis operations
    automatic ttl on cached data
    error handling with fallback
    """

    def __init__(
        self,
        redis_url: str = None,
        ttl: int = 60,
        max_retries: int = 3,
    ):
        """
        initializes the redis cache, takes in the redis url, the time to live for cached
        data and the maximum number of retries for redis operations
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.ttl = ttl
        self.max_retries = max_retries
        self.client: Optional[redis.Redis] = None
        self.total_updates = 0
        self.total_failed = 0
    
    async def initialize(self):
        """
        initializes the redis client and connects to the redis server
        """
        try:
            logger.info("Initializing Redis cache...")
            self.client = await redis.from_url(self.redis_url, decode_responses=False, max_connections=10)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            raise CacheError(f"Failed to initialize Redis cache: {e}") from e
    
    async def update_latest(self, snapshot: Dict[str, Any]) -> bool:
        """
        update the latest order book snapshot in redis
        takes in the snapshot, a normalized order book snapshot
        returns true if successful, false otherwise
        """
        if not self.client:
            logger.warning("Redis cache not initialized, skipping update")
            return False
        key = f"orderbook:{snapshot['symbol']}"

        try:
            #serialize the snapshot to json
            value = json.dumps(snapshot).encode('utf-8')

            #set with ttl
            await self.client.setex(key, self.ttl, value)

            self.total_updates += 1
            logger.debug(f"updated redis cache for {key}")
            return True
        except Exception as e:
            self.total_failed +=1
            logger.warning(f"failed to update redis cache for {key}: {e}")
            return False
    

    async def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        get the latest order book snapshot from redis
        takes in the symbol
        returns the snapshot if found, None otherwise
        """

        if not self.client:
            logger.warning("Redis cache not initialized, returning None")
            return None
        key = f"orderbook:{symbol}"

        try:
            #get the value from redis
            value = await self.client.get(key)
            if value:
                #deserialize the json
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.warning(f"failed to get latest order book snapshot for {symbol}: {e}")
            return None
    

    async def close(self):
        """
        close the redis client
        """
        if self.client:
            print("closing redis client...")
            await self.client.close()
            logger.info("Redis cache closed successfully")
            self.client = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        get statistics about the redis cache
        """
        return {
            "total_updates": self.total_updates,
            "total_failed": self.total_failed,
            "ttl": self.ttl,
        }