"""
the main ingestor service, orchestrates all components for data ingestion.
"""

import asyncio
import logging
import signal
from typing import Optional, List
from datetime import datetime, timezone

from app.data.streamer import OrderBookStreamer
from app.data.buffer import DataBuffer
from app.data.storage import StorageEngine
from app.data.cache import RedisCache
from app.data.exceptions import IngestorException

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class IngestorService:
    """
    main orchetrator for data ingestion pipeline, manages streamers, buffers, storage and cache.
    """
    def __init__(
        self,
        buffer_size: int = 400,
        buffer_interval: float = 5.0,
        enable_storage: bool = True,
        enable_cache: bool = True
    ):
        """
        initializes the ingestor service, takes in the buffer size, buffer interval, enable storage and enable cache
        """
        self.streamer = OrderBookStreamer()
        self.buffer = DataBuffer(flush_size=buffer_size, flush_interval=buffer_interval)
        self.storage = StorageEngine() if enable_storage else None
        self.cache = RedisCache() if enable_cache else None
 
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []

        self.start_time: Optional[datetime] = None
        self.total_received = 0

    
    async def _stream_consumer(self):
        """
        consumes order book snapshows from the streamer and buffers them
        """
        try:
            async for snapshot in self.streamer.stream():
                if not self.is_running or self.shutdown_event.is_set():
                    break
                await self.buffer.add(snapshot)
                self.total_received += 1

                #update the cache with the latest snapshot
                if self.cache:
                    await self.cache.update_latest(snapshot)
        
        except Exception as e:
            logger.error(f"Error in stream consumer: {e}")
            if not self.shutdown_event.is_set():
                raise IngestorException(f"Error in stream consumer: {e}") from e
    
    async def _storage_writer(self):
        """
        flushes the buffered data to storage (timescale db)
        """
        if not self.storage:
            return
        try:
            while self.is_running and not self.shutdown_event.is_set():
                should_flush = (
                    self.buffer.should_flush_by_size() or
                    self.buffer.should_flush_by_time()
                )
                if should_flush:
                    batch = await self.buffer.get_batch(block=False)
                    if batch:
                        try:
                            await self.storage.write_batch(batch)
                        except Exception as e:
                            logger.error(f"Error writing batch to storage: {e}")
                
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Storage writer task cancelled")
        except Exception as e:
            logger.error(f"Error in storage writer: {e}")


    async def _flush_remaining(self):
        """
        flushes any remaining data in the buffer before shutdown
        """
        logger.info("Flushing remaining data in buffer before shutdown...")
        remaining = await self.buffer.flush()
        if remaining and self.storage:
            try:

                await self.storage.write_batch(remaining)
                logger.info(f"Flushed {len(remaining)} remaining items to storage")
            except Exception as e:
                logger.error(f"Error flushing remaining data to storage: {e}")
    

    async def _health_monitor(self):
        """
        monitor the service health and log stats periodically
        """
        try:
            while not self.shutdown_event.is_set() and self.is_running:
                await asyncio.sleep(30) #log every 30 seconds

                if not self.is_running:
                    break
                
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
                buffer_stats = self.buffer.get_stats()

                stats_msg = (
                    f"Health Check - Uptime: {uptime:.0f}s | "
                    f"Received: {self.total_received} | "
                    f"Buffer: {buffer_stats['current_size']}/{buffer_stats['max_size']} | "
                    f"Processed: {buffer_stats['total_processed']} | "
                    f"Dropped: {buffer_stats['total_dropped']}"
                )

                if self.storage:
                    storage_stats = self.storage.get_stats()
                    stats_msg += f" | DB Written: {storage_stats['total_written']} | Failed: {storage_stats['total_failed']}"
                
                if self.cache:
                    cache_stats = self.cache.get_stats()
                    stats_msg += f" | Cache Updates: {cache_stats['total_updates']}"
                
                logger.info(stats_msg)
                
        except asyncio.CancelledError:
            logger.info("Health monitor cancelled")
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
    

    async def start(self):
        """
        starts the ingestor service
        """

        if self.is_running:
            logger.warning("Ingestor service is already running")
            return
        
        logger.info("-" * 40)
        logger.info("Starting Ingestor Service")
        logger.info("-" * 40)
        
        try:
            if self.storage:
                await self.storage.initialize()
            if self.cache:
                await self.cache.initialize()
            
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)
            self.shutdown_event.clear()

            #start the tasks
            self.tasks = [
                asyncio.create_task(self._stream_consumer(), name="stream_consumer"),
                asyncio.create_task(self._storage_writer(), name="storage_writer"),
                asyncio.create_task(self._health_monitor(), name="health_monitor")
            ]

            logger.info("Ingestor service started successfully")
            logger.info("-" * 40)
            
            await self.shutdown_event.wait()
        
        except Exception as e:
            logger.error(f"Error starting ingestor service: {e}")
            raise IngestorException(f"Error starting ingestor service: {e}") from e
    
    async def stop(self):
        """
        stops the ingestor service
        """
        if not self.is_running:
            return
        
        logger.info("-" * 40)
        logger.info("Stopping Ingestor Service")
        logger.info("-" * 40)

        self.is_running = False
        self.shutdown_event.set()

        await self.streamer.stop()

        for task in self.tasks:
            if not task.done():
                task.cancel()
        #wait for all tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        #flush any remaining data in the buffer
        await self._flush_remaining()
        
        if self.storage:
            await self.storage.close()
        if self.cache:
            await self.cache.close()
        
        logger.info("Ingestor service stopped successfully")
        logger.info("-" * 40)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        
        stats = {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'total_received': self.total_received,
            'buffer': self.buffer.get_stats(),
        }
        
        if self.storage:
            stats['storage'] = self.storage.get_stats()
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats


async def main():
    """
    main entry point for the ingestor service
    """
    ingestor = IngestorService()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(ingestor.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await ingestor.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await ingestor.stop()


if __name__ == "__main__":
    asyncio.run(main())