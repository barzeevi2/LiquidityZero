"""
data buffer, manages buffering of order book snapshots for batch processing
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from app.data.exceptions import BufferOverflowError

logger = logging.getLogger(__name__)

class DataBuffer:
    """
    buffers orderbook snapshots and flushes in batches
    async queue for thread safety
    automatic flushing based on size and time 
    overflow protection
    """

    def __init__(
        self,
        max_size: int = 10000,
        flush_size: int = 1000,
        flush_interval: float = 5.0
        ):
        """
        initializes a buffer, takes in max queue szie before overflow error, flush size which is number
        of items to accumulate before flush
        and time in seconds before auto flush
        """
        self.max_size = max_size
        self.flush_size = flush_size
        self.flush_interval = flush_interval

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.last_flush_time: Optional[datetime] = None
        self.total_processed = 0
        self.total_dropped = 0
    

    async def add(self, data:Dict[str, Any]) -> bool:
        """
        add order book snapshot to buffer
        takes in data: orderbook snapshot dict
        returns true if worked, false if queue full
        raises bufferoverflowerror if needed
        """
        try:
            await asyncio.wait_for(self.queue.put(data), timeout = 0.1)
            return True
        except asyncio.TimeoutError:
            #queue is full
            self.total_dropped +=1
            logger.warning(
                f"buffer queue full, dropped message."
                f"Total dropped: {self.total_dropped}, Queue size: {self.queue.qsize()}"
            )

            if self.queue.qsize() >= self.max_size:
                raise BufferOverflowError(
                    f"Buffer overflow: queue size {self.queue.qsize()} > max_size: {self.max_size}")
            return False
    

    async def flush(self) -> List[Dict[str, Any]]:
        """
        flush all items currently in the buffer
        returns a list of order book snapshots
        """
        items = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                items.append(item)
                self.total_processed += 1
            except asyncio.QueueEmpty():
                break
        
        if items:
            self.last_flush_time = datetime.now(timezone.utc)
            logger.debug(f"Flushed {len(items)} items from buffer")
        return items
    


    async def get_batch(self, block: bool = True) -> List[Dict[str, Any]]:
        """
        get a batch of items, wait if needed
        takes in block: if true, wait for items, if false return immediately
        returns list of orderbook snapshots 
        """
        batch = []
        if block:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout = self.flush_interval)
                batch.append(item)
            except asyncio.TimeoutError:
                pass
        
        #try to fill up to flush size, not blocking
        while len(batch) < self.flush_size:
            try:
                item = self.queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break
        

        if batch:
            self.total_processed += len(batch)
            self.last_flush_time = datetime.now(timezone.utc)
            logger.debug(f"Retrieved batch of {len(batch)} items from buffer")
        return batch

    

    def should_flush_by_time(self) -> bool:
        """
        checks if buffer should flush based on time interval
        returns true if flush interval has passed since last flush
        """
        if self.last_flush_time is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.last_flush_time).total_seconds()
        return elapsed >= self.flush_interval

    def should_flush_by_size(self) -> bool:
        """
        checks if buffer should flushed based on size
        """
        return self.queue.qsize() >= self.flush_size


    def size(self) -> int:
        #get current queue size
        return self.queue.qsize()
    

    def get_stats(self) -> Dict[str, Any]:
        # get buffer dictionary with stats
        return {
            'current_size': self.queue.qsize(),
            'max_size': self.max_size,
            'flush_size': self.flush_size,
            'total_processed': self.total_processed,
            'total_dropped': self.total_dropped,
            'last_flush_time': self.last_flush_time.isoformat() if self.last_flush_time else None
        }




