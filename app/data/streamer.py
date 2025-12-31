"""
manages web socket connections and data streaming from Binance, streams order book data.
Handles reconnection logic with exponential backoff.
"""

import asyncio
import logging
import ccxt.pro
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime, timezone

from app.core.config import settings
from app.data.exceptions import StreamConnectionError, StreamReconnectionError, DataValidationError

logger = logging.getLogger(__name__)

class OrderBookStreamer:
    """
    streams level 2 order book data from binance using ccxt websocket
    automatic reconnection with exponential backoff
    connection health monitoring
    data validation before emitting
    """
    

    def __init__(
        self,
        symbol: str = None,
        max_reconnect_attempts: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
    ):
        """
        initializes the streamer with configuration parameters
        """
        self.symbol = symbol or settings.SYMBOL
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier

        self.exchange: Optional[ccxt.pro.binance] = None
        self.is_running = False
        self.reconnect_count = 0
        self.last_message_time: Optional[datetime] = None

    
    def _validate_orderbook(self, orderbook: Dict[str, Any]) -> bool:
        """
        validate order book data structure
        takes in an order book snapshot from exhcnage
        true if valid, false if not
        """
        required_keys = ['bids', 'asks', 'timestamp']

        if not all(key in orderbook for key in required_keys):
            return False

        #validate non empty lists
        if not isinstance(orderbook['bids'], list) or not orderbook['bids']:
            return False
        if not isinstance(orderbook['asks'], list) or not orderbook['asks']:
            return False

        #validate bid / ask are price amount pairs
        for bid in orderbook['bids']:
            if not isinstance(bid, list) or len(bid) < 2:
                return False
            try:
                float(bid[0]) #price
                float(bid[1]) #amount
            except (ValueError, TypeError):
                return False
        

        for ask in orderbook['asks']:
            if not isinstance(ask, list) or len(ask) < 2:
                return False
            try:
                float(ask[0]) #price
                float(ask[1]) #amount
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _normalize_orderbook(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        normalizes order book data structure and adds computed fields
        takes in a raw order book from exchnage
        returns normalized orderbook with additional metadata
        """
        best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else None
        best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else None
        spread = best_ask - best_bid if (best_bid and best_ask) else None

        normalized = {
            'symbol': self.symbol,
            'timestamp': orderbook.get('timestamp'),
            'datetime': orderbook.get('datetime') or datetime.now(timezone.utc).isoformat(),
            'bids': orderbook['bids'],
            'asks': orderbook['asks'],
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_pct': (spread / best_bid * 100) if (spread and best_bid) else None
        }
        return normalized

    async def _create_exchange(self) -> ccxt.pro.binance:
        """
        creates and configures ccxt exchange instance (binance for now)
        returns the configured instance
        """
        exchange = ccxt.pro.binance({
            'apiKey': '',
            'secret': '',
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        logger.info(f"Created exchange instance for {self.symbol}")
        return exchange
    

    async def _calculate_backoff(self, attempt: int) -> float:
        """
        calculates backoff delay using exponential backoff
        takes in current attempt number, 0 indexed
        returns backoff delay in seconds
        """
        delay = self.initial_backoff * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_backoff)
    

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        THE main streaming method, yields normalized order book snapshots
        handles reconnection automatically
        raises stream reconnection error if max attempts exceeded
        """
        self.is_running = True
        self.reconnect_count = 0

        while self.is_running:
            try:
                #create new exchange instance
                self.exchange = await self._create_exchange()
                logger.info(f"connecting to binance WebSocket for {self.symbol}...")

                #watch_order_book returns a coroutine that resolves to a single orderbook
                while self.is_running:
                    orderbook = await self.exchange.watch_order_book(self.symbol)
                    
                    if not self.is_running:
                        break

                    #validate data
                    if not self._validate_orderbook(orderbook):
                        logger.warning(f"Invalid orderbook data received, skipping: {orderbook}")
                        continue

                    #normalize
                    normalized = self._normalize_orderbook(orderbook)
                    self.last_message_time = datetime.now(timezone.utc)
                    self.reconnect_count = 0 #reset of success

                    yield normalized
            except asyncio.CancelledError:
                logger.info("Stream cancelled by user")
                break
            except Exception as e:
                if not self.is_running:
                    break
                    
                self.reconnect_count += 1
                logger.error(f"Stream error (attempt {self.reconnect_count} / {self.max_reconnect_attempts}): {str(e)}", exc_info=True)


                if self.reconnect_count >= self.max_reconnect_attempts:
                    raise StreamReconnectionError(f"Failed to reconnect after {self.max_reconnect_attempts} attempts") from e
                
                #calculate backoff and wait
                backoff = await self._calculate_backoff(self.reconnect_count - 1)
                logger.info(f"reconnecting in {backoff:.2f} seconds ...")
                await asyncio.sleep(backoff)

            finally:
                if self.exchange:
                    try:
                        await self.exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing exchange: {e}")
                        self.exchange = None
    

    async def stop(self):
        """
        stops the streamer gracefully
        """
        logger.info("Stopping OrderBookStreamer...")
        self.is_running = False
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange during stop: {e}")


