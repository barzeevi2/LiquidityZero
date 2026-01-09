"""
market simulator that replays historical orderbook data
"""

import asyncio
import asyncpg
import logging
from typing import List, Optional, Dict, Iterator
from datetime import datetime, timedelta
from app.sim.orderbook import OrderBook
from app.sim.matching_engine import MatchingEngine
from app.sim.portfolio import Portfolio
from app.sim.orders import Order, LimitOrder
from app.core.config import settings
from app.sim.exceptions import SimulationError

logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    replays historical orderbook data and simulates agent trading
    """

    def __init__(
        self,
        symbol: str,
        connection_string: Optional[str] = None,
        initial_cash: float = 10000.0
    ):
        self.symbol = symbol
        self.connection_string = connection_string or settings.TIMESCALE_URL
        self.orderbook = OrderBook(symbol)
        self.match_engine = MatchingEngine()
        self.portfolio = Portfolio(initial_cash = initial_cash, symbol = symbol)


        self.snapshots: List[Dict] =[]
        self.current_index = 0
        self.current_time: Optional[datetime] = None

        self.is_running = False



    async def load_historical_data(self, start_time: datetime, end_time: datetime) -> None:
        """
        loads orderbook snapshots from timescale db
        """
        try:
            conn = await asyncpg.connect(self.connection_string)

            query = """
                SELECT time, symbol, bids, asks, best_bid, best_ask, spread, spread_pct
                FROM orderbook_snapshots
                WHERE symbol =$1
                AND time >= $2
                AND time <= $3
                ORDER BY time ASC
            """

            rows = await conn.fetch(
                query,
                self.symbol,
                start_time,
                end_time
            )

            self.snapshots = []

            for row in rows:
                #convert to dict
                snapshot = {
                    'datetime': row['time'].isoformat(),
                    'timestamp': row['time'].timestamp(),
                    'symbol': row['symbol'],
                    'bids': row['bids'],  # Already JSONB, should be list
                    'asks': row['asks'],
                    'best_bid': float(row['best_bid']) if row['best_bid'] else None,
                    'best_ask': float(row['best_ask']) if row['best_ask'] else None,
                    'spread': float(row['spread']) if row['spread'] else None,
                    'spread_pct': float(row['spread_pct']) if row['spread_pct'] else None,
                }
                self.snapshots.append(snapshot)

            await conn.close()

            if not self.snapshots:
                raise SimulationError(f"No orderbook snapshots found for {self.symbol} between {start_time} and {end_time}")
            
            logger.info(f"Loaded {len(self.snapshots)} orderbook snapshots for {self.symbol} between {start_time} and {end_time}")
        
        except Exception as e:
            raise SimulationError(f"Error loading historical data: {str(e)}") from e
        
    

    def reset(self, start_index: int =0) -> None:
        """
        reset simulator to start of data or specific index
        """

        if not self.snapshots:
            raise SimulationError("No snapshots loaded. Call load_historical_data() first.")
        
        if start_index >= len(self.snapshots) or start_index < 0:
            raise SimulationError(f"Start index {start_index} is out of range [0, {len(self.snapshots)})")
        
        self.current_index = start_index
        self.orderbook = OrderBook(self.symbol)
        
        #reset portfolio to initial state preserving initial_cash
        initial_cash = self.portfolio.initial_cash
        self.portfolio = Portfolio(initial_cash=initial_cash, symbol=self.symbol)

        if start_index < len(self.snapshots):
            self.orderbook.update_from_snapshot(self.snapshots[start_index])
            self.current_time = datetime.fromisoformat(self.snapshots[start_index]['datetime'])
    


    async def step(self) -> bool:
        """
        advance simulation by one time step

        Steps:
        1. update order book from next snapshot
        2. match agent limit order against updated book
        3. advance time

        returns true if more data available, false if end of data
        """

        if self.current_index >= len(self.snapshots) - 1:
            return False
        
        #advance to the next snapshot
        self.current_index += 1
        snapshot = self.snapshots[self.current_index]


        #update order book
        self.orderbook.update_from_snapshot(snapshot)
        self.current_time = datetime.fromisoformat(snapshot['datetime'])
        
        #re add agent orders to the orderbook after snapshot update
        #(update_from_snapshot clears bids/asks, so we need to restore active agent orders)
        #only re add orders that are still open in portfolio with remaining quantity
        for order in self.portfolio.open_orders.values():
            if order.order_type.value == "LIMIT" and isinstance(order, LimitOrder):
                if order.order_id in self.orderbook.agent_orders:
                    remaining_qty = order.remaining_quantity
                    if remaining_qty > 0:
                        price = order.price
                        side = order.side.value
                        
                        if side.upper() == 'BUY':
                            if price in self.orderbook.bids:
                                self.orderbook.bids[price] += remaining_qty
                            else:
                                self.orderbook.bids[price] = remaining_qty
                        else:  # SELL
                            if price in self.orderbook.asks:
                                self.orderbook.asks[price] += remaining_qty
                            else:
                                self.orderbook.asks[price] = remaining_qty


        #match agent open orders against updated book
        orders_to_check = list(self.portfolio.open_orders.values())
        for order in orders_to_check:
            if order.order_type.value == "LIMIT" and isinstance(order, LimitOrder):
                fills = self.match_engine.match_limit_order(order, self.orderbook)

                for fill in fills:
                    self.portfolio.record_fill(fill)
        return True

    
    def place_agent_order(self, order: LimitOrder) -> None:
        """
        place a limit order from the agent

        steps:
        1. add to portfolio
        2. add to order book so it can be matched
        """

        self.portfolio.place_order(order)
        self.orderbook.add_agent_order(
            order.order_id,
            order.price,
            order.quantity,
            order.side.value
        )

    
    def cancel_agent_order(self, order_id: str) -> bool:
        """
        cancel an agent limit order
        """
        success = self.portfolio.cancel_order(order_id)
        if success:
            self.orderbook.remove_agent_order(order_id)
        return success
    


    def get_current_state(self) -> Dict:
        """
        get the current market state for observation
        """
        mid_price = self.orderbook.get_mid_price()

        return {
            'timestamp': self.current_time,
            'best_bid': self.orderbook.get_best_bid(),
            'best_ask': self.orderbook.get_best_ask(),
            'mid_price': mid_price,
            'spread': self.orderbook.get_spread(),
            'spread_pct': self.orderbook.get_spread_percentage(),
            'imbalance': self.orderbook.get_imbalance(),
            'depth': self.orderbook.get_depth_levels(10),
            'portfolio': self.portfolio.get_stats(current_price=mid_price)
        }

