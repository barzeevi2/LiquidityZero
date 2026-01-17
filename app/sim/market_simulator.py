"""
market simulator that replays historical orderbook data
"""

import asyncio
import asyncpg
import json
import logging
from typing import List, Optional, Dict, Iterator
from datetime import datetime, timedelta

# Use orjson for faster JSON parsing (3-5x faster than standard json)
try:
    import orjson
    def json_loads(data: str) -> any:
        return orjson.loads(data)
except ImportError:
    # Fallback to standard json if orjson not available
    def json_loads(data: str) -> any:
        return json.loads(data)
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



    async def load_historical_data(
        self, 
        start_time: datetime, 
        end_time: datetime,
        max_snapshots: Optional[int] = None,
        sample_interval: Optional[int] = None,
        skip_gap_check: bool = False
    ) -> None:
        """
        loads orderbook snapshots from timescale db
        
        Args:
            start_time: Start time for data
            end_time: End time for data
            max_snapshots: Maximum number of snapshots to load (None = no limit)
            sample_interval: Sample every Nth snapshot (None = load all)
        """
        try:
            logger.info(f"Loading historical data for {self.symbol} from {start_time} to {end_time}...")
            conn = await asyncpg.connect(self.connection_string)

            # First, check how many rows we have
            count_query = """
                SELECT COUNT(*) as total
                FROM orderbook_snapshots
                WHERE symbol = $1
                AND time >= $2
                AND time <= $3
            """
            count_result = await conn.fetchrow(
                count_query,
                self.symbol,
                start_time,
                end_time
            )
            total_count = count_result['total'] if count_result else 0
            logger.info(f"Found {total_count} total snapshots in database")

            # Build query with optional sampling
            if sample_interval and sample_interval > 1:
                # Use TABLESAMPLE or row_number() for sampling
                # Pre-filter invalid dates and spreads at SQL level for better performance
                query = """
                    SELECT time, symbol, bids, asks, best_bid, best_ask, spread, spread_pct
                    FROM (
                        SELECT time, symbol, bids, asks, best_bid, best_ask, spread, spread_pct,
                               ROW_NUMBER() OVER (ORDER BY time ASC) as rn
                        FROM orderbook_snapshots
                        WHERE symbol = $1
                        AND time >= $2
                        AND time <= $3
                        AND time >= '2020-01-01'::timestamptz
                        AND time <= '2100-01-01'::timestamptz
                        AND (spread_pct IS NULL OR spread_pct <= 1.0)
                        AND spread_pct >= 0
                    ) sub
                    WHERE rn % $4 = 1
                    ORDER BY time ASC
                """
                query_params = (self.symbol, start_time, end_time, sample_interval)
            else:
                # Pre-filter invalid dates and unrealistic spreads at SQL level for performance
                query = """
                    SELECT time, symbol, bids, asks, best_bid, best_ask, spread, spread_pct
                    FROM orderbook_snapshots
                    WHERE symbol = $1
                    AND time >= $2
                    AND time <= $3
                    AND time >= '2020-01-01'::timestamptz
                    AND time <= '2100-01-01'::timestamptz
                    AND (spread_pct IS NULL OR spread_pct <= 1.0)
                    AND spread_pct >= 0
                    ORDER BY time ASC
                """
                if max_snapshots:
                    query += f" LIMIT {max_snapshots}"
                query_params = (self.symbol, start_time, end_time)

            logger.info("Executing query...")
            rows = await conn.fetch(query, *query_params)
            logger.info(f"Query returned {len(rows)} rows")

            self.snapshots = []
            total_rows = len(rows)
            logger.info(f"Processing {total_rows:,} snapshots (this may take a minute)...")

            skipped_count = 0
            last_time = None
            max_gap_seconds = 3600  # Max 1 hour gap between snapshots (filter out bad data)
            # Skip gap check if sampling (gaps expected) or if explicitly disabled
            should_check_gaps = not skip_gap_check and sample_interval is None
            
            # Progress logging every 5k rows for better feedback
            progress_interval = 5000
            
            for i, row in enumerate(rows):
                # Log progress more frequently
                if (i + 1) % progress_interval == 0:
                    progress_pct = ((i + 1) / total_rows) * 100
                    logger.info(f"Processing: {i + 1:,}/{total_rows:,} ({progress_pct:.1f}%) - {len(self.snapshots):,} valid snapshots so far...")
                # Validate and convert bids/asks
                # asyncpg returns JSONB as strings, need to parse them
                bids_raw = row.get('bids')
                asks_raw = row.get('asks')
                
                # Handle None values
                if bids_raw is None or asks_raw is None:
                    skipped_count += 1
                    continue
                
                # Parse JSONB - asyncpg may return as list or dict already
                try:
                    # asyncpg can return JSONB as list directly, or as string
                    if isinstance(bids_raw, str):
                        bids = json_loads(bids_raw)
                    elif isinstance(bids_raw, list):
                        bids = bids_raw  # Already parsed by asyncpg
                    elif isinstance(bids_raw, dict):
                        # Sometimes JSONB comes as dict, convert to list format
                        bids = list(bids_raw.items()) if bids_raw else []
                    else:
                        skipped_count += 1
                        continue
                    
                    if isinstance(asks_raw, str):
                        asks = json_loads(asks_raw)
                    elif isinstance(asks_raw, list):
                        asks = asks_raw  # Already parsed by asyncpg
                    elif isinstance(asks_raw, dict):
                        asks = list(asks_raw.items()) if asks_raw else []
                    else:
                        skipped_count += 1
                        continue
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    skipped_count += 1
                    continue
                
                # Ensure bids/asks are lists
                if not isinstance(bids, list) or not isinstance(asks, list):
                    skipped_count += 1
                    continue
                
                # Skip empty orderbooks
                if len(bids) == 0 or len(asks) == 0:
                    skipped_count += 1
                    continue
                
                # Validate first entry format (spot check - full validation happens in orderbook)
                try:
                    if len(bids) > 0:
                        first_bid = bids[0]
                        if not isinstance(first_bid, (list, tuple)) or len(first_bid) != 2:
                            skipped_count += 1
                            continue
                    if len(asks) > 0:
                        first_ask = asks[0]
                        if not isinstance(first_ask, (list, tuple)) or len(first_ask) != 2:
                            skipped_count += 1
                            continue
                except (IndexError, TypeError, AttributeError):
                    skipped_count += 1
                    continue
                
                # Dates and spreads are already filtered at SQL level, but double-check
                current_time = row['time']
                # Skip date check (already done in SQL) but validate spread_pct
                if row['spread_pct'] is not None and (row['spread_pct'] > 1.0 or row['spread_pct'] < 0):
                    skipped_count += 1
                    continue
                
                # Filter out snapshots with large time gaps (indicates bad data)
                # This check is still needed as it depends on sequence, not SQL filtering
                # OPTIMIZATION: Skip gap check if we're sampling (gaps are expected) or if disabled
                if should_check_gaps and last_time is not None:
                    gap_seconds = (current_time - last_time).total_seconds()
                    if gap_seconds > max_gap_seconds or gap_seconds < 0:
                        skipped_count += 1
                        continue  # Skip this snapshot if gap too large or negative
                
                #convert to dict
                snapshot = {
                    'datetime': row['time'].isoformat(),
                    'timestamp': row['time'].timestamp(),
                    'symbol': row['symbol'],
                    'bids': bids,
                    'asks': asks,
                    'best_bid': float(row['best_bid']) if row['best_bid'] else None,
                    'best_ask': float(row['best_ask']) if row['best_ask'] else None,
                    'spread': float(row['spread']) if row['spread'] else None,
                    'spread_pct': float(row['spread_pct']) if row['spread_pct'] else None,
                }
                self.snapshots.append(snapshot)
                last_time = current_time
            
            # Final summary
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count:,} invalid snapshots (malformed bids/asks, empty orderbook, invalid dates, large gaps, or unrealistic spreads)")

            await conn.close()

            if not self.snapshots:
                raise SimulationError(f"No orderbook snapshots found for {self.symbol} between {start_time} and {end_time}")
            
            logger.info(f"✓ Loaded {len(self.snapshots)} orderbook snapshots for {self.symbol} between {start_time} and {end_time}")
        
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

