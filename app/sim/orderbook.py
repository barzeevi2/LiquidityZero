"""
in memory orderbook representation with efficient price level management
"""

from typing import Dict, List, Optional, Tuple
from sortedcontainers import SortedDict
from datetime import datetime


class OrderBook:
    """
    maintains order book state with efficient lookup and matching operations

    using sorteddict for O(logn) insert delete lookups
    it maintains sorted order automatically
    efficient iteration over price levels

    """

    def __init__(self, symbol:str):
        self.symbol = symbol

        #SortedDict: price -> quantity at that price level
        #bids: descending (highest price first) reverse=True
        #asks: ascending (lowest price first) default
        self.bids = SortedDict() #price -> quantity
        self.asks = SortedDict() #price -> quantity

        #agent orders stored separately: order_id -> (price, quantity, side)
        self.agent_orders: Dict[str, Tuple[float, float, str]] = {}

        self.last_update_time: Optional[datetime] = None
    

    def update_from_snapshot(self, snapshot:Dict) -> None:
        """
        rreplace entire orderbook from a snapshot

        full replacement because snapshot are the source of truth from the exchange
        ensures consistency and correct state 
        """
        self.bids.clear()
        self.asks.clear()
        

        #bids: price -> quantity will be sorted in descending order
        # Validate bids format before processing
        bids = snapshot.get('bids', [])
        if not isinstance(bids, list):
            raise ValueError(f"Invalid bids format: expected list, got {type(bids)}")
        
        for bid_entry in bids:
            try:
                if not isinstance(bid_entry, (list, tuple)) or len(bid_entry) != 2:
                    continue  # Skip malformed entries
                price, quantity = bid_entry
                price_float = float(price)
                quantity_float = float(quantity)
                if price_float in self.bids:
                    self.bids[price_float] += quantity_float
                else:
                    self.bids[price_float] = quantity_float
            except (ValueError, TypeError, IndexError) as e:
                # Skip malformed bid entries
                continue

        #asks: price -> quantity will be sorted in ascending order
        # Validate asks format before processing
        asks = snapshot.get('asks', [])
        if not isinstance(asks, list):
            raise ValueError(f"Invalid asks format: expected list, got {type(asks)}")
        
        for ask_entry in asks:
            try:
                if not isinstance(ask_entry, (list, tuple)) or len(ask_entry) != 2:
                    continue  # Skip malformed entries
                price, quantity = ask_entry
                price_float = float(price)
                quantity_float = float(quantity)
                if price_float in self.asks:
                    self.asks[price_float] += quantity_float
                else:
                    self.asks[price_float] = quantity_float
            except (ValueError, TypeError, IndexError) as e:
                # Skip malformed ask entries
                continue

        # SortedDict maintains ascending order automatically
        #for bids (descending), we'll iterate in reverse when needed

        self.last_update_time = datetime.fromisoformat(
            snapshot.get('datetime', snapshot.get('timestamp', ''))
        ) if snapshot.get('datetime') else None

    
    def add_agent_order(self, order_id: str, price: float, quantity: float, side: str) -> None:
        """
        add an agent order to the orderbook

        tracked seprately because they dont exist in historical snapshots
        we need to match them against market updates
        we need to cancel / modify them independently 
        """
        if side.upper() == 'BUY':
            if price in self.bids:
                self.bids[price] += quantity
            else:
                self.bids[price] = quantity
                
        else:  # SELL
            if price in self.asks:
                self.asks[price] += quantity
            else:
                self.asks[price] = quantity
    
        self.agent_orders[order_id] = (price, quantity, side)
    

    def remove_agent_order(self, order_id: str) -> bool:
        """
        remove an agent order from the orderbook
        """
        if order_id in self.agent_orders:
            price, quantity, side = self.agent_orders[order_id]
            if side.upper() == 'BUY':
                if price in self.bids:
                    self.bids[price] -= quantity
                    if self.bids[price] <= 0:
                        del self.bids[price]
            else:  # SELL
                if price in self.asks:
                    self.asks[price] -= quantity
                    if self.asks[price] <= 0:
                        del self.asks[price]
            
            del self.agent_orders[order_id]
            return True
        
        return False

    
    def get_best_bid(self) -> Optional[float]:
        """
        get the best bid price
        """
        if not self.bids:
            return None
        # SortedDict maintains ascending order, so last key is highest
        return list(self.bids.keys())[-1]
    
    def get_best_ask(self) -> Optional[float]:
        """
        get the best ask price
        """
        if not self.asks:
            return None
        # SortedDict maintains ascending order, so first key is lowest
        return next(iter(self.asks.keys()))
    
    def get_mid_price(self) -> Optional[float]:
        """
        calculate mid price (average of best bid and best ask)
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """
        calculate spread (difference between best bid and best ask)
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_spread_percentage(self) -> Optional[float]:
        """
        calculate spread percentage (spread / mid price)
        """
        spread = self.get_spread()
        mid_price = self.get_mid_price()

        if spread and mid_price and mid_price > 0:
            return (spread / mid_price) * 100
        return None
    
    def get_depth(self, price: float, side: str) -> Optional[float]:
        """
        get total liquidity available at or better than given price
        determines if we can fill large orders
        calculates slippage or market orders
        used for feature engineering (market depth)
        """

        total = 0.0
        if side.upper() == 'BUY':
            #for buy orders we want asks at or  below price
            for ask_price, quantity in self.asks.items():
                if ask_price <= price:
                    total += quantity
                else:
                    break
        else: #sell
            #for sell we want bids at or above price
            # iterate in reverse (highest to lowest) to get best prices first
            for bid_price, quantity in reversed(list(self.bids.items())):
                if bid_price >= price:
                    total += quantity
                else:
                    break
        return total
        
    

    def get_imbalance(self) -> Optional[float]:
        """
        calculate order book imbalance (difference between buy and sell volume)
        formula: (buy volume - sell volume) / (buy volume + sell volume)
        range: -1 to 1
        -1: all sell orders, no buy orders (bearish)
        0: balanced (neutral)
        1: all buy orders, no sell orders (bullish)

        predicts short term price movement
        important feature
        """
        #get volume at top 10 levels
        #for bids, top levels are the last 10 (highest prices)
        bid_items = list(self.bids.values())
        bid_volume = sum(bid_items[-10:]) if len(bid_items) > 10 else sum(bid_items)
        #for asks, top levels are the first 10 (lowest prices)
        ask_items = list(self.asks.values())
        ask_volume = sum(ask_items[:10])

        total_volume = bid_volume + ask_volume
        if total_volume ==0 :
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume

    

    def get_depth_levels(self, n_levels: int) -> Tuple[List[List[float]], List[List[float]]]:
        """
        get price quantity pairs for top N levels on each side
        returns: (bid_levels, ask_levels)
        each level is a list [price, quantity]

        used for feature engineering (orderbook depth)
        """
        #for bids, take last n_levels (highest prices) in reverse order
        bid_items = list(self.bids.items())
        bid_levels = [
            [price, quantity] for price, quantity in reversed(bid_items[-n_levels:])
        ]
        #for asks, take first n_levels (lowest prices)
        ask_levels = [
            [price, quantity] for price, quantity in list(self.asks.items())[:n_levels]
        ]
        return bid_levels, ask_levels
