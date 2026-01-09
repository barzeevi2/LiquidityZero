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
        this.symbol = symbol

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
        for price, quantity in snapshot['bids']:
            price_float = float(price)
            quantity_float = float(quantity)
            if price_float in self.bids:
                self.bids[price_float] += quantity_float
            else:
                self.bids[price_float] = quantity_float
        

        #asks: price -> quantity will be sorted in ascending order
        for price, quantity in snapshot['asks']:
            price_float = float(price)
            quantity_float = float(quantity)
            if price_float in self.asks:
                self.asks[price_float] += quantity_float
            else:
                self.asks[price_float] = quantity_float

        bids_dict = dict(self.bids.items())
        self.bids = SortedDict({k: v for k, v in sorted(bids_dict.items(), reverse=True)})

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
                #resort
                bids_dict = dict(self.bids.items())
                self.bids = SortedDict({k: v for k, v in sorted(bids_dict.items(), reverse=True)})
        else:  # SELL
            if price in self.asks:
                self.asks[price] += quantity
            else:
                self.asks[price] = quantity
    
        self.agent_orders[order_id] = (price, quantity, side)
    

    def remove_agent_order(self, order_id: str) -> None:
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

    
    def get_best_bid(self) -> Optional[float]:
        """
        get the best bid price
        """
        if not self.bids:
            return None
        return self.bids.keys()[-1] #last key in descending order is the highest
    
    def get_best_ask(self) -> Optional[float]:
        """
        get the best ask price
        """
        if not self.asks:
            return None
        return self.asks.keys()[0] #first key in ascending order is the lowest
    
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
            for bid_price, quantity in self.bids.items():
                if bid_price >= price:
                    total += self.bids[bid_price]
                else:
                    break
        return total
        
        