"""
order matching engine with realistic slippage and fee calculations
use a priority queue for efficient order processing
"""

from typing import List, Optional, Tuple
from app.sim.orderbook import OrderBook
from app.sim.orders import Order, LimitOrder, MarketOrder, Fill, OrderSide, OrderStatus
from app.sim.exceptions import InsufficientLiquidityError, InvalidOrderError



class MatchingEngine:
    """
    matches orders against order book with realistic excecution
    """
    def __init__(
        self,
        maker_fee_rate: float = -0.0001,
        taker_fee_rate: float = 0.0001
        ):

        """
        initializes matching engine with fee rates
        """

        self.maker_fee_rate = maker_fee_rate
        self.taker_fee_rate = taker_fee_rate

    
    def match_limit_order(
        self,
        order: LimitOrder,
        order_book: OrderBook
    ) -> List[Fill]:
        """
        match a limit order against the order book
        limit buy: matches if ask price <= order price
        limit sell: matches if bid price >= order price
        fill through price levels until order is filled or price limit hit
        calculate maker fee

        returns a list of fills  (may be multiple partial fills)
        """

        fills = []
        remaining_qty = order.remaining_quantity

        if order.side == OrderSide.BUY:
            #buy orders match against asks
            if not order_book.asks:
                return fills
            
            best_ask = order_book.get_best_ask()
            if best_ask is None or best_ask > order.price:
                return fills
            
            #convert to list to avoid modification during iteration issues
            for ask_price, ask_qty in list(order_book.asks.items()):
                if ask_price > order.price:
                    break

                if remaining_qty <= 0:
                    break

                #skip if this level was already consumed
                if ask_price not in order_book.asks:
                    continue

                fill_qty = min(remaining_qty, order_book.asks[ask_price])
                fill_price = ask_price #fill at the ask price

                #maker fee (rebate so negative)
                fee = fill_qty * self.maker_fee_rate * fill_price

                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    fee=fee
                )
                fills.append(fill)
                remaining_qty -= fill_qty

                order_book.asks[ask_price] -= fill_qty
                if order_book.asks[ask_price] <= 0:
                    del order_book.asks[ask_price]
        

        else: #SELL
            #sell orders match against bids
            if not order_book.bids:
                return fills
            
            best_bid = order_book.get_best_bid()
            if best_bid is None or best_bid < order.price:
                return fills
            
            #convert to list to avoid modification during iteration issues
            for bid_price in reversed(list(order_book.bids.keys())):
                if bid_price < order.price:
                    break

                if remaining_qty <= 0:
                    break

                #skip if this level was already consumed
                if bid_price not in order_book.bids:
                    continue

                bid_qty = order_book.bids[bid_price]
                fill_qty = min(remaining_qty, bid_qty)
                fill_price = bid_price #fill at the bid price

                #maker fee (rebate so negative)
                fee = fill_qty * self.maker_fee_rate * fill_price

                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    fee=fee
                )
                fills.append(fill)
                remaining_qty -= fill_qty

                order_book.bids[bid_price] -= fill_qty
                if order_book.bids[bid_price] <= 0:
                    del order_book.bids[bid_price]


        

        if fills:
            total_filled = sum(f.quantity for f in fills)
            order.filled_quantity += total_filled

            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

        return fills

    def match_market_order(
        self,
        order: MarketOrder,
        order_book: OrderBook
    ) -> List[Fill]:
        """
        match a market order immediately at best available price.
        """
        fills = []
        remaining_qty = order.remaining_quantity
        
        if order.side == OrderSide.BUY:
            if not order_book.asks:
                raise InsufficientLiquidityError("No asks available for market buy")
            
            # Fill through asks until order is filled
            for ask_price, ask_qty in list(order_book.asks.items()):
                if remaining_qty <= 0:
                    break
                
                if ask_price not in order_book.asks:
                    continue
                
                fill_qty = min(remaining_qty, order_book.asks[ask_price])
                fill_price = ask_price
                
                # Taker fee (cost)
                fee = fill_qty * fill_price * self.taker_fee_rate
                
                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    fee=fee
                )
                fills.append(fill)
                
                remaining_qty -= fill_qty
                
                order_book.asks[ask_price] -= fill_qty
                if order_book.asks[ask_price] <= 0:
                    del order_book.asks[ask_price]
        
        else:  # SELL
            if not order_book.bids:
                raise InsufficientLiquidityError("No bids available for market sell")
            for bid_price in reversed(list(order_book.bids.keys())):
                if remaining_qty <= 0:
                    break
                
                if bid_price not in order_book.bids:
                    continue
                
                bid_qty = order_book.bids[bid_price]
                fill_qty = min(remaining_qty, bid_qty)
                fill_price = bid_price
                
                fee = fill_qty * fill_price * self.taker_fee_rate
                
                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    price=fill_price,
                    fee=fee
                )
                fills.append(fill)
                
                remaining_qty -= fill_qty
                
                order_book.bids[bid_price] -= fill_qty
                if order_book.bids[bid_price] <= 0:
                    del order_book.bids[bid_price]
        
        if remaining_qty > 0:
            # Market order couldn't be fully filled (rare but possible)
            raise InsufficientLiquidityError(
                f"Market order partially filled. Remaining: {remaining_qty}"
            )
        
        # Market orders always fill completely (or raise error)
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        
        return fills
    


    def calculate_slippage(
        self,
        fills: List[Fill],
        expected_price: float,
        side: OrderSide
    ) -> float:
        """
        calculate the slippage of a list of fills
        Slippage = (average_fill_price - expected_price) / expected_price
        """

        if not fills:
            return 0.0

        total_value = sum(f.quantity * f.price for f in fills)
        total_quantity = sum(f.quantity for f in fills)
        avg_fill_price = total_value / total_quantity

        if side == OrderSide.BUY:
            #for buys slippage is positive if avg fill price is higher than expected
            slippage = (avg_fill_price - expected_price) / expected_price
        else:
            #for sells slippage is negative if avg fill price is lower than expected
            slippage = (expected_price - avg_fill_price) / expected_price

        return slippage
    

        