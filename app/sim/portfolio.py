"""
portfolio management for tracking cash , position, and profit/loss
"""

from typing import Dict, List, Optional
from datetime import datetime
from app.sim.orders import Order, Fill, OrderSide, OrderStatus


class Portfolio:
    """

    tracks the agent's financial state and order history
    """
    def __init__(self, initial_cash:float = 10000.0, symbol:str = "BTC/USDT"):
        """
        initializes portfolio with initial cash and symbol
        """

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0
        self.symbol = symbol

        self.open_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []

        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0

    

    def place_order(self, order: Order) -> None:
        """
        adds order to open orders and updates portfolio
        """
        if order.order_id in self.open_orders:
            raise ValueError(f"Order with ID {order.order_id} already exists")
        
        if order.order_type.value == "LIMIT":
            if order.side == OrderSide.BUY:
                required_cash = order.quantity * order.price
                if required_cash > self.cash:
                    raise ValueError(f"Insufficient cash to place order. Required: {required_cash}, Available: {self.cash}")
            else: #sell
                if order.quantity > self.position:
                    raise ValueError(f"Insufficient position to place order. Required: {order.quantity}, Available: {self.position}")
            self.open_orders[order.order_id] = order

    

    def cancel_order(self, order_id: str) -> bool:
        """
        cancels an open order
        """
        if order_id not in self.open_orders:
            return False
        
        order = self.open_orders.pop(order_id)
        order.status = OrderStatus.CANCELLED

        #move it to history
        self.order_history.append(order)
        return True
    

    def record_fill(self, fill:Fill) -> None:
        """
        process a fill and update portfolio state
        """
        self.fill_history.append(fill)
        self.total_fees_paid += abs(fill.fee)

        if fill.side == OrderSide.BUY:
            #buying, spend cash, gain position
            cost = fill.quantity * fill.price + fill.fee
            self.cash -= cost
            self.position += fill.quantity
        else: #selling, gain cash, lose position
            revenue = fill.quantity * fill.price - abs(fill.fee)
            self.cash += revenue
            self.position -= fill.quantity
        

        #update order
        if fill.order_id in self.open_orders:
            order = self.open_orders[fill.order_id]
            order.filled_quantity += fill.quantity

            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.order_history.append(order)
                del self.open_orders[fill.order_id]
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                
    def get_total_value(self, current_price:float) -> float:
        """
        calculates the total portfolio value (cash + position value)

        """

        return self.cash + self.position * current_price
    

    def get_realized_pnl(self) -> float:
        """
        calculate realized pnl from closed positions
        uses average cost basis: realized pnl = sell revenue - (avg_buy_cost * quantity_closed)
        """
        if not self.fill_history:
            return 0.0
        
        total_buy_cost = 0.0
        total_buy_quantity = 0.0
        total_sell_revenue = 0.0
        total_sell_quantity = 0.0

        for fill in self.fill_history:
            if fill.side == OrderSide.BUY:
                total_buy_cost += fill.quantity * fill.price + fill.fee
                total_buy_quantity += fill.quantity
            else: #sell
                total_sell_revenue += fill.quantity * fill.price - abs(fill.fee)
                total_sell_quantity += fill.quantity
        
        # if we have no sells, no realized pnl
        if total_sell_quantity == 0:
            return 0.0
        
        # if we have no buys (only sells, shorting), realized pnl is just sell revenue
        if total_buy_quantity == 0:
            # for short positions, we'd need to track differently
            return 0.0
        
        # calculate average buy cost
        avg_buy_cost = total_buy_cost / total_buy_quantity
        
        # quantity that has been closed (realized)
        # if position >= 0: we still hold some, so quantity_closed = total_buy_quantity - position
        # but if position < 0: we've sold more than we bought (short), so all buys are closed
        if self.position >= 0:
            quantity_closed = max(0, min(total_buy_quantity - self.position, total_sell_quantity))
        else:
            # short position: all buys are closed
            quantity_closed = min(total_buy_quantity, total_sell_quantity)
        
        cost_of_closed = avg_buy_cost * quantity_closed
        realized_pnl = total_sell_revenue - cost_of_closed
        return realized_pnl
    
    def get_unrealized_pnl(self, current_price:float) -> float:
        """
        calculate unrealized pnl from open positions
        unrealized pnl = (current price - average cost) * position
        uses average cost basis method
        """

        if self.position == 0:
            return 0.0
        
        # calculate average cost of current position
        # for long positions: use average cost of buys
        # for short positions: use average price of sells
        if self.position > 0:
            # long position: calculate average buy cost
            buy_fills = [fill for fill in self.fill_history if fill.side == OrderSide.BUY]
            if not buy_fills:
                return 0.0
            
            total_cost = sum(fill.quantity * fill.price + fill.fee for fill in buy_fills)
            total_quantity = sum(fill.quantity for fill in buy_fills)
            if total_quantity == 0:
                return 0.0
            
            average_cost = total_cost / total_quantity
            return (current_price - average_cost) * self.position
        else:
            # short position: calculate average sell price
            sell_fills = [fill for fill in self.fill_history if fill.side == OrderSide.SELL]
            if not sell_fills:
                return 0.0
            
            total_revenue = sum(fill.quantity * fill.price - abs(fill.fee) for fill in sell_fills)
            total_quantity = sum(fill.quantity for fill in sell_fills)
            if total_quantity == 0:
                return 0.0
            
            average_price = total_revenue / total_quantity
            return (average_price - current_price) * abs(self.position)
        


    def get_stats(self, current_price: Optional[float] = None) -> Dict:
        """get portfolio statistics for debugging """
        stats = {
            'cash': self.cash,
            'position': self.position,
            'open_orders_count': len(self.open_orders),
            'total_fills': len(self.fill_history),
            'total_fees_paid': self.total_fees_paid,
            'realized_pnl': self.get_realized_pnl(),
        }
        
        if current_price:
            stats['total_value'] = self.get_total_value(current_price)
            stats['unrealized_pnl'] = self.get_unrealized_pnl(current_price)
            stats['total_pnl'] = stats['realized_pnl'] + stats['unrealized_pnl']
            stats['return_pct'] = (
                (stats['total_value'] - self.initial_cash) / self.initial_cash * 100
            )
        
        return stats