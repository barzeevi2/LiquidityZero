"""
orders classes for limit and market orders
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    """
    base order class
    dataclass because clean and immutable
    __repr__ for easy debugging
    easy serialization if needed
    """
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: float = 0.0
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    price: Optional[float] = None



    @property
    def remaining_quantity(self) -> float:
        """
        remaining quantity to be filled
        """
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """
        check if order is filled
        """
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """
        check if the order is active
        """
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]



@dataclass
class LimitOrder(Order):
    """
    limit order class
    separate class for type safety, can add limit specific logic
    """
    order_type: OrderType = field(default=OrderType.LIMIT, init=False)

    def __post_init__(self):
        if self.price is None:
            raise ValueError("Limit order must have a price")


@dataclass
class MarketOrder(Order):
    """
    market order class
    separate class for type safety, can add market specific logic
    """
    order_type: OrderType = field(default=OrderType.MARKET, init=False)
    price: Optional[float] = field(default=None, init=False)



@dataclass
class Fill:
    """
    represents a fill, partial or full of an order
    orders can have multiple fills
    track execution details
    historical record of all trades
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cost(self) -> float:
        """
        calculate the cost of the fill
        """
        return self.quantity * self.price + self.fee


