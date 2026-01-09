"""
custom exceptions for the simulator
"""

class SimulationError(Exception):
    """
    base class for all simulation exceptions
    """
    pass

class InsufficientLiquidityError(SimulationError):
    """
    raised when there is insufficient liquidity to complete a trade
    """
    pass

class InvalidOrderError(SimulationError):
    """
    raised when an order is invalid
    """
    pass
