""" 
utility functions for the simulator
"""

def round_to_tick(price:float, tick_size:float) -> float:
    """
    round a price to the nearest tick size
    """
    return round(price / tick_size) * tick_size

def round_to_precision(price:float, precision:int) -> float:
    """
    round a price to the nearest precision
    """
    return round(price, precision)
