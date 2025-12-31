"""
pytest config and fixtures
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def sample_orderbook():
    """Sample valid orderbook for testing"""
    return {
        'bids': [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 3.0]],
        'asks': [[50001.0, 1.2], [50002.0, 2.5], [50003.0, 1.8]],
        'timestamp': 1234567890,
        'datetime': '2024-01-01T00:00:00+00:00'
    }