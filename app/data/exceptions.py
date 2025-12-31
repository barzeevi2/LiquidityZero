"""
Custom exceptions for the data ingestion module
"""


class IngestorException(Exception):
    """Base exception for all ingestor related errors """
    pass

class StreamConnectionError(IngestorException):
    """Raised when WebSocket connection fails"""
    pass

class StreamReconnectionError(IngestorException):
    """Raised when reconnection attempts are exhausted"""
    pass

class BufferOverflowError(IngestorException):
    """Raised when buffer exceeds maximum size"""
    pass

class StorageError(IngestorException):
    """Raised when database write operations fail"""
    pass

class CacheError(IngestorException):
    """Raised when Redis cashe operations fail"""
    pass

class DataValidationError(IngestorException):
    """Raised when order book data fails validation"""
    pass



