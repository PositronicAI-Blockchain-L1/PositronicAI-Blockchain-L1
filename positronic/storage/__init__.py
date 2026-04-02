"""Positronic - Storage Layer"""


class StorageError(Exception):
    """Base class for storage-layer errors."""
    pass


class DiskFullError(StorageError):
    """Raised when a write fails due to insufficient disk space."""
    pass


class StorageFatalError(StorageError):
    """Raised for unrecoverable storage errors requiring node halt."""
    pass
