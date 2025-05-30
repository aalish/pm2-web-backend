from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration"""

    # Sync intervals
    PROCESS_SYNC_INTERVAL: int = 120  # Increased from 30
    METAGRAPH_SYNC_INTERVAL: int = 300  # 5 minutes
    LOG_SYNC_INTERVAL: int = 120  # 2 minutes

    # Resource limits
    MAX_LOG_LINES: int = 20  # Reduced from 100
    MAX_CONCURRENT_SUBPROCESSES: int = 3
    SUBPROCESS_TIMEOUT: int = 5  # Reduced from 10

    # Caching
    ENABLE_LOG_CACHE: bool = True
    LOG_CACHE_TTL: int = 120
    METAGRAPH_CACHE_TTL: int = 300

    # Performance
    USE_ASYNC_SUBPROCESS: bool = True
    ENABLE_INCREMENTAL_UPDATES: bool = True
    ADAPTIVE_SYNC: bool = True


config = Config()
