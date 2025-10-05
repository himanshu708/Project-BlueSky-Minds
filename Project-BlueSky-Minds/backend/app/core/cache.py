# app/core/cache.py
import redis
import json
import logging
from typing import Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        try:
            # Connect to Redis/Valkey
            self.redis_client = redis.Redis(
                host=getattr(settings, "REDIS_HOST", "localhost"),
                port=getattr(settings, "REDIS_PORT", 6379),
                db=getattr(settings, "REDIS_DB", 0),
                decode_responses=True,
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Cache service connected successfully")
        except Exception as e:
            logger.warning(f"Cache service unavailable: {e}")

            self.redis_client = None

    # ADD this function at the end of cache.py


    async def init_redis():
        """Initialize Redis connection for startup (legacy compatibility)"""
        try:
            cache = await get_cache()
            if cache.is_available():
                logger.info("Redis/Cache initialized successfully")
                return True
            else:
                logger.warning("Redis/Cache not available, continuing without cache")
                return False
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            return False


    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        if not self.redis_client:
            return False

        try:
            serialized_value = json.dumps(value, default=str)
            return self.redis_client.setex(key, expire, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False

        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if cache is available"""
        return self.redis_client is not None


# Global cache instance
cache_service = CacheService()


async def get_cache():
    """Dependency for FastAPI"""
    return cache_service
