import redis
from .config import settings

redis_client = redis.Redis.from_url(settings.redis_url)

def get_redis():
    return redis_client
