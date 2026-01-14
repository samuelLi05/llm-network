from redis.asyncio import Redis as aioredis
from typing import Any, Optional, List
import json
import random
import time

"""Asynchronous Redis-based cache for storing and retrieving agent responses based on agent id."""
class RedisCache:
    def __init__(self, host:str='localhost', port:int=6379, db:int=1, prefix: str='cache:'):
        self.redis = aioredis(host=host, port=port, db=db)
        self.prefix = prefix
        self.ordered_set_key = f"{self.prefix}timestamps"

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    async def get_responses(self, agent_id: str, last_n: Optional[int] = None) -> List[Any]:
        """
        Retrieve responses for `agent_id`. If last_n is None, returns the full list.
        Returns a list of strings (JSON strings for structured items).
        """
        key = self._key(agent_id)
        if last_n is None:
            items = await self.redis.lrange(key, 0, -1)
        else:
            # fetch last `last_n` items
            items = await self.redis.lrange(key, -last_n, -1)
        return items or []
    
    async def get_random_responses(self, sample_size: int) -> List[Any]:
        """Retrieve a random sample of responses across all agents in the cache."""
        cursor = b'0'
        pattern = f"{self.prefix}*"
        all_items = []
        while cursor:
            cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                items = await self.redis.lrange(key, 0, -1)
                all_items.extend(items)
        if len(all_items) <= sample_size:
            return all_items
        return random.sample(all_items, sample_size)
    
    async def get_last_responses(self, sample_size:int) -> List[Any]:
        """Retrieve the last `sample_size` responses added across all agents in the cache based on their publish timestamp."""
        entries = await self.redis.zrevrange(self.ordered_set_key, 0, sample_size - 1)
        return entries
    
    async def append_response(self, agent_id:str, value: Any, expire: Optional[int] = None):
        key = self._key(agent_id)
        # normalize to string (JSON for dicts/lists)
        if isinstance(value, (dict, list)):
            payload = json.dumps(value, ensure_ascii=False)
        else:
            payload = str(value)
        # Add to key based list
        await self.redis.rpush(key, payload)
        # add Redis sorted set entry for timestamp ordering using current time as score
        timestamp = time.time()
        await self.redis.zadd(self.ordered_set_key, {payload: timestamp})

        if expire is not None:
            await self.redis.expire(key, expire)

    async def clear(self, agent_id: str) -> None:
        """Delete the entire list for this agent."""
        await self.redis.delete(self._key(agent_id))

    async def clear_all(self) -> None:
        """Delete all keys with the current prefix."""
        cursor = b'0'
        pattern = f"{self.prefix}*"
        while cursor:
            cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await self.redis.delete(*keys)

    async def close(self) -> None:
        await self.redis.close()
        await self.redis.connection_pool.disconnect() 

