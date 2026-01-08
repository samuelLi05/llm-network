import aioredis
from typing import Any, Optional, List
import json

"""Asynchronous Redis-based cache for storing and retrieving agent responses based on agent id."""
class RedisCache:
    def __init__(self, host:str='localhost', port:int=6379, db:int=1, prefix: str='cache:'):
        self.redis = aioredis.Redis(host=host, port=port, db=db)
        self.prefix = prefix

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
    
    async def append_response(self, agent_id:str, value: Any, expire: Optional[int] = None):
        key = self._key(agent_id)
        # normalize to string (JSON for dicts/lists)
        if isinstance(value, (dict, list)):
            payload = json.dumps(value, ensure_ascii=False)
        else:
            payload = str(value)
        await self.redis.rpush(key, payload)
        if expire is not None:
            await self.redis.expire(key, expire)

    async def clear(self, agent_id: str) -> None:
        """Delete the entire list for this agent."""
        await self.redis.delete(self._key(agent_id))

    async def close(self) -> None:
        await self.redis.close()
        await self.redis.connection_pool.disconnect() 

