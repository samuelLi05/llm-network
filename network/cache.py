from redis.asyncio import Redis as aioredis
from typing import Any, Optional, List
import json
import random
import time
import hashlib

"""Asynchronous Redis-based cache for storing and retrieving agent responses based on agent id."""
class RedisCache:
    def __init__(self, host:str='localhost', port:int=6379, db:int=1, prefix: str='cache:'):
        self.redis = aioredis(host=host, port=port, db=db)
        self.prefix = prefix
        self.ordered_set_key = f"{self.prefix}timestamps"
        self._dedupe_prefix = f"{self.prefix}dedupe:"

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

    @staticmethod
    def normalize_for_fingerprint(text: str) -> str:
        t = (text or "").lower()
        # Coarse normalization: collapse whitespace and strip urls/hashtags/mentions.
        # (NetworkAgent also does its own normalization; this is just a stable fallback.)
        import re

        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"[#@]\w+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @classmethod
    def fingerprint_text(cls, text: str) -> str:
        norm = cls.normalize_for_fingerprint(text)
        return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()

    async def reserve_fingerprint(self, fingerprint: str, *, ttl_s: int = 180) -> bool:
        """Attempt to reserve a fingerprint globally for a short TTL.

        This provides a cheap, atomic "idempotency" guard against multiple agents
        publishing identical/near-identical outputs concurrently.
        """
        fp = (fingerprint or "").strip()
        if not fp:
            return True
        key = f"{self._dedupe_prefix}{fp}"
        try:
            # SET key 1 NX EX ttl
            ok = await self.redis.set(key, "1", ex=int(ttl_s), nx=True)
            return bool(ok)
        except Exception:
            # If Redis is unavailable, fail open (do not block publishing).
            return True

    async def release_fingerprint(self, fingerprint: str) -> None:
        fp = (fingerprint or "").strip()
        if not fp:
            return
        key = f"{self._dedupe_prefix}{fp}"
        try:
            await self.redis.delete(key)
        except Exception:
            return

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

