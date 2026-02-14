from redis.asyncio import Redis as aioredis
from typing import Any, Optional, List
import json
import random
import time
import hashlib
import asyncio
import os

"""Asynchronous Redis-based cache for storing and retrieving agent responses based on agent id."""
class RedisCache:
    def __init__(self, host:str='localhost', port:int=6379, db:int=1, prefix: str='cache:'):
        self.redis = aioredis(host=host, port=port, db=db)
        self.prefix = prefix
        self.ordered_set_key = f"{self.prefix}timestamps"
        self._dedupe_prefix = f"{self.prefix}dedupe:"

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _timeout_s(self) -> float:
        return float(os.getenv("REDIS_OP_TIMEOUT_S", "1.5"))
    
    async def get_responses(self, agent_id: str, last_n: Optional[int] = None) -> List[Any]:
        """
        Retrieve responses for `agent_id`. If last_n is None, returns the full list.
        Returns a list of strings (JSON strings for structured items).
        """
        key = self._key(agent_id)
        try:
            if last_n is None:
                items = await asyncio.wait_for(self.redis.lrange(key, 0, -1), timeout=self._timeout_s())
            else:
                # fetch last `last_n` items
                items = await asyncio.wait_for(self.redis.lrange(key, -last_n, -1), timeout=self._timeout_s())
            return items or []
        except Exception:
            return []
    
    async def get_random_responses(self, sample_size: int) -> List[Any]:
        """Retrieve a random sample of responses across all agents in the cache."""
        cursor = b'0'
        pattern = f"{self.prefix}*"
        all_items = []
        while cursor:
            try:
                cursor, keys = await asyncio.wait_for(
                    self.redis.scan(cursor=cursor, match=pattern, count=100),
                    timeout=self._timeout_s(),
                )
            except Exception:
                break
            for key in keys:
                try:
                    items = await asyncio.wait_for(self.redis.lrange(key, 0, -1), timeout=self._timeout_s())
                except Exception:
                    items = []
                all_items.extend(items)
        if len(all_items) <= sample_size:
            return all_items
        return random.sample(all_items, sample_size)
    
    async def get_last_responses(self, sample_size:int) -> List[Any]:
        """Retrieve the last `sample_size` responses added across all agents in the cache based on their publish timestamp."""
        try:
            entries = await asyncio.wait_for(
                self.redis.zrevrange(self.ordered_set_key, 0, sample_size - 1),
                timeout=self._timeout_s(),
            )
            return entries
        except Exception:
            return []
    
    async def append_response(self, agent_id:str, value: Any, expire: Optional[int] = None):
        key = self._key(agent_id)
        # normalize to string (JSON for dicts/lists)
        if isinstance(value, (dict, list)):
            payload = json.dumps(value, ensure_ascii=False)
        else:
            payload = str(value)
        # Add to key based list
        try:
            await asyncio.wait_for(self.redis.rpush(key, payload), timeout=self._timeout_s())
        except Exception:
            return
        # add Redis sorted set entry for timestamp ordering using current time as score
        timestamp = time.time()
        try:
            await asyncio.wait_for(
                self.redis.zadd(self.ordered_set_key, {payload: timestamp}),
                timeout=self._timeout_s(),
            )
        except Exception:
            return

        if expire is not None:
            try:
                await asyncio.wait_for(self.redis.expire(key, expire), timeout=self._timeout_s())
            except Exception:
                return

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
            ok = await asyncio.wait_for(
                self.redis.set(key, "1", ex=int(ttl_s), nx=True),
                timeout=self._timeout_s(),
            )
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
            await asyncio.wait_for(self.redis.delete(key), timeout=self._timeout_s())
        except Exception:
            return

    async def clear(self, agent_id: str) -> None:
        """Delete the entire list for this agent."""
        try:
            await asyncio.wait_for(self.redis.delete(self._key(agent_id)), timeout=self._timeout_s())
        except Exception:
            return

    async def clear_all(self) -> None:
        """Delete all keys with the current prefix."""
        cursor = b'0'
        pattern = f"{self.prefix}*"
        while cursor:
            try:
                cursor, keys = await asyncio.wait_for(
                    self.redis.scan(cursor=cursor, match=pattern, count=100),
                    timeout=self._timeout_s(),
                )
            except Exception:
                break
            if keys:
                try:
                    await asyncio.wait_for(self.redis.delete(*keys), timeout=self._timeout_s())
                except Exception:
                    return

    async def close(self) -> None:
        await self.redis.close()
        await self.redis.connection_pool.disconnect() 

