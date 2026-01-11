import random
from typing import List, Optional

from agents.network_agent import NetworkAgent
from network.cache import RedisCache
from network.stream import RedisStream

"""Manages ordering and coordination of responses among multiple NetworkAgents.

Selects a random agent excluding the sender of the most recent message
from the Redis stream. The selected next responder is stored in Redis
so all agents see a consistent value.
"""

# Redis key for storing the designated next responder
NEXT_RESPONDER_KEY = "llm_network:next_responder"


class OrderManager:
    def __init__(
        self,
        agents: List[NetworkAgent],
        message_cache: RedisCache,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
    ):
        self.agents = agents
        self.message_cache = message_cache
        # RedisStream uses a synchronous redis client with decode_responses=True
        self.stream_client = RedisStream(host=redis_host, port=redis_port)

    def get_random_agent(self, exclude_agent_id: Optional[str] = None) -> str:
        """Return a random agent id, optionally excluding one agent.

        Args:
            exclude_agent_id: Agent id to exclude from selection (e.g., last publisher).

        Returns:
            The selected agent's id.

        Raises:
            ValueError: If no agents are available.
        """
        if not self.agents:
            raise ValueError("No agents available")

        candidates = [a for a in self.agents if a.id != exclude_agent_id]
        if not candidates:
            # If exclusion removes everyone, fall back to all agents
            candidates = self.agents

        chosen = random.choice(candidates)
        return chosen.id

    def get_last_publisher(self, stream_name: str) -> Optional[str]:
        """Return the sender_id of the most recent message in the stream, or None."""
        try:
            entries = self.stream_client.redis.xrevrange(stream_name, count=1)
            if entries:
                _, data = entries[0]
                sender = data.get("sender_id") if isinstance(data, dict) else None
                if isinstance(sender, bytes):
                    return sender.decode()
                return sender
        except Exception:
            pass
        return None

    def select_and_store_next_responder(self, exclude_agent_id: Optional[str] = None) -> str:
        """Select a random next responder, store it in Redis, and return the agent id.
        
        This should be called ONCE after a message is published to designate
        who responds next. All agents will then read this same value.
        """
        next_agent_id = self.get_random_agent(exclude_agent_id=exclude_agent_id)
        # Store in Redis so all agents see the same value
        self.stream_client.redis.set(NEXT_RESPONDER_KEY, next_agent_id)
        return next_agent_id

    def get_designated_responder(self) -> Optional[str]:
        """Get the currently designated next responder from Redis.
        
        Returns None if no responder is designated.
        """
        try:
            value = self.stream_client.redis.get(NEXT_RESPONDER_KEY)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode()
            return str(value)
        except Exception:
            return None

    def clear_designated_responder(self) -> None:
        """Clear the designated responder (called after an agent responds)."""
        try:
            self.stream_client.redis.delete(NEXT_RESPONDER_KEY)
        except Exception:
            pass

    def is_my_turn(self, agent_id: str) -> bool:
        """Check if the given agent is the designated next responder."""
        designated = self.get_designated_responder()
        return designated == agent_id

    # Legacy method - now just reads from Redis instead of computing
    def get_next_agent(self, stream_name: str) -> Optional[str]:
        """Get the designated next responder (reads from Redis, does not select new one)."""
        return self.get_designated_responder()