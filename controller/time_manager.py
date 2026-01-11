import time
import threading
import asyncio
from typing import Dict, Optional

"""Manages time intervals for agents to control publishing frequency.

Overarching rule: only one agent can publish at a time, and a global minimum
interval must elapse between any two publishes on the stream.
"""


class TimeManager:
    def __init__(
        self,
        global_interval: float = 3.0,
        default_interval: float = 3.0,
        intervals: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            global_interval: Minimum seconds between any two publishes (stream-wide).
            default_interval: Default per-agent interval if not specified.
            intervals: Optional per-agent interval overrides.
        """
        self.global_interval = global_interval
        self.default_interval = default_interval
        self.intervals: Dict[str, float] = intervals if intervals is not None else {}
        self.last_publish: Dict[str, float] = {}
        self._last_global_publish: float = 0.0
        self.lock = threading.RLock()

    def _time(self) -> float:
        return time.monotonic()

    def set_interval(self, agent_id: str, interval: float):
        with self.lock:
            self.intervals[agent_id] = interval

    def get_interval(self, agent_id: str) -> float:
        with self.lock:
            return self.intervals.get(agent_id, self.default_interval)

    def time_since_last_global_publish(self) -> float:
        """Return seconds since last publish by any agent."""
        with self.lock:
            return self._time() - self._last_global_publish

    def next_allowed_time(self) -> float:
        """Return seconds to wait before next publish is allowed (global)."""
        with self.lock:
            elapsed = self._time() - self._last_global_publish
            wait = self.global_interval - elapsed
            return max(0.0, wait)

    def can_publish(self) -> bool:
        """Check if enough time has passed globally to allow a new publish."""
        return self.time_since_last_global_publish() >= self.global_interval

    def record_publish(self, agent_id: str) -> None:
        """Record that an agent just published (updates global and per-agent times)."""
        now = self._time()
        with self.lock:
            self._last_global_publish = now
            self.last_publish[agent_id] = now

    async def wait_until_allowed(self) -> None:
        """Async wait until a publish is allowed."""
        while True:
            wait = self.next_allowed_time()
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    async def acquire(self) -> None:
        """Acquire permission to publish (waits if needed)."""
        await self.wait_until_allowed()

    # Context manager for controlled publishing
    class PublishLock:
        def __init__(self, time_manager: "TimeManager", agent_id: str):
            self.time_manager = time_manager
            self.agent_id = agent_id

        async def __aenter__(self):
            await self.time_manager.acquire()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.time_manager.record_publish(self.agent_id)

    def publish_lock(self, agent_id: str) -> "TimeManager.PublishLock":
        """Return an async context manager that waits for permission and records publish."""
        return TimeManager.PublishLock(self, agent_id)