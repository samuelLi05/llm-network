import time
import threading
import asyncio
from typing import Dict, Optional

class TimeManager:
    def __init__(self, default_interval: float = 1.0, intervals: Optional[Dict[str, float]] = None):
        self.default_interval = default_interval
        self.intervals: Dict[str, float] = intervals if intervals is not None else {}
        self.last_publish: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.time = time.monotonic()

    def set_interval(self, agent_id: str, interval:float):
        with self.lock:
            self.intervals[agent_id] = interval
    
    def get_interval(self, agent_id: str) -> float:
        with self.lock:
            return self.intervals.get(agent_id, self.default_interval)
        
    def next_allowed_time(self, agent_id: str) -> float:
        now = self._time()
        with self._lock:
            last = self._last_publish.get(agent_id, 0.0)
        interval = self.get_interval(agent_id)
        wait = (last + interval) - now
        return max(0.0, wait)
    
    def can_publish(self, agent_id: str) -> bool:
        now = self._time()
        with self.lock:
            last = self.last_publish.get(agent_id, 0.0)
        interval = self.get_interval(agent_id)
        return (now - last) >= interval
    
    async def acquire(self, agent_id: str):
        loop = asyncio.get_event_loop()

        while True:
            if self.can_publish(agent_id):
                return True
            sleep_time = self.next_allowed_time(agent_id)
            await asyncio.sleep(sleep_time)

class PublishLock:
    def __init__(self, time_manager: TimeManager, agent_id: str):
        self.time_manager = time_manager
        self.agent_id = agent_id
        pass