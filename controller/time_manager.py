import time
import threading
import asyncio
import random
import math
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
        poisson_mean: float = 5.0,
        time_unit_ms: int = 1000,
    ):
        """
        Args:
            global_interval: Minimum seconds between any two publishes (stream-wide).
            default_interval: Default per-agent interval if not specified.
            intervals: Optional per-agent interval overrides.
            poisson_mean: average number of messages in timeslice
            time_unit_ms: length of discrete time slice in ms for poisson process
        """
        self.global_interval = global_interval
        self.default_interval = default_interval
        self.intervals: Dict[str, float] = intervals if intervals is not None else {}
        self.last_publish: Dict[str, float] = {}
        self._last_global_publish: float = 0.0
        self.lock = threading.RLock()

        # "fixed" (wall-clock min interval) or "poisson" (simulated clock + exp waits).
        self.mode = "poisson"
        self.poisson_mean = float(poisson_mean)
        self.time_unit_ms = max(1, int(time_unit_ms))

        self._publish_mutex: Optional[asyncio.Lock] = None

        # Simulated clock state (poisson mode).
        self._run_start_wall = time.monotonic()
        self._sim_time_ms: int = 0
        self._sim_last_wall: float = self._run_start_wall
        self._sim_paused: bool = False
        self._next_allowed_sim_ms: int = 0

        # Dedicated RNG for inter-arrival sampling.
        self._rng = random.Random()

    def _time(self) -> float:
        return time.monotonic()

    def _advance_sim_time_locked(self) -> None:
        """Advance simulated time based on wall time (poisson mode only)."""
        if self.mode != "poisson":
            return
        if self._sim_paused:
            return
        now = self._time()
        dt = now - self._sim_last_wall
        if dt <= 0:
            self._sim_last_wall = now
            return
        self._sim_time_ms += int(dt * 1000.0)
        self._sim_last_wall = now

    def now_ms(self) -> int:
        """Current run time in milliseconds (simulated in poisson mode)."""
        with self.lock:
            if self.mode == "poisson":
                self._advance_sim_time_locked()
                return int(self._sim_time_ms)
            return int((self._time() - self._run_start_wall) * 1000.0)

    def now_s(self) -> float:
        return self.now_ms() / 1000.0

    def time_slice(self) -> int:
        """0-indexed discrete time slice for opinion dynamics."""
        return int(self.now_ms() // self.time_unit_ms)

    def time_info(self) -> dict:
        """Compact time metadata to include in logs."""
        t_ms = self.now_ms()
        return {
            "mode": str(self.mode),
            "t_ms": int(t_ms),
            "t_s": float(t_ms) / 1000.0,
            "time_slice": int(t_ms // self.time_unit_ms),
            "time_unit_ms": int(self.time_unit_ms),
        }

    def set_interval(self, agent_id: str, interval: float):
        with self.lock:
            self.intervals[agent_id] = interval

    def get_interval(self, agent_id: str) -> float:
        with self.lock:
            return self.intervals.get(agent_id, self.default_interval)

    def time_since_last_global_publish(self) -> float:
        """Return seconds since last publish by any agent."""
        with self.lock:
            if self.mode == "poisson":
                self._advance_sim_time_locked()
                return (float(self._sim_time_ms) / 1000.0) - float(self._last_global_publish)
            return self._time() - self._last_global_publish

    def next_allowed_time(self) -> float:
        """Return seconds to wait before next publish is allowed (global)."""
        with self.lock:
            if self.mode == "poisson":
                self._advance_sim_time_locked()
                remaining_ms = int(self._next_allowed_sim_ms) - int(self._sim_time_ms)
                return max(0.0, float(remaining_ms) / 1000.0)

            elapsed = self._time() - self._last_global_publish
            wait = self.global_interval - elapsed
            return max(0.0, wait)

    def can_publish(self) -> bool:
        """Check if enough time has passed globally to allow a new publish."""
        if self.mode == "poisson":
            return self.next_allowed_time() <= 0.0
        return self.time_since_last_global_publish() >= self.global_interval

    def _get_publish_mutex(self) -> asyncio.Lock:
        if self._publish_mutex is None:
            self._publish_mutex = asyncio.Lock()
        return self._publish_mutex

    def record_publish(self, agent_id: str) -> None:
        """Record that an agent just published (updates global and per-agent times)."""
        with self.lock:
            if self.mode == "poisson":
                self._advance_sim_time_locked()
                now_s = float(self._sim_time_ms) / 1000.0
                self._last_global_publish = now_s
                self.last_publish[agent_id] = now_s

                lam_per_slice = max(1e-9, float(self.poisson_mean))
                slice_s = max(1e-9, float(self.time_unit_ms) / 1000.0)
                rate_per_s = lam_per_slice / slice_s

                dt_s = float(self._rng.expovariate(rate_per_s))
                dt_ms = max(1, int(math.ceil(dt_s * 1000.0)))
                self._next_allowed_sim_ms = int(self._sim_time_ms + dt_ms)
                return

            now = self._time()
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

    def _pause_clock_for_publish(self) -> None:
        with self.lock:
            if self.mode != "poisson":
                return
            self._advance_sim_time_locked()
            self._sim_paused = True

    def _resume_clock_after_publish(self) -> None:
        with self.lock:
            if self.mode != "poisson":
                return
            # Resume from current wall time without advancing sim time while paused.
            self._sim_last_wall = self._time()
            self._sim_paused = False

    # Context manager for controlled publishing
    class PublishLock:
        def __init__(self, time_manager: "TimeManager", agent_id: str):
            self.time_manager = time_manager
            self.agent_id = agent_id

        async def __aenter__(self):
            mutex = self.time_manager._get_publish_mutex()
            while True:
                await self.time_manager.acquire()
                await mutex.acquire()
                if self.time_manager.can_publish():
                    break
                mutex.release()
                await asyncio.sleep(0)

            self.time_manager._pause_clock_for_publish()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            try:
                self.time_manager.record_publish(self.agent_id)
            finally:
                self.time_manager._resume_clock_after_publish()
                try:
                    self.time_manager._get_publish_mutex().release()
                except RuntimeError:
                    pass

    def publish_lock(self, agent_id: str) -> "TimeManager.PublishLock":
        """Return an async context manager that waits for permission and records publish."""
        return TimeManager.PublishLock(self, agent_id)