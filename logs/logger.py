import os
import asyncio
import queue
import time
import logging
import threading
from typing import Any, Optional

"""Asynchronous Logger that writes agent publish logs to a file using a separate thread.

Console logging uses Python's logging module with DEBUG, WARNING, CRITICAL levels.
Agent publish events are written to timestamped log files under logs/network_logs/.
"""

# Configure console logger for non-publish messages
console_logger = logging.getLogger("llm_network")
console_logger.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
if not console_logger.handlers:
    console_logger.addHandler(_console_handler)


class Logger:
    """Thread-safe async logger for agent publish events."""

    def __init__(self, num_agents: int, log_dir: Optional[str] = None, agent_config_dir: Optional[str] = None, stance_config_dir: Optional[str] = None):
        # Ensure log directories exist
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "network_logs")
        os.makedirs(log_dir, exist_ok=True)

        if agent_config_dir is None:
            agent_config_dir = os.path.join(os.path.dirname(__file__), "agent_config_logs")
        os.makedirs(agent_config_dir, exist_ok=True)

        if stance_config_dir is None:
            stance_config_dir = os.path.join(os.path.dirname(__file__), "stance_logs")
        os.makedirs(stance_config_dir, exist_ok=True)

        self.file_path = os.path.join(
            log_dir, 
            time.strftime(f"log_%Y%m%d-%H%M%S_{num_agents}.log")
        )

        self.agent_config_path = os.path.join(
            agent_config_dir,
            time.strftime(f"log_%Y%m%d-%H%M%S_{num_agents}_agent_configs.log")
        )

        self.stance_config_path = os.path.join(
            stance_config_dir,
            time.strftime(f"log_%Y%m%d-%H%M%S_{num_agents}_stance_configs.log")
        )
        # Thread-safe queue for log items
        self.log_queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_logging_thread()

    def _start_logging_thread(self):
        """Start the background logging thread."""
        self._thread = threading.Thread(target=self._log_worker, daemon=True, name="LoggerThread")
        self._thread.start()

    def _log_worker(self):
        """Worker that writes queued items to the log file."""
        with open(self.file_path, 'a') as f:
            while not self._stop_event.is_set():
                try:
                    item = self.log_queue.get(timeout=1)
                    if item is None:
                        break
                except queue.Empty:
                    continue
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"[{timestamp}] {item}\n")
                f.flush()

    def log_publish(self, agent_id: str, message: str) -> None:
        """Synchronously queue an agent publish event for logging."""
        self.log_queue.put(f"PUBLISH agent={agent_id} | {message}")

    async def async_log_publish(self, agent_id: str, message: str) -> None:
        """Asynchronously queue an agent publish event for logging."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.log_publish, agent_id, message)

    async def async_put(self, item: Any) -> None:
        """Awaitable put: pushes item into the thread-safe queue using executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.log_queue.put, item)

    def log_agent_configs(self, configs: dict) -> None:
        """Log the initial agent configurations to a separate file."""
        with open(self.agent_config_path, 'w') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{timestamp}] Agent Configurations:\n")
            for agent_id, config in configs.items():
                f.write(f"Agent {agent_id}:\n{config}\n\n")

    def log_stance_configs(self, configs: dict[str, float], *, header: str = "Baseline Stance Weights") -> None:
        """Log per-agent stance weights (used in baseline runner / analysis)."""
        with open(self.stance_config_path, 'w') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{timestamp}] {header}:\n")
            for agent_id, weight in configs.items():
                f.write(f"Agent {agent_id}: weight={float(weight)}\n")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker thread to stop and wait for it to finish."""
        self._stop_event.set()
        try:
            self.log_queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    async def async_stop(self, timeout: float = 2.0) -> None:
        """Awaitable stop: signals the worker and waits for thread join."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.stop, timeout)

        


    
