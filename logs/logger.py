import os
import asyncio
import queue
import time
import threading
from typing import Any

"""Asynchronous Logger that writes log messages to a file using a separate thread."""
class Logger:
    def __init__(self, num_agents: int):
        self.file_path = os.path.join(os.path.dirname(__file__), time.strftime(f"log_%Y%m%d-%H%M%S_{num_agents}.log"))
        self.log_queue = queue.Queue()
        self._stop_event = threading.Event()
        # start logging thread
        self._start_logging_thread()
    
    # Start ongoing therad in main process
    def _start_logging_thread(self):
        self._thread = threading.Thread(target=self._log, daemon=True, name="LoggerThread")
        self._thread.start()

    def _stop_logging_thread(self):
        self._stop_event.set()
        try:
            self.log_queue.put_nowait(None)  # Unblock the thread if waiting
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join()

    # Log with timestamp
    def _log(self):
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

    # Asynchronous methods to be called from async code
    async def async_put(self, item: Any) -> None:
        """Awaitable put: pushes item into the thread-safe queue using executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.log_queue.put, item)

    async def async_stop(self, timeout: float = 2.0) -> None:
        """Awaitable stop: signals the worker and waits for thread join."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.stop, timeout)

        


    
