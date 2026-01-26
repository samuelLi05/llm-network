import json
import os
import queue
import threading
import time
from typing import Any, Optional


class TopologyLogger:
    """Threaded JSONL logger for topology snapshots.

    Writes one JSON object per line to logs/topology_logs/.
    Intended to make it easy to prove the recommender is driving context
    and to later model the network as a graph.
    """

    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            log_dir = os.path.join(os.path.dirname(__file__), "topology_logs")
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, time.strftime("topology_%Y%m%d-%H%M%S.jsonl"))

        self.file_path = file_path
        self._q: queue.Queue[Any] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="TopologyLoggerThread")
        self._thread.start()

    def _worker(self) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                try:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception:
                    # best-effort logging
                    pass

    def log_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._q.put(snapshot)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
