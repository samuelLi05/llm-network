import asyncio
from typing import Optional, Sequence, Dict, Any

from agents.local_llm import HuggingFaceLLM


class LLMService:
    """Simple async queue for local LLM generation to avoid blocking event loop."""

    def __init__(self, llm: Optional[HuggingFaceLLM] = None):
        self.llm = llm or HuggingFaceLLM()
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._stop_event.set()
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def generate(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put({"type": "generate", "messages": messages, "kwargs": kwargs, "future": fut})
        return await fut

    async def score_label_logprob(self, messages: Sequence[Dict[str, str]], label: str, **kwargs: Any) -> dict:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(
            {"type": "score", "messages": messages, "label": label, "kwargs": kwargs, "future": fut}
        )
        return await fut

    async def _worker_loop(self):
        while not self._stop_event.is_set():
            job = await self._queue.get()
            try:
                if job.get("type") == "score":
                    result = await asyncio.to_thread(
                        self.llm.score_label_logprob,
                        job["messages"],
                        job["label"],
                        **job.get("kwargs", {}),
                    )
                else:
                    result = await asyncio.to_thread(
                        self.llm.generate,
                        job["messages"],
                        **job.get("kwargs", {}),
                    )
                job["future"].set_result(result)
            except Exception as exc:
                if not job["future"].done():
                    job["future"].set_exception(exc)
