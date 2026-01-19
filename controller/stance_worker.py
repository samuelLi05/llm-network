import asyncio
import time
from typing import Optional, List, Dict, Any

from network.stream import RedisStream
from network.cache import RedisCache
from controller.stance_analyzer import StanceAnalyzer
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService


class StanceWorker:
    """Background worker that consumes messages and writes stance analysis results.

    - Uses Redis streams for inbound messages (non-blocking consumer loop).
    - Runs OpenAI/local stance classification per message.
    - Batches SBERT similarity every N messages or every T seconds.
    - Writes outputs to a separate Redis prefix (stance:*).
    """

    def __init__(
        self,
        topic: str,
        stream_name: str,
        group_name: str = "stance_group",
        consumer_name: str = "stance_consumer",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        batch_size: int = 10,
        batch_interval: float = 10.0,
        local_llm: Optional[LocalLLM] = None,
        llm_service: Optional[LLMService] = None,
        use_openai: bool = True,
    ):
        self.topic = topic
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.stream = RedisStream(host=redis_host, port=redis_port)
        self.stance_cache = RedisCache(host=redis_host, port=redis_port, prefix="stance:")

        self.analyzer = StanceAnalyzer(topic, local_llm=local_llm, llm_service=llm_service)
        self.use_openai = use_openai
        self.use_local = bool(local_llm or llm_service)

        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self._buffer: List[Dict[str, Any]] = []
        self._last_batch_time = time.monotonic()

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        await self.stream.create_consumer_group(self.stream_name, self.group_name)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.stance_cache.close()

    async def _run(self):
        async for message_id, message_data in self.stream.consume_messages(
            self.stream_name, self.group_name, self.consumer_name
        ):
            if self._stop_event.is_set():
                break

            sender_id = message_data.get("sender_id")
            content = message_data.get("content")
            if not content:
                continue

            result: Dict[str, Any] = {
                "message_id": message_id,
                "sender_id": sender_id,
                "content": content,
                "timestamp": time.time(),
            }

            # Per-message stance classification (OpenAI/local)
            if self.use_openai:
                try:
                    label, stance_score, chosen_prob = await self.analyzer.openai_get_log_prob_classification(content)
                    result["openai"] = {
                        "label": label,
                        "stance_score": stance_score,
                        "chosen_prob": chosen_prob,
                    }
                except Exception as exc:
                    result["openai_error"] = str(exc)

            if self.use_local:
                try:
                    local_res = await self.analyzer.local_llm_classification(content)
                    if local_res is not None:
                        label, stance_score, chosen_prob = local_res
                        result["local"] = {
                            "label": label,
                            "stance_score": stance_score,
                            "chosen_prob": chosen_prob,
                        }
                except Exception as exc:
                    result["local_error"] = str(exc)

            await self.stance_cache.append_response(f"message:{sender_id}", result)

            # Buffer for similarity batching
            self._buffer.append({"message_id": message_id, "content": content})
            now = time.monotonic()
            if len(self._buffer) >= self.batch_size or (now - self._last_batch_time) >= self.batch_interval:
                await self._flush_similarity_batch()
                self._last_batch_time = now

    async def _flush_similarity_batch(self):
        if not self._buffer:
            return
        texts = [b["content"] for b in self._buffer]
        message_ids = [b["message_id"] for b in self._buffer]

        try:
            # Offload SBERT similarity to a thread to avoid blocking
            matrix = await asyncio.to_thread(self.analyzer.find_similarity, texts)
            payload = {
                "message_ids": message_ids,
                "similarity_matrix": matrix,
                "timestamp": time.time(),
            }
            await self.stance_cache.append_response("similarity_batches", payload)
        except Exception as exc:
            await self.stance_cache.append_response(
                "similarity_batches",
                {"message_ids": message_ids, "error": str(exc), "timestamp": time.time()},
            )
        finally:
            self._buffer.clear()