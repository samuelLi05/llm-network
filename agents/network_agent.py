import os
import asyncio
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv
from openai import OpenAI
from network.stream import RedisStream
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService

from controller.stance_analysis.agent_profile_store import AgentProfileStore
from controller.stance_analysis.rolling_embedding_store import RollingEmbeddingStore
from controller.stance_analysis.network_topology import NetworkTopologyTracker

from logs.logger import console_logger

if TYPE_CHECKING:
    from controller.time_manager import TimeManager
    from controller.order_manager import OrderManager
    from network.cache import RedisCache
    from logs.logger import Logger

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class NetworkAgent:
    """An agent that listens to a Redis stream, generates responses, and publishes back.

    Integrates with TimeManager for rate limiting, OrderManager for turn selection,
    RedisCache for message history, and Logger for publish event logging.
    """

    def __init__(
        self,
        id: str,
        init_prompt: str,
        topic: Optional[str] = None,
        stream_name: str,
        stream_group: str,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        seed: Optional[int] = None,
        time_manager: Optional["TimeManager"] = None,
        order_manager: Optional["OrderManager"] = None,
        message_cache: Optional["RedisCache"] = None,
        logger: Optional["Logger"] = None,
        local_llm: Optional[LocalLLM] = None,
        llm_service: Optional[LLMService] = None,
        rolling_store: Optional[RollingEmbeddingStore] = None,
        profile_store: Optional[AgentProfileStore] = None,
        topology_tracker: Optional[NetworkTopologyTracker] = None,
        analysis_lock: Optional[asyncio.Lock] = None,
        context_top_k: int = 8,
    ):
        self.id = id
        self.init_prompt = init_prompt
        self.topic = topic
        self.stream_client = RedisStream(host=redis_host, port=redis_port)
        self.stream_name = stream_name
        self.stream_group = stream_group
        self.seed = seed
        self.consumer_name = f"{self.stream_group}-consumer"
        self.is_consuming = asyncio.Event()

        # Controller integrations
        self.time_manager = time_manager
        self.order_manager = order_manager
        self.message_cache = message_cache
        self.logger = logger
        self.local_llm = local_llm
        self.llm_service = llm_service

        # Embedding-based context + profiles
        self.rolling_store = rolling_store
        self.profile_store = profile_store
        self.topology_tracker = topology_tracker
        self.analysis_lock = analysis_lock
        self.context_top_k = int(context_top_k)

        self._publish_lock = asyncio.Lock()
        self._consumer_task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize consumer group and start listening for messages."""
        await self.stream_client.create_consumer_group(self.stream_name, self.stream_group)

        # Ensure the agent has a profile seeded from its init prompt.
        if self.profile_store and self.topic:
            try:
                await self.profile_store.ensure_initialized(
                    self.id,
                    seed_text=self.init_prompt,
                    topic_for_embedding=self.topic,
                )
            except Exception as exc:
                console_logger.info(f"Agent {self.id} profile init failed (continuing): {exc}")

        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self.consume_stream())
        await self.is_consuming.wait()
        console_logger.info(f"Agent {self.id} started and is listening for messages.")

    async def stop(self):
        """Stop background consumption (useful for notebook reruns/cleanup)."""
        task = self._consumer_task
        self._consumer_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def consume_stream(self):
        """Consume messages from the Redis stream and process them."""
        console_logger.info(f"Agent {self.id} is consuming from stream '{self.stream_name}' as part of group '{self.stream_group}'.")
        self.is_consuming.set()
        async for message_id, message_data in self.stream_client.consume_messages(
            self.stream_name, self.stream_group, self.consumer_name
        ):
            sender_id = message_data.get('sender_id')
            if sender_id == self.id:
                # Ignore own messages
                # console_logger.info(f"Agent {self.id} ignored its own message {message_id}.")
                await asyncio.sleep(0.1)
                continue

            # console_logger.info(f"Agent {self.id} received message {message_id}: {message_data}")
            incoming_post = message_data.get('content', '')
            if not incoming_post:
                continue

            # Check if this agent is the designated responder (OrderManager logic)
            if self.order_manager:
                if not self.order_manager.is_my_turn(self.id):
                    # Not our turn, skip responding
                    designated = self.order_manager.get_designated_responder()
                    # console_logger.info(f"Agent {self.id} skipped (designated responder is {designated}).")
                    continue
                # Clear the designation since we're responding
                self.order_manager.clear_designated_responder()

            # Update this agent's profile with the consumed message (fast-path: use stored vector)
            try:
                await self._on_consumed_message(message_id, sender_id, incoming_post)
            except Exception as exc:
                console_logger.info(f"Agent {self.id} consumed-profile update failed (continuing): {exc}")

            # Generate and publish response
            wrapped_prompt = (
                "React to the post below in your own voice. " \
                "Do NOT copy or paraphrase it line-by-line. " \
                "Write a NEW, distinct social-media-style post that reflects your fixed stance and worldview, " \
                "and directly contradict ideas from other posts to strongly support your fixed stance. " \
                "Keep it attention-grabbing, concise, and assertive.\n" \
                "POST TO REACT TO:\n" \
                f"{incoming_post}"
            )
            response = await self.generate_response(wrapped_prompt)
            await self.publish_message(response)

    async def publish_message(self, message: str):
        """Publish a message to the Redis stream with rate limiting and logging."""
        # Wait for TimeManager permission if available
        if self.time_manager:
            async with self.time_manager.publish_lock(self.id):
                await self._do_publish(message)
        else:
            await self._do_publish(message)

    async def _do_publish(self, message: str):
        """Internal publish: designate next responder FIRST, then send to stream, cache, and log."""
        async with self._publish_lock:
            message_data = {'sender_id': self.id, 'content': message}

            # Designate the next responder BEFORE publishing the message
            if self.order_manager:
                next_responder = self.order_manager.select_and_store_next_responder(exclude_agent_id=self.id)
                console_logger.info(f"Agent {self.id} designating next responder: {next_responder}")

            # Now publish the message (other agents will see the designated responder)
            message_id = await self.stream_client.publish_message(self.stream_name, message_data)

        # Append to RedisCache (per-agent message history)
        if self.message_cache:
            await self.message_cache.append_response(self.id, message_data)

        # Index into rolling embedding store + update authored profile (off the critical path)
        if self.rolling_store and self.profile_store and self.topic:
            asyncio.create_task(self._on_published_message(str(message_id), message))

        # Log the publish event
        if self.logger:
            await self.logger.async_log_publish(self.id, message)

            console_logger.debug(f"Agent {self.id} published message.")

    async def generate_response(self, prompt: Optional[str] = None) -> str:
        """Generate a response using the OpenAI API."""
        console_logger.info(f"Agent {self.id} is generating response")
        try:
            last_message = (prompt or "").strip()
            if not last_message:
                # Helps avoid an empty user message on the first post.
                last_message = "Write your next post."

            # Build assistant context (recommended neighbors instead of last-N-per-agent)
            recent_messages = await self._build_context(last_message)
            # console_logger.info(f"Agent {self.id} recent messages for context: {recent_messages}")
            # console_logger.debug(f"Agent {self.id} Generating Response")
            if not self.local_llm and not self.llm_service:
                # Call OpenAI responses API in a thread and await with timeout
                response_obj = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.responses.create,
                        model="gpt-4o-mini",
                        input=[
                            {"role": "system", "content": self.init_prompt},
                            {"role": "user", "content": last_message},
                            {"role": "assistant", "content": recent_messages},
                        ],
                        temperature=0.7,
                        max_output_tokens=300,
                    ),
                    timeout=60,
                )
                # Extract the textual output
                response = getattr(response_obj, "output_text", None) or str(response_obj)
            else:
                messages = [
                    {"role": "system", "content": self.init_prompt},
                    {"role": "user", "content": last_message},
                    {"role": "assistant", "content": recent_messages},
                ]
                if self.llm_service:
                    response = await self.llm_service.generate(
                        messages,
                        max_new_tokens=300,
                        temperature=0.7,
                        stop=None,
                    )
                else:
                    # Run the local blocking generation in a thread so the asyncio loop isn't blocked
                    response = await asyncio.to_thread(
                        self.local_llm.generate,
                        messages,
                        max_new_tokens=300,
                        temperature=0.7,
                        stop=None,
                    )
                
            # console_logger.info(f"Agent {self.id} generated response: {response}")
            return response
        except asyncio.TimeoutError:
            console_logger.error(f"Timed out generating response for agent {self.id} (Redis/OpenAI timeout).")
            return "I'm sorry, I timed out while generating a response."
        except Exception as e:
            console_logger.error(f"Error generating response for agent {self.id}: {e}")
            return "I'm sorry, I couldn't process that."

    async def _build_context(self, last_message: str) -> str:
        """Build context used to condition generation.

        Primary path: use the agent's precomputed profile vector to retrieve
        relevant embedded messages from the rolling store.
        Fallback: original last-N messages from RedisCache.
        """
        if self.rolling_store and self.profile_store and self.topic:
            try:
                profile = await self.profile_store.load(self.id)
                if profile is not None and profile.vector is not None:
                    view = await self.profile_store.get_agent_topic_view(self.id, topic=self.topic)
                    if view is not None:
                        recos = await self.rolling_store.recommend_for_agent_vector(
                            agent_vector=profile.vector,
                            agent_stance_score=float(view.get("stance_score", 0.0)),
                            agent_strength=float(view.get("strength", 0.0)),
                            top_k=self.context_top_k,
                            exclude_sender_id=self.id,
                        )
                        if recos:
                            texts = [r.get("text", "") for r in recos if r.get("text")]
                            if texts:
                                return "\n".join(texts)
            except Exception as exc:
                console_logger.info(f"Agent {self.id} recommendation context failed (fallback): {exc}")

        # Fallback: original cache behavior
        if not self.message_cache:
            return ""

        agents_list = None
        if self.order_manager and getattr(self.order_manager, "agents", None):
            agents_list = self.order_manager.agents
        else:
            agents_list = [self]

        parts = []
        for a in agents_list:
            try:
                items = await self.message_cache.get_responses(a.id, last_n=5)
            except Exception:
                console_logger.error(f"Error fetching messages for agent {a.id} from cache.")
                items = []
            normalized = [str(item) for item in items]
            if normalized:
                parts.append(f"{a.id}: " + " | ".join(normalized))

        return "\n".join(parts)

    async def _on_published_message(self, message_id: str, content: str) -> None:
        if not (self.rolling_store and self.profile_store and self.topic):
            return

        lock = self.analysis_lock
        if lock is None:
            class _Noop:
                async def __aenter__(self):
                    return None
                async def __aexit__(self, exc_type, exc, tb):
                    return False
            lock = _Noop()  # type: ignore

        async with lock:
            if self.rolling_store.get_by_id(str(message_id)) is not None:
                return
            # Embed+score once, then index everywhere.
            embedded = await self.rolling_store.analyzer.embed_and_score(content, include_vector=True)
            if embedded is None or "vector" not in embedded:
                return

            await self.rolling_store.add_scored_vector(
                id=str(message_id),
                text=content,
                vector=embedded["vector"],
                scored=embedded,
                created_at=None,
                metadata={"sender_id": self.id, "message_id": str(message_id)},
                persist=True,
            )

            await self.profile_store.add_interaction_vector(
                self.id,
                vector=embedded["vector"],
                interaction_type="authored",
                ts=None,
                metadata={"message_id": str(message_id)},
            )

        # Topology update is optional and rate-limited.
        if self.topology_tracker and self.order_manager and getattr(self.order_manager, "agents", None):
            agent_ids = [a.id for a in self.order_manager.agents]
            asyncio.create_task(self.topology_tracker.maybe_update(agent_ids))

    async def _on_consumed_message(self, message_id: str, sender_id: Optional[str], content: str) -> None:
        if not (self.rolling_store and self.profile_store and self.topic):
            return
        if not message_id:
            return

        lock = self.analysis_lock
        if lock is None:
            class _Noop:
                async def __aenter__(self):
                    return None
                async def __aexit__(self, exc_type, exc, tb):
                    return False
            lock = _Noop()  # type: ignore

        async with lock:
            item = self.rolling_store.get_by_id(str(message_id))
            if item is None:
                # Rare race/cold start: embed+index so the profile update can still happen.
                embedded = await self.rolling_store.analyzer.embed_and_score(content, include_vector=True)
                if embedded is None or "vector" not in embedded:
                    return
                item = await self.rolling_store.add_scored_vector(
                    id=str(message_id),
                    text=content,
                    vector=embedded["vector"],
                    scored=embedded,
                    created_at=None,
                    metadata={"sender_id": sender_id, "message_id": str(message_id)},
                    persist=True,
                )

            await self.profile_store.add_interaction_vector(
                self.id,
                vector=item.vector,
                interaction_type="consumed",
                ts=None,
                metadata={"message_id": str(message_id), "sender_id": sender_id},
            )
        
