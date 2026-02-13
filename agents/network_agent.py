import os
import asyncio
import difflib
import json
import re
import hashlib
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
        stream_name: Optional[str] = None,
        stream_group: Optional[str] = None,
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
        # IMPORTANT: consumer_name must be unique per consumer in a group. If multiple
        # agents share the same name, Redis treats them as the same consumer which
        # can lead to odd delivery/ordering behavior.
        self.consumer_name = f"{self.stream_group}-{self.id}-consumer" if self.stream_group else None
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

        # Validation/logging knobs
        # LOG_RECOMMENDATIONS=1 logs feed source + top-k ids/distances and key state.
        # LOG_RECO_DEBUG=1 adds additional per-step detail (can be noisy).
        self.log_recommendations = True
        self.log_reco_debug = False
        self.log_reco_max_items = 5

        # Anti-repetition guard (helps prevent agents from converging on the same meme-y template)
        # REGEN_ON_REPEAT=1 will retry generation if output is too similar to the agent's recent posts.
        self.regen_on_repeat = True
        self.regen_max_attempts = 2
        self.regen_similarity_threshold = 0.86
        self.regen_history_last_n = 6

        # Global anti-repetition guard (across ALL agents). This is critical when stream
        # processing is fully async and multiple agents can generate concurrently.
        self.global_regen_on_repeat = True
        self.global_regen_max_attempts = 3
        self.global_regen_similarity_threshold = 0.90
        self.global_regen_history_last_n = 40
        self.global_dedupe_ttl_s = int(os.getenv("GLOBAL_DEDUPE_TTL_S", "180"))
        self._pending_publish_fingerprint: Optional[str] = None

        self._authoritative_sentence = self._extract_authoritative_sentence(self.init_prompt)

        self._publish_lock = asyncio.Lock()
        self._consumer_task: Optional[asyncio.Task] = None

        # Last recommendation metadata used to build generation context.
        self._last_feed_meta: Optional[dict] = None

    @staticmethod
    def _extract_authoritative_sentence(init_prompt: str) -> Optional[str]:
        """Best-effort extraction of the agent's unique stance sentence.

        main.py currently embeds it in the init prompt like:
        "The sentence <X> is your fixed stance ..."
        """
        if not init_prompt:
            return None
        m = re.search(r"The sentence\s+(.*?)\s+is your fixed stance", init_prompt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        stance = (m.group(1) or "").strip()
        return stance or None

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[#@]\w+", "", text)  # strip hashtags/mentions
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_heading(text: str) -> str:
        """Extract the first line for stance/topic/strength scoring.

        This reduces noise from hashtags, long threads, and rhetorical tails.
        """
        t = (text or "").strip()
        if not t:
            return ""
        head = (t.splitlines()[0] or "").strip()
        return head or t

    async def _select_score_text_and_vec(self, text: str) -> tuple[str, Optional[list[float]]]:
        """Choose a robust score span for volatile social posts.

        Prefer an embedder-selected sentence/span when possible; fall back to
        the first line.
        """
        analyzer = getattr(self.rolling_store, "analyzer", None) if self.rolling_store else None
        if analyzer is None:
            head = self._extract_heading(text)
            return head, None

        try:
            mode = str(getattr(analyzer, "score_span_mode", "heading") or "heading").strip().lower()

            if mode == "full":
                # Score the entire post as-is.
                t = (text or "").strip()
                return (t or self._extract_heading(text)), None

            if mode == "weighted":
                agg = await analyzer.aggregate_score_vector(
                    text,
                    max_spans=10,
                    stance_weight=float(getattr(analyzer, "score_span_stance_weight", 0.25)),
                    temperature=float(getattr(analyzer, "score_span_temperature", 0.05)),
                    min_topic_similarity=float(getattr(analyzer, "score_span_min_topic_similarity", 0.10)),
                )
                if agg:
                    # Text is only used for metadata; the vector is what matters.
                    return self._extract_heading(text), agg

            if mode == "best_span":
                span, vec = await analyzer.select_best_score_span(
                    text,
                    max_spans=10,
                    stance_weight=float(getattr(analyzer, "score_span_stance_weight", 0.25)),
                )
                span = (span or "").strip()
                if span and vec:
                    return span, vec

            # Default: score the first line.
            head = self._extract_heading(text)
            return head, None
        except Exception:
            pass

        head = self._extract_heading(text)
        return head, None

    @classmethod
    def _similarity_ratio(cls, a: str, b: str) -> float:
        a_n = cls._normalize_for_similarity(a)
        b_n = cls._normalize_for_similarity(b)
        if not a_n or not b_n:
            return 0.0
        return difflib.SequenceMatcher(None, a_n, b_n).ratio()

    @staticmethod
    def _extract_cached_content(item) -> str:
        if item is None:
            return ""
        if isinstance(item, (bytes, bytearray)):
            try:
                item = item.decode("utf-8", errors="ignore")
            except Exception:
                item = str(item)
        if isinstance(item, dict):
            return str(item.get("content") or item.get("text") or item)
        # Common case: RedisCache stores JSON strings of dicts.
        if isinstance(item, str):
            s = item.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return str(parsed.get("content") or parsed.get("text") or parsed)
                except Exception:
                    pass
            return s
        return str(item)

    @classmethod
    def _fingerprint_text(cls, text: str) -> str:
        norm = cls._normalize_for_similarity(text)
        return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()

    async def _apply_global_repeat_guard(self, response: str, messages: list[dict]) -> str:
        """Ensure the candidate response isn't too similar to recent cross-agent posts.

        Uses two layers:
        1) Lexical similarity check against recent global cache entries.
        2) Atomic Redis fingerprint reservation to prevent concurrent identical publishes.
        """
        if not (self.global_regen_on_repeat and self.message_cache):
            return response

        # Always clear any previous pending reservation.
        self._pending_publish_fingerprint = None

        regen_rule = (
            "Rewrite your post to be substantially different from recent posts across the whole network. "
            "Change the opening line and overall structure. Avoid the same cadence, slogans, and line breaks. "
            "Do not reuse the same hashtags or call-to-action phrasing."
        )

        # Pull global recent history once per attempt (cheap) and compare.
        for attempt in range(self.global_regen_max_attempts + 1):
            try:
                recent_items = await self.message_cache.get_last_responses(self.global_regen_history_last_n)
            except Exception:
                recent_items = []

            recent_texts = [self._extract_cached_content(it) for it in (recent_items or [])]
            best = 0.0
            for prev in recent_texts:
                best = max(best, self._similarity_ratio(response, prev))

            # Reserve fingerprint to prevent simultaneous identical publishes.
            fp = self._fingerprint_text(response)
            reserved = True
            try:
                reserved = await self.message_cache.reserve_fingerprint(fp, ttl_s=self.global_dedupe_ttl_s)
            except Exception:
                reserved = True

            if best < self.global_regen_similarity_threshold and reserved:
                self._pending_publish_fingerprint = fp
                return response

            if self.log_recommendations:
                reason = []
                if best >= self.global_regen_similarity_threshold:
                    reason.append(f"similarity={best:.2f}")
                if not reserved:
                    reason.append("fingerprint_collision")
                reason_s = ",".join(reason) if reason else "unknown"
                console_logger.info(
                    f"Agent {self.id} global regen triggered: {reason_s} threshold={self.global_regen_similarity_threshold:.2f}"
                )

            # If we couldn't reserve (someone else published the same thing) or it's too similar, regenerate.
            if attempt >= self.global_regen_max_attempts:
                # Fail open: publish as-is if we exhaust attempts.
                return response

            regen_messages = messages + [{"role": "system", "content": regen_rule}]
            response = await self._generate_from_messages(regen_messages)

        return response

    async def _generate_from_messages(self, messages: list[dict]) -> str:
        if not self.local_llm and not self.llm_service:
            response_obj = await asyncio.wait_for(
                asyncio.to_thread(
                    client.responses.create,
                    model="gpt-4o-mini",
                    input=messages,
                    temperature=0.7,
                    max_output_tokens=300,
                ),
                timeout=60,
            )
            return getattr(response_obj, "output_text", None) or str(response_obj)

        if self.llm_service:
            return await self.llm_service.generate(
                messages,
                max_new_tokens=300,
                temperature=0.7,
                stop=None,
            )

        return await asyncio.to_thread(
            self.local_llm.generate,
            messages,
            max_new_tokens=300,
            temperature=0.7,
            stop=None,
        )

    async def start(self):
        """Initialize consumer group and start listening for messages."""
        await self.stream_client.create_consumer_group(self.stream_name, self.stream_group)

        # Ensure the agent has a profile seeded from its init prompt.
        if self.profile_store and self.topic:
            try:
                # Do not block startup on embedding/profile work.
                asyncio.create_task(
                    self.profile_store.ensure_initialized(
                        self.id,
                        seed_text=self.init_prompt,
                        topic_for_embedding=self.topic,
                    )
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

            # Neighbor-based routing: ignore messages from non-neighbors.
            if self.order_manager and hasattr(self.order_manager, "get_neighbors"):
                try:
                    neighbors = self.order_manager.get_neighbors(self.id)
                except Exception:
                    neighbors = None
                if neighbors:
                    if sender_id not in set(neighbors):
                        await asyncio.sleep(0.05)
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
                "Write your next social-media-style post reacting to the post below. "
                "Sound like a real person with a consistent worldview. "
                "Do not quote the post verbatim and do not paraphrase line-by-line. "
                "Make one clear claim, be punchy, and invite engagement.\n\n"
                "POST YOU'RE REACTING TO:\n"
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

            # Last-chance idempotency: if generate_response reserved a fingerprint, keep it.
            # If it didn't, attempt to reserve now (fail open if Redis is unavailable).
            if self.message_cache and self.global_regen_on_repeat:
                try:
                    fp = self._pending_publish_fingerprint or self._fingerprint_text(message)
                    ok = await self.message_cache.reserve_fingerprint(fp, ttl_s=self.global_dedupe_ttl_s)
                    if ok:
                        self._pending_publish_fingerprint = fp
                except Exception:
                    pass

            # Designate the next responder BEFORE publishing the message
            if self.order_manager:
                next_responder = await self.order_manager.select_and_store_next_responder(exclude_agent_id=self.id)
                console_logger.info(f"Agent {self.id} designating next responder: {next_responder}")

            # Now publish the message (other agents will see the designated responder)
            message_id = await self.stream_client.publish_message(self.stream_name, message_data)

        # Clear reservation marker after publish attempt.
        self._pending_publish_fingerprint = None

        # Append to RedisCache (per-agent message history)
        if self.message_cache:
            await self.message_cache.append_response(self.id, message_data)

        # Index into rolling embedding store + update authored profile (off the critical path)
        if self.rolling_store and self.profile_store and self.topic:
            asyncio.create_task(self._on_published_message(str(message_id), message))

        # Log the publish event
        if self.logger:
            log_message = message
            metrics: dict = {"sender_id": self.id, "message_id": str(message_id)}

            # Score the published message so logs include the same stance/topic features used elsewhere.
            if self.rolling_store and getattr(self.rolling_store, "analyzer", None):
                try:
                    score_text, score_vec = await self._select_score_text_and_vec(message)
                    scored = await asyncio.wait_for(
                        self.rolling_store.analyzer.embed_and_score(
                            message,
                            include_vector=False,
                            score_text=score_text,
                            precomputed_score_vector=score_vec,
                        ),
                        timeout=float(os.getenv("PUBLISH_SCORE_TIMEOUT_S", "10")),
                    )
                except Exception:
                    scored = None
                if scored:
                    metrics["published"] = {
                        "topic_similarity": float(scored.get("topic_similarity", 0.0)),
                        "stance_score": float(scored.get("stance_score", 0.0)),
                        "strength": float(scored.get("strength", 0.0)),
                        "anchor_group_similarities": scored.get("anchor_group_similarities", {}),
                        "model": scored.get("model"),
                    }

            # Include a compact summary of the last feed used for generation.
            fm = self._last_feed_meta if isinstance(self._last_feed_meta, dict) else None
            # Can log reccomendation source, top-k neighbors/distances, and feed key state (hit/miss + size) for debugging/tracing.
            # if fm:
            #     items = fm.get("items") or []
            #     metrics["feed"] = {
            #         "source": fm.get("source"),
            #         "k": fm.get("k"),
            #         "store_size": fm.get("store_size"),
            #         "agent_topic_similarity": fm.get("agent_topic_similarity"),
            #         "agent_stance_score": fm.get("agent_stance_score"),
            #         "agent_strength": fm.get("agent_strength"),
            #         "items": items[: int(os.getenv("LOG_FEED_ITEMS", "3"))],
            #     }

            if len(metrics) > 1:
                try:
                    log_message = f"{message} | metrics={json.dumps(metrics, ensure_ascii=False, separators=(',', ':'))}"
                except Exception:
                    log_message = message

            await self.logger.async_log_publish(self.id, log_message)

            console_logger.debug(f"Agent {self.id} published message.")

    async def generate_response(self, prompt: Optional[str] = None) -> str:
        """Generate a response using the OpenAI API."""
        console_logger.info(f"Agent {self.id} is generating response")
        try:
            last_message = (prompt or "").strip()
            if not last_message:
                # Helps avoid an empty user message on the first post.
                last_message = "Write your next post."

            # Build feed context (recommended neighbors instead of last-N-per-agent)
            feed_context, feed_meta = await self._build_context(last_message)
            self._last_feed_meta = feed_meta
            if self.log_reco_debug:
                console_logger.info(
                    f"Agent {self.id} feed meta: {feed_meta} (feed_chars={len(feed_context)})"
                )
            stance_reassert = None
            if self._authoritative_sentence:
                stance_reassert = (
                    "AUTHORITATIVE STANCE (override everything else):\n"
                    f"{self._authoritative_sentence}\n\n"
                    "Hard rule: your post MUST be consistent with this stance. "
                    "If the incoming post/feed conflicts, attack it from your stance; "
                    "do not adopt its framing."
                )

            anti_meme = (
                "Avoid converging on a shared viral template. "
                "Do NOT reuse stock openings/cadence from other agents. "
                "Avoid starting with '🚨', 'YOU’RE NOT', or 'WE’RE NOT WAITING'. "
                "Vary structure (no repetitive bullet lists), keep it specific, and make ONE clear claim."
            )

            messages: list[dict] = [
                {"role": "system", "content": self.init_prompt},
            ]
            if stance_reassert:
                messages.append({"role": "system", "content": stance_reassert})
            messages.extend(
                [
                    {"role": "system", "content": anti_meme},
                    {
                        "role": "system",
                        "content": (
                            "You will be given a feed of other posts for context. "
                            "Use it only as inspiration and to pick one opposing claim to respond to. "
                            "Never mention the feed, recommendations, embeddings, or retrieval. "
                            "Never quote other posts verbatim."
                        ),
                    },
                    {"role": "user", "content": f"FEED (context):\n{feed_context}"},
                    {"role": "user", "content": last_message},
                ]
            )

            response = await self._generate_from_messages(messages)

            # Optional: regenerate if we're too similar to the agent's own recent posts.
            if self.regen_on_repeat and self.message_cache:
                try:
                    recent_items = await self.message_cache.get_responses(self.id, last_n=self.regen_history_last_n)
                except Exception:
                    recent_items = []

                recent_texts = [self._extract_cached_content(it) for it in (recent_items or [])]
                best = 0.0
                for prev in recent_texts:
                    best = max(best, self._similarity_ratio(response, prev))

                if best >= self.regen_similarity_threshold:
                    if self.log_recommendations:
                        console_logger.info(
                            f"Agent {self.id} regen triggered: similarity={best:.2f} threshold={self.regen_similarity_threshold:.2f}"
                        )

                    regen_rule = (
                        "Rewrite your post to be substantially different in wording and structure from your recent posts. "
                        "Change the opening line. Use different phrasing and rhythm. "
                        "Do NOT use '🚨', '👇', or the phrases 'YOU’RE NOT CHOOSING' / 'WE’RE NOT WAITING'."
                    )
                    for attempt in range(self.regen_max_attempts):
                        regen_messages = messages + [{"role": "system", "content": regen_rule}]
                        candidate = await self._generate_from_messages(regen_messages)
                        cand_best = 0.0
                        for prev in recent_texts:
                            cand_best = max(cand_best, self._similarity_ratio(candidate, prev))
                        if cand_best < best and cand_best < self.regen_similarity_threshold:
                            response = candidate
                            break
                        best = min(best, cand_best) if cand_best < best else cand_best

            # Global guard: prevent cross-agent convergence on identical templates when
            # stream processing is fully async.
            response = await self._apply_global_repeat_guard(response, messages)
            return response
            
        except asyncio.TimeoutError:
            console_logger.error(f"Timed out generating response for agent {self.id} (Redis/OpenAI timeout).")
            return "I'm sorry, I timed out while generating a response."
        except Exception as e:
            console_logger.error(f"Error generating response for agent {self.id}: {e}")
            return "I'm sorry, I couldn't process that."

    async def _build_context(self, last_message: str) -> tuple[str, dict]:
        """Build context used to condition generation.

        Primary path: use the agent's precomputed profile vector to retrieve
        relevant embedded messages from the rolling store.
        Fallback: original last-N messages from RedisCache.
        """
        if not (self.rolling_store and self.profile_store and self.topic):
            meta = {
                "source": "cache",
                "reason": "reco_disabled",
                "has_rolling_store": bool(self.rolling_store),
                "has_profile_store": bool(self.profile_store),
                "has_topic": bool(self.topic),
            }
            if self.log_recommendations:
                console_logger.info(f"Agent {self.id} reco disabled -> fallback: {meta}")
        else:
            try:
                profile = await self.profile_store.load(self.id)
                if profile is None:
                    if self.log_recommendations:
                        console_logger.info(f"Agent {self.id} reco fallback: profile_missing")
                elif profile.vector is None:
                    if self.log_recommendations:
                        console_logger.info(f"Agent {self.id} reco fallback: profile_vector_missing")
                else:
                    view = await self.profile_store.get_agent_topic_view(self.id, topic=self.topic)
                    if view is None:
                        if self.log_recommendations:
                            console_logger.info(f"Agent {self.id} reco fallback: topic_view_missing")
                    else:
                        store_size = None
                        try:
                            store_size = len(getattr(self.rolling_store, "_items", []) or [])
                        except Exception:
                            store_size = None

                        if self.log_reco_debug:
                            console_logger.info(
                                f"Agent {self.id} reco query: store_size={store_size} top_k={self.context_top_k}"
                            )

                        allowed_sender_ids = None
                        if self.order_manager and hasattr(self.order_manager, "get_neighbors"):
                            try:
                                neigh = self.order_manager.get_neighbors(self.id)
                            except Exception:
                                neigh = None
                            if neigh:
                                allowed_sender_ids = list(neigh)

                        recos = await self.rolling_store.recommend_for_agent_vector(
                            agent_vector=profile.vector,
                            agent_stance_score=float(view.get("stance_score", 0.0)),
                            agent_strength=float(view.get("strength", 0.0)),
                            top_k=self.context_top_k,
                            exclude_sender_id=self.id,
                            allowed_sender_ids=allowed_sender_ids,
                        )

                        if not recos:
                            if self.log_recommendations:
                                console_logger.info(
                                    f"Agent {self.id} reco fallback: no_recos (store_size={store_size})"
                                )
                        else:
                            texts = [r.get("text", "") for r in recos if r.get("text")]
                            if not texts:
                                if self.log_recommendations:
                                    console_logger.info(
                                        f"Agent {self.id} reco fallback: empty_texts (reco_count={len(recos)})"
                                    )
                            else:
                                meta = {
                                    "source": "reco",
                                    "k": len(texts),
                                    "store_size": store_size,
                                    "agent_updated_at": float(getattr(profile, "updated_at", 0.0) or 0.0),
                                    "agent_stance_score": float(view.get("stance_score", 0.0)),
                                    "agent_strength": float(view.get("strength", 0.0)),
                                    "agent_topic_similarity": float(view.get("topic_similarity", 0.0)),
                                    "items": [
                                        {
                                            "id": r.get("id"),
                                            "distance": r.get("distance"),
                                            "semantic_similarity": r.get("semantic_similarity"),
                                            "semantic_term": r.get("semantic_term"),
                                            "stance_delta": r.get("stance_delta"),
                                            "stance_term": r.get("stance_term"),
                                            "strength_delta": r.get("strength_delta"),
                                            "strength_term": r.get("strength_term"),
                                            "sender_id": (r.get("metadata") or {}).get("sender_id"),
                                        }
                                        for r in recos[: max(1, self.log_reco_max_items)]
                                    ],
                                }

                                # Debug/validation: ensure neighbor filtering is actually applied.
                                if allowed_sender_ids is not None:
                                    allowed = set(allowed_sender_ids)
                                    violations = [
                                        it.get("sender_id")
                                        for it in meta.get("items", [])
                                        if it.get("sender_id") not in allowed and it.get("sender_id") != "__seed__"
                                    ]
                                    if violations:
                                        console_logger.warning(
                                            f"Agent {self.id} reco neighbor violation: {violations} (neighbors={len(allowed)})"
                                        )
                                    meta["neighbor_filter"] = {
                                        "enabled": True,
                                        "neighbors": len(allowed),
                                        "violations": violations,
                                    }
                                if self.log_recommendations:
                                    console_logger.info(f"Agent {self.id} using reco feed: {meta}")
                                return "\n".join(texts), meta
            except Exception as exc:
                console_logger.info(f"Agent {self.id} recommendation context failed (fallback): {exc}")

        # Fallback: original cache behavior, get last N messages per each agent
        if not self.message_cache:
            meta = {"source": "empty", "reason": "no_message_cache"}
            if self.log_recommendations:
                console_logger.info(f"Agent {self.id} context empty: {meta}")
            return "", meta

        agents_list = None
        if self.order_manager and getattr(self.order_manager, "agents", None):
            agents_list = self.order_manager.agents
            if hasattr(self.order_manager, "get_neighbors"):
                try:
                    neigh = self.order_manager.get_neighbors(self.id)
                except Exception:
                    neigh = None
                if neigh:
                    allowed = set(neigh)
                    agents_list = [a for a in agents_list if a.id in allowed]
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

        meta = {"source": "cache", "agents": len(agents_list), "per_agent_last_n": 5}
        if self.log_recommendations:
            console_logger.info(f"Agent {self.id} using cache feed: {meta}")
        return "\n".join(parts), meta

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
                if self.log_reco_debug:
                    console_logger.info(f"Agent {self.id} ingest skip (already indexed): {message_id}")
                return
            # Embed+score once, then index everywhere.
            if self.log_reco_debug:
                console_logger.info(f"Agent {self.id} ingest start: message_id={message_id}")
            score_text, score_vec = await self._select_score_text_and_vec(content)
            embedded = await self.rolling_store.analyzer.embed_and_score(
                content,
                include_vector=True,
                semantic_text=content,
                score_text=score_text,
                precomputed_score_vector=score_vec,
            )
            if embedded is None or "vector" not in embedded:
                console_logger.info(f"Agent {self.id} ingest failed: embed_and_score returned None")
                return

            await self.rolling_store.add_scored_vector(
                id=str(message_id),
                text=content,
                vector=embedded["vector"],
                score_vector=embedded.get("score_vector"),
                scored=embedded,
                created_at=None,
                metadata={"sender_id": self.id, "message_id": str(message_id)},
                persist=True,
            )

            await self.profile_store.add_interaction_vector(
                self.id,
                vector=embedded["vector"],
                score_vector=embedded.get("score_vector"),
                interaction_type="authored",
                ts=None,
                metadata={"message_id": str(message_id)},
            )

            if self.log_recommendations:
                try:
                    store_size = len(getattr(self.rolling_store, "_items", []) or [])
                except Exception:
                    store_size = None
                console_logger.info(
                    f"Agent {self.id} ingest ok: message_id={message_id} store_size={store_size} "
                    f"stance={float(embedded.get('stance_score', 0.0)):.3f} strength={float(embedded.get('strength', 0.0)):.3f} "
                    f"topic_sim={float(embedded.get('topic_similarity', 0.0)):.3f}"
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
                if self.log_reco_debug:
                    console_logger.info(
                        f"Agent {self.id} consume cold-index: message_id={message_id} sender_id={sender_id}"
                    )
                score_text, score_vec = await self._select_score_text_and_vec(content)
                embedded = await self.rolling_store.analyzer.embed_and_score(
                    content,
                    include_vector=True,
                    semantic_text=content,
                    score_text=score_text,
                    precomputed_score_vector=score_vec,
                )
                if embedded is None or "vector" not in embedded:
                    console_logger.info(f"Agent {self.id} consume failed: embed_and_score returned None")
                    return
                item = await self.rolling_store.add_scored_vector(
                    id=str(message_id),
                    text=content,
                    vector=embedded["vector"],
                    score_vector=embedded.get("score_vector"),
                    scored=embedded,
                    created_at=None,
                    metadata={"sender_id": sender_id, "message_id": str(message_id)},
                    persist=True,
                )

            await self.profile_store.add_interaction_vector(
                self.id,
                vector=item.vector,
                score_vector=getattr(item, "score_vector", None),
                interaction_type="consumed",
                ts=None,
                metadata={"message_id": str(message_id), "sender_id": sender_id},
            )

            if self.log_reco_debug:
                console_logger.info(
                    f"Agent {self.id} consume ok: message_id={message_id} sender_id={sender_id}"
                )
        
