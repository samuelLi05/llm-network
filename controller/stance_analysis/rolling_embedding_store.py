import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from network.cache import RedisCache
from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.vector_ops import dot_similarity_normalized, to_np

import numpy as np


@dataclass(frozen=True)
class EmbeddedPost:
    id: str
    topic: str
    text: str
    created_at: float
    embedding_model: str
    # Stored as float32 ndarray in memory for fast math.
    # When persisted to Redis, we serialize to a JSON-compatible list[float].
    vector: np.ndarray
    topic_similarity: float
    stance_score: float
    strength: float
    anchor_group_similarities: dict[str, float]
    metadata: dict[str, Any]


class RollingEmbeddingStore:
    """Stores embedded posts and supports recommendation-style nearest-neighbor queries.

    Supports in-memory storage, and optional persistence to Redis via RedisCache.

    Distance function (lower is more similar):

        d = alpha * |Δstance| + beta * |Δstrength| + gamma * (1 - cos(v1, v2))

    Where v1/v2 are normalized embedding vectors.
    """

    def __init__(
        self,
        topic: str,
        analyzer: EmbeddingAnalyzer,
        redis_cache: Optional[RedisCache] = None,
        redis_key: Optional[str] = None,
        max_items: int = 2000,
    ):
        self.topic = topic
        self.analyzer = analyzer
        self.redis_cache = redis_cache
        self.redis_key = redis_key or f"stance_embeddings:{topic}"
        self.max_items = max_items
        self._items: list[EmbeddedPost] = []

    @staticmethod
    def _serialize_item(item: EmbeddedPost) -> dict[str, Any]:
        payload = asdict(item)
        vec = payload.get("vector")
        if isinstance(vec, np.ndarray):
            payload["vector"] = vec.astype(np.float32, copy=False).tolist()
        return payload

    @staticmethod
    def _distance(
        query: EmbeddedPost,
        item: EmbeddedPost,
        *,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:
        stance_term = alpha * abs(query.stance_score - item.stance_score)
        strength_term = beta * abs(query.strength - item.strength)
        # Vectors are L2-normalized on ingest, so cosine == dot.
        semantic_term = gamma * (1.0 - dot_similarity_normalized(query.vector, item.vector))
        return stance_term + strength_term + semantic_term

    async def add(
        self,
        text: str,
        *,
        id: Optional[str] = None,
        created_at: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        persist: bool = True,
    ) -> EmbeddedPost:
        """Embed a post and add it to the rolling store."""
        created_at = float(created_at if created_at is not None else time.time())
        metadata = metadata or {}
        embedded = await self.analyzer.embed_and_score(text, include_vector=True)
        if embedded is None:
            raise RuntimeError("Failed to embed text")

        post_id = id or metadata.get("message_id") or f"{int(created_at * 1000)}"
        item = EmbeddedPost(
            id=str(post_id),
            topic=self.topic,
            text=text,
            created_at=created_at,
            embedding_model=embedded["model"],
            vector=to_np(embedded["vector"], dtype=np.float32),
            topic_similarity=float(embedded["topic_similarity"]),
            stance_score=float(embedded["stance_score"]),
            strength=float(embedded["strength"]),
            anchor_group_similarities=dict(embedded.get("anchor_group_similarities", {})),
            metadata=dict(metadata),
        )

        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

        if persist and self.redis_cache:
            await self.redis_cache.append_response(self.redis_key, self._serialize_item(item))

        return item

    async def load_from_redis(self, last_n: int = 500) -> int:
        """Loads the last N items from Redis into memory."""
        if not self.redis_cache:
            return 0
        raw_items = await self.redis_cache.get_responses(self.redis_key, last_n=last_n)
        items: list[EmbeddedPost] = []
        for raw in raw_items:
            try:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                if isinstance(data.get("vector"), list):
                    data["vector"] = to_np(data["vector"], dtype=np.float32)
                items.append(EmbeddedPost(**data))
            except Exception:
                continue
        self._items = items
        return len(self._items)

    async def recommend(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        min_topic_similarity: float = 0.15,
        min_strength: float = 0.05,
        exclude_sender_id: Optional[str] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest neighbors by the composite distance function."""
        embedded = await self.analyzer.embed_and_score(query_text, include_vector=True)
        if embedded is None:
            return []

        query = EmbeddedPost(
            id="__query__",
            topic=self.topic,
            text=query_text,
            created_at=time.time(),
            embedding_model=embedded["model"],
            vector=to_np(embedded["vector"], dtype=np.float32),
            topic_similarity=float(embedded["topic_similarity"]),
            stance_score=float(embedded["stance_score"]),
            strength=float(embedded["strength"]),
            anchor_group_similarities=dict(embedded.get("anchor_group_similarities", {})),
            metadata={},
        )

        scored: list[tuple[float, EmbeddedPost]] = []
        for item in self._items:
            if item.topic_similarity < min_topic_similarity or item.strength < min_strength:
                continue
            if exclude_sender_id is not None and item.metadata.get("sender_id") == exclude_sender_id:
                continue
            d = self._distance(query, item, alpha=alpha, beta=beta, gamma=gamma)
            scored.append((d, item))

        scored.sort(key=lambda x: x[0])
        results: list[dict[str, Any]] = []
        for d, item in scored[:top_k]:
            results.append(
                {
                    "distance": float(d),
                    "id": item.id,
                    "created_at": item.created_at,
                    "text": item.text,
                    "topic_similarity": item.topic_similarity,
                    "stance_score": item.stance_score,
                    "strength": item.strength,
                    "metadata": item.metadata,
                }
            )
        return results

    async def recommend_for_agent_vector(
        self,
        *,
        agent_vector: list[float],
        agent_stance_score: float,
        agent_strength: float,
        top_k: int = 10,
        min_topic_similarity: float = 0.15,
        min_strength: float = 0.05,
        exclude_sender_id: Optional[str] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Rank stored posts for an agent using the same composite distance.

        This avoids any embedding call at request-time.
        """
        query = EmbeddedPost(
            id="__agent__",
            topic=self.topic,
            text="",
            created_at=time.time(),
            embedding_model=self.analyzer.embedding_model,
            vector=to_np(agent_vector, dtype=np.float32),
            topic_similarity=1.0,
            stance_score=float(agent_stance_score),
            strength=float(agent_strength),
            anchor_group_similarities={},
            metadata={},
        )

        scored: list[tuple[float, EmbeddedPost]] = []
        for item in self._items:
            if item.topic_similarity < min_topic_similarity or item.strength < min_strength:
                continue
            if exclude_sender_id is not None and item.metadata.get("sender_id") == exclude_sender_id:
                continue
            d = self._distance(query, item, alpha=alpha, beta=beta, gamma=gamma)
            scored.append((d, item))

        scored.sort(key=lambda x: x[0])
        results: list[dict[str, Any]] = []
        for d, item in scored[:top_k]:
            results.append(
                {
                    "distance": float(d),
                    "id": item.id,
                    "created_at": item.created_at,
                    "text": item.text,
                    "topic_similarity": item.topic_similarity,
                    "stance_score": item.stance_score,
                    "strength": item.strength,
                    "metadata": item.metadata,
                }
            )
        return results
