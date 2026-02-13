import json
import time
import random
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
    score_model: str
    # Stored as float32 ndarray in memory for fast math.
    # When persisted to Redis, we serialize to a JSON-compatible list[float].
    vector: np.ndarray
    score_vector: np.ndarray
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
        self._id_to_index: dict[str, int] = {}

        # Cached matrix view for fast retrieval
        self._dirty_matrix: bool = True
        self._matrix: Optional[np.ndarray] = None  # shape: (N, D), float32
        self._stance: Optional[np.ndarray] = None  # shape: (N,), float32
        self._strength: Optional[np.ndarray] = None  # shape: (N,), float32
        self._topic_sim: Optional[np.ndarray] = None  # shape: (N,), float32
        self._sender_ids: Optional[list[Optional[str]]] = None

    def _mark_dirty(self) -> None:
        self._dirty_matrix = True

    def _rebuild_id_index(self) -> None:
        self._id_to_index = {item.id: i for i, item in enumerate(self._items)}

    def get_by_id(self, post_id: str) -> Optional[EmbeddedPost]:
        idx = self._id_to_index.get(str(post_id))
        if idx is None:
            return None
        if idx < 0 or idx >= len(self._items):
            return None
        return self._items[idx]

    def _ensure_matrix(self) -> None:
        if not self._dirty_matrix and self._matrix is not None:
            return

        if not self._items:
            self._matrix = None
            self._stance = None
            self._strength = None
            self._topic_sim = None
            self._sender_ids = None
            self._dirty_matrix = False
            return

        self._matrix = np.stack([item.vector for item in self._items]).astype(np.float32, copy=False)
        self._stance = np.asarray([item.stance_score for item in self._items], dtype=np.float32)
        self._strength = np.asarray([item.strength for item in self._items], dtype=np.float32)
        self._topic_sim = np.asarray([item.topic_similarity for item in self._items], dtype=np.float32)
        self._sender_ids = [
            (item.metadata.get("sender_id") if isinstance(item.metadata, dict) else None)
            for item in self._items
        ]
        self._dirty_matrix = False

    @staticmethod
    def _serialize_item(item: EmbeddedPost) -> dict[str, Any]:
        payload = asdict(item)
        vec = payload.get("vector")
        if isinstance(vec, np.ndarray):
            payload["vector"] = vec.astype(np.float32, copy=False).tolist()
        score_vec = payload.get("score_vector")
        if isinstance(score_vec, np.ndarray):
            payload["score_vector"] = score_vec.astype(np.float32, copy=False).tolist()
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
        score_vec = embedded.get("score_vector") or embedded.get("vector")
        item = EmbeddedPost(
            id=str(post_id),
            topic=self.topic,
            text=text,
            created_at=created_at,
            embedding_model=embedded["model"],
            score_model=str(embedded.get("score_model") or embedded.get("model") or ""),
            vector=to_np(embedded["vector"], dtype=np.float32),
            score_vector=to_np(score_vec, dtype=np.float32),
            topic_similarity=float(embedded["topic_similarity"]),
            stance_score=float(embedded["stance_score"]),
            strength=float(embedded["strength"]),
            anchor_group_similarities=dict(embedded.get("anchor_group_similarities", {})),
            metadata=dict(metadata),
        )

        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        self._rebuild_id_index()
        self._mark_dirty()

        if persist and self.redis_cache:
            await self.redis_cache.append_response(self.redis_key, self._serialize_item(item))

        return item

    async def add_scored_vector(
        self,
        *,
        id: str,
        text: str,
        vector: list[float] | np.ndarray,
        score_vector: Optional[list[float] | np.ndarray] = None,
        scored: dict[str, Any],
        created_at: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        persist: bool = True,
    ) -> EmbeddedPost:
        """Add a post using a precomputed embedding vector and score dict.

        Use this when integrating with streams: embed/score once, then index.
        Expects `scored` to contain: model, topic_similarity, stance_score, strength.
        """
        created_at = float(created_at if created_at is not None else time.time())
        metadata = metadata or {}
        score_vec = score_vector if score_vector is not None else scored.get("score_vector") or vector

        item = EmbeddedPost(
            id=str(id),
            topic=self.topic,
            text=text,
            created_at=created_at,
            embedding_model=str(scored.get("model") or getattr(self.analyzer, "semantic_embedding_model", "")),
            score_model=str(scored.get("score_model") or scored.get("model") or getattr(self.analyzer, "score_embedding_model", "")),
            vector=to_np(vector, dtype=np.float32),
            score_vector=to_np(score_vec, dtype=np.float32),
            topic_similarity=float(scored["topic_similarity"]),
            stance_score=float(scored["stance_score"]),
            strength=float(scored["strength"]),
            anchor_group_similarities=dict(scored.get("anchor_group_similarities", {})),
            metadata=dict(metadata),
        )

        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        self._rebuild_id_index()
        self._mark_dirty()

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
                if isinstance(data.get("score_vector"), list):
                    data["score_vector"] = to_np(data["score_vector"], dtype=np.float32)
                elif "score_vector" not in data:
                    data["score_vector"] = data.get("vector")
                if "score_model" not in data:
                    data["score_model"] = data.get("embedding_model") or ""
                items.append(EmbeddedPost(**data))
            except Exception:
                continue
        self._items = items
        self._rebuild_id_index()
        self._mark_dirty()
        return len(self._items)

    async def recommend(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        min_topic_similarity: float = 0.15,
        min_strength: float = 0.05,
        allowed_sender_ids: Optional[list[str]] = None,
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
            score_model=str(embedded.get("score_model") or embedded.get("model") or ""),
            vector=to_np(embedded["vector"], dtype=np.float32),
            score_vector=to_np(embedded.get("score_vector") or embedded["vector"], dtype=np.float32),
            topic_similarity=float(embedded["topic_similarity"]),
            stance_score=float(embedded["stance_score"]),
            strength=float(embedded["strength"]),
            anchor_group_similarities=dict(embedded.get("anchor_group_similarities", {})),
            metadata={},
        )

        if not self._items:
            return []

        self._ensure_matrix()
        if self._matrix is None or self._stance is None or self._strength is None or self._topic_sim is None:
            return []

        # Candidate mask
        mask = (self._topic_sim >= float(min_topic_similarity)) & (self._strength >= float(min_strength))

        if allowed_sender_ids is not None and self._sender_ids is not None:
            allowed = set(str(x) for x in allowed_sender_ids)
            # Always allow seed posts when present (useful for cold-start).
            allowed.add("__seed__")
            sender_mask = np.asarray(
                [
                    (sid is not None and str(sid) in allowed)
                    for sid in self._sender_ids
                ],
                dtype=bool,
            )
            mask = mask & sender_mask

        if exclude_sender_id is not None and self._sender_ids is not None:
            exclude_mask = np.asarray(
                [sid == exclude_sender_id for sid in self._sender_ids],
                dtype=bool,
            )
            mask = mask & (~exclude_mask)

        if not bool(mask.any()):
            return []

        qv = query.vector.astype(np.float32, copy=False)
        dots = self._matrix @ qv

        distances = (
            float(alpha) * np.abs(self._stance - float(query.stance_score))
            + float(beta) * np.abs(self._strength - float(query.strength))
            + float(gamma) * (1.0 - dots)
        )

        distances = np.where(mask, distances, np.inf)
        k = min(int(top_k), int(mask.sum()))
        idx = np.argpartition(distances, kth=k - 1)[:k]
        idx = idx[np.argsort(distances[idx])]

        results: list[dict[str, Any]] = []
        for i in idx.tolist():
            item = self._items[i]
            stance_delta = float(abs(float(self._stance[i]) - float(query.stance_score)))
            strength_delta = float(abs(float(self._strength[i]) - float(query.strength)))
            semantic_sim = float(dots[i])
            results.append(
                {
                    "distance": float(distances[i]),
                    "semantic_similarity": semantic_sim,
                    "stance_delta": stance_delta,
                    "strength_delta": strength_delta,
                    "stance_term": float(alpha) * stance_delta,
                    "strength_term": float(beta) * strength_delta,
                    "semantic_term": float(gamma) * (1.0 - semantic_sim),
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
        allowed_sender_ids: Optional[list[str]] = None,
        exclude_sender_id: Optional[str] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        diversity_prob: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Rank stored posts for an agent using the same composite distance.

        This avoids any embedding call at request-time.
        """
        query = EmbeddedPost(
            id="__agent__",
            topic=self.topic,
            text="",
            created_at=time.time(),
            embedding_model=str(getattr(self.analyzer, "semantic_embedding_model", "")),
            score_model=str(getattr(self.analyzer, "score_embedding_model", "")),
            vector=to_np(agent_vector, dtype=np.float32),
            score_vector=to_np(agent_vector, dtype=np.float32),
            topic_similarity=1.0,
            stance_score=float(agent_stance_score),
            strength=float(agent_strength),
            anchor_group_similarities={},
            metadata={},
        )

        if not self._items:
            return []

        self._ensure_matrix()
        if self._matrix is None or self._stance is None or self._strength is None or self._topic_sim is None:
            return []

        mask = (self._topic_sim >= float(min_topic_similarity)) & (self._strength >= float(min_strength))

        if allowed_sender_ids is not None and self._sender_ids is not None:
            allowed = set(str(x) for x in allowed_sender_ids)
            allowed.add("__seed__")
            sender_mask = np.asarray(
                [
                    (sid is not None and str(sid) in allowed)
                    for sid in self._sender_ids
                ],
                dtype=bool,
            )
            mask = mask & sender_mask

        if exclude_sender_id is not None and self._sender_ids is not None:
            exclude_mask = np.asarray(
                [sid == exclude_sender_id for sid in self._sender_ids],
                dtype=bool,
            )
            mask = mask & (~exclude_mask)

        if not bool(mask.any()):
            return []

        qv = query.vector.astype(np.float32, copy=False)
        dots = self._matrix @ qv

        k = min(int(top_k), int(mask.sum()))

        # For disagreement: get larger sample using semantic similarity only
        large_distances = (1.0 - dots)
        large_distances = np.where(mask, large_distances, np.inf)
        large_k = min(int(top_k) * 3, int(mask.sum()))
        large_idx = np.argpartition(large_distances, kth=large_k - 1)[:large_k]
        large_idx = large_idx[np.argsort(large_distances[large_idx])]

        # Filter for most disagreeing in strength and stance (top k by combined deltas)
        strength_deltas = np.abs(self._strength - float(query.strength))
        stance_deltas = np.abs(self._stance - float(query.stance_score))
        disagreement_candidates = sorted(large_idx, key=lambda i: float(alpha) * stance_deltas[i] + float(beta) * strength_deltas[i], reverse=True)[:k]
        disagreement_idx = disagreement_candidates

        distances = (
            float(alpha) * np.abs(self._stance - float(query.stance_score))
            + float(beta) * np.abs(self._strength - float(query.strength))
            + float(gamma) * (1.0 - dots)
        )
        distances = np.where(mask, distances, np.inf)

        idx = np.argpartition(distances, kth=k - 1)[:k]
        idx = idx[np.argsort(distances[idx])]

        agreement_distances = distances
        agreement_idx = idx

        # Selection with diversity
        selected = []  # list of (idx, mode)
        used_ids = set()
        for _ in range(k):
            if random.random() < diversity_prob:
                mode = "disagreement"
                idx_list = disagreement_idx
            else:
                mode = "agreement"
                idx_list = agreement_idx
            for j in range(len(idx_list)):
                idx = idx_list[j]
                if self._items[idx].id not in used_ids:
                    selected.append((idx, mode))
                    used_ids.add(self._items[idx].id)
                    break

        results: list[dict[str, Any]] = []
        for idx, mode in selected:
            item = self._items[idx]
            stance_delta = float(abs(float(self._stance[idx]) - float(query.stance_score)))
            strength_delta = float(abs(float(self._strength[idx]) - float(query.strength)))
            semantic_sim = float(dots[idx])
            if mode == "agreement":
                stance_term = float(alpha) * stance_delta
                strength_term = float(beta) * strength_delta
                semantic_term = float(gamma) * (1.0 - semantic_sim)
                distance = stance_term + strength_term + semantic_term
            else:
                stance_term = float(alpha) * stance_delta
                strength_term = float(beta) * strength_delta
                semantic_term = float(gamma) * (1.0 - semantic_sim)
                distance = stance_term + strength_term + semantic_term
            results.append(
                {
                    "distance": distance,
                    "semantic_similarity": semantic_sim,
                    "stance_delta": stance_delta,
                    "strength_delta": strength_delta,
                    "stance_term": stance_term,
                    "strength_term": strength_term,
                    "semantic_term": float(gamma) * (1.0 - semantic_sim),
                    "mode": mode,
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
