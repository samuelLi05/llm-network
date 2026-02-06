import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from redis.asyncio import Redis as AsyncRedis

from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.vector_ops import add_scaled_np, l2_normalize_np, sub_scaled_np, to_np

import numpy as np


@dataclass
class AgentProfile:
    agent_id: str
    embedding_model: str
    window_size: int
    seed_weight: float
    authored_weight: float
    consumed_weight: float
    created_at: float
    updated_at: float

    # Normalized mean vector representing the agent.
    vector: list[float]

    # Sliding window state (for exact window updates)
    total_weight: float
    sum_vector: list[float]
    window: list[dict[str, Any]]  # each: {"vector": [...], "weight": float, "type": str, "ts": float}

    # --- Optional / backwards-compatible fields (must come after non-defaults) ---
    score_embedding_model: str = ""

    # Normalized mean vector in the scoring model space (used for stance/topic/strength scoring).
    score_vector: Optional[list[float]] = None
    score_sum_vector: Optional[list[float]] = None

    seed_vector: Optional[list[float]] = None
    score_seed_vector: Optional[list[float]] = None


class AgentProfileStore:
    """Maintains a rolling (sliding-window) embedding profile per agent.

    The agent vector is a normalized weighted mean of:
      - a fixed seed embedding derived from the agent's system prompt (weighted heavily)
      - the last N interactions (authored + consumed), each embedded once

    This lets you precompute agent stance and do feed ranking using only dot products.
    """

    def __init__(
        self,
        redis: Optional[AsyncRedis] = None,
        *,
        window_size: int = 200,
        seed_weight: float = 5.0,
        authored_weight: float = 1.0,
        consumed_weight: float = 0.7,
        key_prefix: str = "agent_profile:",
        use_local_embedding_model: bool = False,
        use_baseline_statement: bool = False,
        openai_embedding_model: str = "text-embedding-3-small",
        local_embedding_model: str = "all-mpnet-base-v2",
    ):
        self.redis = redis
        self.window_size = window_size
        self.seed_weight = seed_weight
        self.authored_weight = authored_weight
        self.consumed_weight = consumed_weight
        self.key_prefix = key_prefix
        self._mem: dict[str, AgentProfile] = {}

        # Embedding backend configuration (controlled by main.py)
        self.use_local_embedding_model = bool(use_local_embedding_model)
        self.use_baseline_statement = bool(use_baseline_statement)
        self.openai_embedding_model = str(openai_embedding_model)
        self.local_embedding_model = str(local_embedding_model)

    def _make_analyzer(self, topic: str) -> EmbeddingAnalyzer:
        return EmbeddingAnalyzer(
            topic,
            use_local_embedding_model=self.use_local_embedding_model,
            use_baseline_statement=self.use_baseline_statement,
            openai_embedding_model=self.openai_embedding_model,
            local_embedding_model=self.local_embedding_model,
        )

    def _key(self, agent_id: str) -> str:
        return f"{self.key_prefix}{agent_id}"

    @staticmethod
    def _zeros(dim: int) -> list[float]:
        return [0.0] * dim

    async def load(self, agent_id: str) -> Optional[AgentProfile]:
        if self.redis is None:
            return self._mem.get(agent_id)

        raw = await self.redis.get(self._key(agent_id))
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return AgentProfile(**data)
        except Exception:
            return None

    async def save(self, profile: AgentProfile) -> None:
        if self.redis is None:
            self._mem[profile.agent_id] = profile
            return

        # Convert any ndarrays to lists for JSON.
        payload = asdict(profile)
        for k in (
            "vector",
            "sum_vector",
            "seed_vector",
            "score_vector",
            "score_sum_vector",
            "score_seed_vector",
        ):
            v = payload.get(k)
            if isinstance(v, np.ndarray):
                payload[k] = v.astype(np.float32, copy=False).tolist()
        window = payload.get("window") or []
        for item in window:
            vec = item.get("vector")
            if isinstance(vec, np.ndarray):
                item["vector"] = vec.astype(np.float32, copy=False).tolist()
            svec = item.get("score_vector")
            if isinstance(svec, np.ndarray):
                item["score_vector"] = svec.astype(np.float32, copy=False).tolist()

        await self.redis.set(self._key(profile.agent_id), json.dumps(payload, ensure_ascii=False))

    async def ensure_initialized(
        self,
        agent_id: str,
        *,
        seed_text: str,
        topic_for_embedding: str,
    ) -> AgentProfile:
        """Initialize a profile if missing.

        We use EmbeddingAnalyzer(topic_for_embedding) only to access the configured
        embedding model and embed the seed_text once.
        """
        existing = await self.load(agent_id)
        if existing is not None:
            return existing

        analyzer = self._make_analyzer(topic_for_embedding)
        seed = await analyzer.embed_and_score(seed_text, include_vector=True)
        if seed is None or "vector" not in seed:
            raise RuntimeError("Failed to embed seed_text")

        seed_vec = seed["vector"]
        seed_vec_np = to_np(seed_vec, dtype=np.float32)
        seed_score_vec = seed.get("score_vector") or seed_vec
        seed_score_vec_np = to_np(seed_score_vec, dtype=np.float32)
        dim = len(seed_vec)

        created = time.time()
        profile = AgentProfile(
            agent_id=agent_id,
            embedding_model=seed["model"],
            score_embedding_model=str(seed.get("score_model") or seed.get("model") or ""),
            window_size=self.window_size,
            seed_weight=self.seed_weight,
            authored_weight=self.authored_weight,
            consumed_weight=self.consumed_weight,
            created_at=created,
            updated_at=created,
            vector=seed_vec_np,
            score_vector=seed_score_vec_np,
            total_weight=0.0,
            sum_vector=to_np(self._zeros(dim), dtype=np.float32),
            score_sum_vector=to_np(self._zeros(len(seed_score_vec_np)), dtype=np.float32),
            window=[],
            seed_vector=seed_vec_np,
            score_seed_vector=seed_score_vec_np,
        )
        # Ensure the stored vector is normalized
        profile.vector = l2_normalize_np(profile.vector)
        if profile.score_vector is not None:
            profile.score_vector = l2_normalize_np(profile.score_vector)
        await self.save(profile)
        return profile

    async def add_interaction_vector(
        self,
        agent_id: str,
        *,
        vector: list[float] | np.ndarray,
        score_vector: Optional[list[float] | np.ndarray] = None,
        interaction_type: str,
        ts: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentProfile:
        """Add an already-embedded interaction vector to the agent profile window.

        This is the fast-path for stream integration: embed once on publish, then
        update authored/consumed profiles without re-embedding.
        """
        ts = float(ts if ts is not None else time.time())
        metadata = metadata or {}

        profile = await self.load(agent_id)
        if profile is None:
            raise RuntimeError("Profile missing; call ensure_initialized() first")

        if interaction_type not in {"authored", "consumed"}:
            raise ValueError("interaction_type must be 'authored' or 'consumed'")

        weight = profile.authored_weight if interaction_type == "authored" else profile.consumed_weight
        vec_np = to_np(vector, dtype=np.float32)
        score_vec_np = to_np(score_vector if score_vector is not None else vector, dtype=np.float32)

        # Initialize sum_vector if needed
        if profile.sum_vector is None or (isinstance(profile.sum_vector, list) and len(profile.sum_vector) == 0):
            profile.sum_vector = to_np(self._zeros(len(vec_np)), dtype=np.float32)
        if profile.score_sum_vector is None or (isinstance(profile.score_sum_vector, list) and len(profile.score_sum_vector) == 0):
            profile.score_sum_vector = to_np(self._zeros(len(score_vec_np)), dtype=np.float32)

        # Append to window
        profile.window.append(
            {
                "vector": vec_np,
                "score_vector": score_vec_np,
                "weight": float(weight),
                "type": interaction_type,
                "ts": ts,
                "meta": metadata,
            }
        )
        profile.sum_vector = add_scaled_np(profile.sum_vector, vec_np, float(weight))
        profile.score_sum_vector = add_scaled_np(profile.score_sum_vector, score_vec_np, float(weight))
        profile.total_weight += float(weight)

        # Enforce sliding window
        while len(profile.window) > profile.window_size:
            old = profile.window.pop(0)
            old_vec = old.get("vector")
            old_score_vec = old.get("score_vector")
            old_w = float(old.get("weight", 1.0))
            if old_vec is not None:
                profile.sum_vector = sub_scaled_np(profile.sum_vector, old_vec, old_w)
            if old_score_vec is not None and profile.score_sum_vector is not None:
                profile.score_sum_vector = sub_scaled_np(profile.score_sum_vector, old_score_vec, old_w)
                profile.total_weight -= old_w

        # Compute final agent vector = normalize(seed_weight*seed_vec + sum_vector)
        combined = to_np(profile.sum_vector, dtype=np.float32)
        if profile.seed_vector is not None and profile.seed_weight > 0:
            combined = add_scaled_np(combined, profile.seed_vector, float(profile.seed_weight))

        profile.vector = l2_normalize_np(combined)

        score_combined = to_np(profile.score_sum_vector if profile.score_sum_vector is not None else profile.sum_vector, dtype=np.float32)
        if profile.score_seed_vector is not None and profile.seed_weight > 0:
            score_combined = add_scaled_np(score_combined, profile.score_seed_vector, float(profile.seed_weight))
        profile.score_vector = l2_normalize_np(score_combined)
        profile.updated_at = time.time()

        await self.save(profile)
        return profile

    async def add_interaction(
        self,
        agent_id: str,
        *,
        text: str,
        interaction_type: str,
        topic: str,
        ts: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentProfile:
        """Add an interaction (authored/consumed) to the agent profile window."""
        ts = float(ts if ts is not None else time.time())
        metadata = metadata or {}

        profile = await self.load(agent_id)
        if profile is None:
            raise RuntimeError("Profile missing; call ensure_initialized() first")

        if interaction_type not in {"authored", "consumed"}:
            raise ValueError("interaction_type must be 'authored' or 'consumed'")

        weight = profile.authored_weight if interaction_type == "authored" else profile.consumed_weight

        analyzer = self._make_analyzer(topic)
        embedded = await analyzer.embed_and_score(text, include_vector=True)
        if embedded is None or "vector" not in embedded:
            raise RuntimeError("Failed to embed interaction")

        return await self.add_interaction_vector(
            agent_id,
            vector=embedded["vector"],
            score_vector=embedded.get("score_vector"),
            interaction_type=interaction_type,
            ts=ts,
            metadata=metadata,
        )

    async def get_agent_topic_view(self, agent_id: str, *, topic: str) -> Optional[dict[str, Any]]:
        """Project the agent vector into a topic anchor frame (fast; no embedding call)."""
        profile = await self.load(agent_id)
        if profile is None:
            return None

        analyzer = self._make_analyzer(topic)
        vec_for_scoring = profile.score_vector if profile.score_vector is not None else profile.vector
        scored = await analyzer.score_vector(vec_for_scoring)
        if scored is None:
            return None

        return {
            "agent_id": agent_id,
            "topic": topic,
            "embedding_model": profile.embedding_model,
            "updated_at": profile.updated_at,
            **scored,
        }
