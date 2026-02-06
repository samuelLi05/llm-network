import time
from typing import Any, Optional

import numpy as np

from controller.stance_analysis.agent_profile_store import AgentProfileStore
from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.vector_ops import to_np
from network.cache import RedisCache


class NetworkTopologyTracker:
    """Computes a lightweight topology snapshot of the agent embedding network.

    This is intended for modeling/analysis, not for runtime-critical path. For the
    default N=10 agents, an O(N^2) similarity matrix is cheap.

    Snapshot format:
      - nodes: per-agent embedding/stance features
      - edges: cosine similarity between agent profile vectors (thresholded)
    """

    def __init__(
        self,
        *,
        topic: str,
        profile_store: AgentProfileStore,
        redis_cache: Optional[RedisCache] = None,
        redis_key: Optional[str] = None,
        min_edge_similarity: float = 0.15,
        update_interval_s: float = 10.0,
        use_local_embedding_model: bool = False,
        use_baseline_statement: bool = False,
        openai_embedding_model: str = "text-embedding-3-small",
        local_embedding_model: str = "all-mpnet-base-v2S",
    ):
        self.topic = topic
        self.profile_store = profile_store
        self.redis_cache = redis_cache
        self.redis_key = redis_key or f"topology:{topic}"
        self.min_edge_similarity = float(min_edge_similarity)
        self.update_interval_s = float(update_interval_s)

        self._analyzer = EmbeddingAnalyzer(
            topic,
            use_local_embedding_model=bool(use_local_embedding_model),
            use_baseline_statement=bool(use_baseline_statement),
            openai_embedding_model=openai_embedding_model,
            local_embedding_model=local_embedding_model,
        )
        self._last_update_ts: float = 0.0

    async def snapshot(self, agent_ids: list[str]) -> dict[str, Any]:
        profiles = []
        profile_by_id: dict[str, Any] = {}
        present_ids: list[str] = []
        for agent_id in agent_ids:
            p = await self.profile_store.load(agent_id)
            if p is None or p.vector is None:
                continue
            present_ids.append(agent_id)
            profile_by_id[agent_id] = p
            profiles.append(to_np(p.vector, dtype=np.float32))

        if not profiles:
            return {
                "topic": self.topic,
                "timestamp": time.time(),
                "nodes": [],
                "edges": [],
            }

        mat = np.stack(profiles).astype(np.float32, copy=False)
        sims = mat @ mat.T  # (N,N) because vectors are normalized

        nodes: list[dict[str, Any]] = []
        for i, agent_id in enumerate(present_ids):
            p = profile_by_id.get(agent_id)
            vec_for_scoring = getattr(p, "score_vector", None) or mat[i].tolist()
            scored = await self._analyzer.score_vector(vec_for_scoring, include_vector=False)
            scored = scored or {}
            nodes.append(
                {
                    "agent_id": agent_id,
                    "topic_similarity": float(scored.get("topic_similarity", 0.0)),
                    "stance_score": float(scored.get("stance_score", 0.0)),
                    "strength": float(scored.get("strength", 0.0)),
                    "updated_at": float(getattr(p, "updated_at", 0.0)),
                }
            )

        edges: list[dict[str, Any]] = []
        n = len(present_ids)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(sims[i, j])
                if w >= self.min_edge_similarity:
                    edges.append({"source": present_ids[i], "target": present_ids[j], "weight": w})

        return {
            "topic": self.topic,
            "timestamp": time.time(),
            "nodes": nodes,
            "edges": edges,
        }

    async def maybe_update(self, agent_ids: list[str], *, force: bool = False) -> Optional[dict[str, Any]]:
        now = time.time()
        if not force and (now - self._last_update_ts) < self.update_interval_s:
            return None

        snap = await self.snapshot(agent_ids)
        self._last_update_ts = now

        if self.redis_cache:
            await self.redis_cache.append_response(self.redis_key, snap)

        return snap
