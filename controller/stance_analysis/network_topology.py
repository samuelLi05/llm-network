import time
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from controller.stance_analysis.agent_profile_store import AgentProfileStore
from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.vector_ops import to_np
from network.cache import RedisCache

if TYPE_CHECKING:
    from controller.time_manager import TimeManager


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
        scoring_topic: Optional[str] = None,
        profile_store: AgentProfileStore,
        time_manager: Optional["TimeManager"] = None,
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
        self.scoring_topic = str(scoring_topic or topic)
        self.profile_store = profile_store
        self.time_manager = time_manager
        self.redis_cache = redis_cache
        self.redis_key = redis_key or f"topology:{topic}"
        self.min_edge_similarity = float(min_edge_similarity)
        self.update_interval_s = float(update_interval_s)

        self._analyzer = EmbeddingAnalyzer(
            self.scoring_topic,
            use_local_embedding_model=bool(use_local_embedding_model),
            use_baseline_statement=bool(use_baseline_statement),
            openai_embedding_model=openai_embedding_model,
            local_embedding_model=local_embedding_model,
        )
        self._last_update_ts: float = 0.0

    def _now_s(self) -> float:
        if self.time_manager is None:
            return time.time()
        try:
            return float(self.time_manager.now_s())
        except Exception:
            return time.time()

    def _time_info(self) -> Optional[dict[str, Any]]:
        if self.time_manager is None:
            return None
        try:
            return dict(self.time_manager.time_info())
        except Exception:
            return None

    async def snapshot(
        self,
        agent_ids: list[str],
        *,
        include_vectors: bool = False,
        include_seed_metadata: bool = False,
    ) -> dict[str, Any]:
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
            info = self._time_info()
            return {
                "tp": self.topic,
                "ts": self._now_s(),
                "n": {},
                **({"time": info} if info else {}),
            }

        mat = np.stack(profiles).astype(np.float32, copy=False)

        nodes: dict[str, dict[str, Any]] = {}
        for i, agent_id in enumerate(present_ids):
            p = profile_by_id.get(agent_id)
            vec_for_scoring = getattr(p, "score_vector", None) or mat[i].tolist()
            scored = await self._analyzer.score_vector(vec_for_scoring, include_vector=False)
            scored = scored or {}
            node = {
                "ts": float(scored.get("topic_similarity", 0.0)),
                "ss": float(scored.get("stance_score", 0.0)),
                "str": float(scored.get("strength", 0.0)),
                "uat": float(getattr(p, "updated_at", 0.0)),
            }
            if include_seed_metadata:
                node["seed_text"] = str(getattr(p, "seed_text", "") or "")
                node["seeded"] = bool(getattr(p, "seed_vector", None) is not None)
            if include_vectors:
                node["vec"] = to_np(getattr(p, "vector", []), dtype=np.float32).astype(np.float32, copy=False).tolist()
                score_out = getattr(p, "score_vector", None)
                if score_out is None:
                    score_out = vec_for_scoring
                node["svec"] = to_np(score_out, dtype=np.float32).astype(np.float32, copy=False).tolist()
            nodes[agent_id] = node

        info = self._time_info()
        return {
            "tp": self.topic,
            "ts": self._now_s(),
            "n": nodes,
            **({"time": info} if info else {}),
        }

    async def maybe_update(self, agent_ids: list[str], *, force: bool = False) -> Optional[dict[str, Any]]:
        now = self._now_s()
        if not force and (now - self._last_update_ts) < self.update_interval_s:
            return None

        snap = await self.snapshot(agent_ids)
        self._last_update_ts = now

        if self.redis_cache:
            await self.redis_cache.append_response(self.redis_key, snap)

        return snap
