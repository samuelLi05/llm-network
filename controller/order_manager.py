import random
import os
import time
import json
from typing import List, Optional

from agents.network_agent import NetworkAgent
from controller.stance_analysis.agent_profile_store import AgentProfileStore
from network.cache import RedisCache
from network.stream import RedisStream

"""Manages ordering and coordination of responses among multiple NetworkAgents.

Selects a random agent excluding the sender of the most recent message
from the Redis stream. The selected next responder is stored in Redis
so all agents see a consistent value.
"""

# Redis key for storing the designated next responder
NEXT_RESPONDER_KEY = "llm_network:next_responder"

# Optional: store the connection graph so agents/processes can fetch it.
CONNECTION_GRAPH_KEY = "llm_network:connection_graph"

# Optional bookkeeping for fairness / debiasing
ORDER_LAST_SELECTED_PREFIX = "llm_network:order:last_selected:"
ORDER_COUNT_PREFIX = "llm_network:order:count:"
ORDER_LAST_META_KEY = "llm_network:order:last_meta"


class OrderManager:
    def __init__(
        self,
        agents: List[NetworkAgent],
        message_cache: RedisCache,
        profile_store: Optional[AgentProfileStore] = None,
        connection_graph: Optional[dict[str, list[str]]] = None,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
    ):
        self.agents = agents
        self.message_cache = message_cache
        self.profile_store = profile_store
        self.connection_graph = connection_graph or None
        # RedisStream uses a synchronous redis client with decode_responses=True
        self.stream_client = RedisStream(host=redis_host, port=redis_port)

        # Best-effort persist the graph for other consumers.
        if self.connection_graph:
            try:
                self.stream_client.redis.set(CONNECTION_GRAPH_KEY, json.dumps(self.connection_graph))
            except Exception:
                pass

        # Ordering policy knobs (env-configurable)
        self.ordering_mode = "topology"  # random|topology
        self.echo_probability = 0.6
        self.fairness_tau_s = 30.0
        self.cooldown_s = 0.0
        self.temperature = 0.8
        self.sim_weight = 1.0
        self.fair_weight = 0.6
        self.extremeness_penalty = 0.35
        self.explore_epsilon = 0.08

    def _key_last_selected(self, agent_id: str) -> str:
        return f"{ORDER_LAST_SELECTED_PREFIX}{agent_id}"

    def _key_count(self, agent_id: str) -> str:
        return f"{ORDER_COUNT_PREFIX}{agent_id}"

    def _get_last_selected_ts(self, agent_id: str) -> Optional[float]:
        try:
            raw = self.stream_client.redis.get(self._key_last_selected(agent_id))
            if raw is None:
                return None
            return float(raw)
        except Exception:
            return None

    def _record_selected(self, agent_id: str, *, meta: Optional[dict] = None) -> None:
        try:
            now = time.time()
            self.stream_client.redis.set(self._key_last_selected(agent_id), str(now))
            try:
                self.stream_client.redis.incr(self._key_count(agent_id))
            except Exception:
                pass
            if meta is not None:
                # keep small; debugging only
                self.stream_client.redis.set(ORDER_LAST_META_KEY, str(meta))
        except Exception:
            pass

    def get_random_agent(self, exclude_agent_id: Optional[str] = None) -> str:
        """Return a random agent id, optionally excluding one agent.

        Args:
            exclude_agent_id: Agent id to exclude from selection (e.g., last publisher).

        Returns:
            The selected agent's id.

        Raises:
            ValueError: If no agents are available.
        """
        if not self.agents:
            raise ValueError("No agents available")

        candidates = [a for a in self.agents if a.id != exclude_agent_id]
        if not candidates:
            # If exclusion removes everyone, fall back to all agents
            candidates = self.agents

        chosen = random.choice(candidates)
        return chosen.id

    def get_neighbors(self, agent_id: str) -> Optional[list[str]]:
        """Return the neighbor list for an agent, if a connection graph is configured."""
        if not self.connection_graph:
            # Try lazy load from Redis (useful if OrderManager is recreated elsewhere).
            try:
                raw = self.stream_client.redis.get(CONNECTION_GRAPH_KEY)
                if raw:
                    self.connection_graph = json.loads(raw)
            except Exception:
                return None
            if not self.connection_graph:
                return None
        neigh = self.connection_graph.get(agent_id)
        if not neigh:
            return None
        # De-dup while preserving order
        seen = set()
        out: list[str] = []
        for x in neigh:
            xs = str(x)
            if xs not in seen and xs != agent_id:
                seen.add(xs)
                out.append(xs)
        return out or None

    async def _select_topology_weighted(
        self,
        *,
        reference_agent_id: Optional[str],
        exclude_agent_id: Optional[str],
        candidate_agent_ids: Optional[set[str]] = None,
    ) -> str:
        """Select next responder using agent profile vectors with guardrails.

        Goals:
          - Mix echo-chamber and cross-cutting replies (echo_probability)
          - Avoid the same agent dominating (recency-based fairness + optional cooldown)
          - Reduce 'only extremes post' trap (penalize distance from population centroid)
          - Keep some exploration (epsilon random)
        """

        # Hard fallback if we can't do topology
        if self.profile_store is None or not self.agents:
            return self.get_random_agent(exclude_agent_id=exclude_agent_id)

        candidates = [a for a in self.agents if a.id != exclude_agent_id]
        if candidate_agent_ids is not None:
            candidates = [a for a in candidates if a.id in candidate_agent_ids]
        if not candidates:
            candidates = list(self.agents)
            if candidate_agent_ids is not None:
                candidates = [a for a in candidates if a.id in candidate_agent_ids]

        # Apply a cooldown to reduce ping-pong / dominance
        if self.cooldown_s > 0:
            now = time.time()
            cooled = []
            for a in candidates:
                last_ts = self._get_last_selected_ts(a.id)
                if last_ts is None or (now - last_ts) >= self.cooldown_s:
                    cooled.append(a)
            if cooled:
                candidates = cooled

        # Epsilon exploration: random (but still exclude current)
        if self.explore_epsilon > 0 and random.random() < self.explore_epsilon:
            return random.choice(candidates).id

        # Load vectors
        ref_vec = None
        if reference_agent_id:
            ref_profile = await self.profile_store.load(reference_agent_id)
            if ref_profile is not None and ref_profile.vector is not None:
                ref_vec = ref_profile.vector

        cand_profiles = []  # (agent_id, vector or None)
        for a in candidates:
            p = await self.profile_store.load(a.id)
            cand_profiles.append((a.id, None if p is None else p.vector))

        # If reference vector missing, fall back to fairness-only weighted random
        if ref_vec is None:
            picked = self._pick_by_fairness([aid for aid, _ in cand_profiles])
            return picked

        # Compute centroid of available vectors for extremeness penalty
        vecs = [v for _, v in cand_profiles if v is not None]
        centroid = None
        if vecs:
            # average then renormalize (best-effort)
            dim = len(vecs[0])
            mean = [0.0] * dim
            for v in vecs:
                for i in range(dim):
                    mean[i] += float(v[i])
            inv = 1.0 / float(len(vecs))
            for i in range(dim):
                mean[i] *= inv
            # normalize
            norm = sum(x * x for x in mean) ** 0.5
            if norm > 0:
                centroid = [x / norm for x in mean]

        wants_echo = random.random() < max(0.0, min(1.0, self.echo_probability))
        now = time.time()

        scored = []  # (agent_id, score, sim, fairness, extremeness)
        for agent_id, v in cand_profiles:
            # Similarity to reference
            sim = 0.0
            if v is not None:
                sim = sum(float(a) * float(b) for a, b in zip(ref_vec, v))

            sim_component = sim if wants_echo else (-sim)

            # Fairness (prefer agents that haven't spoken recently)
            last_ts = self._get_last_selected_ts(agent_id)
            if last_ts is None or self.fairness_tau_s <= 0:
                fairness = 1.0
            else:
                fairness = max(0.0, min(1.0, (now - last_ts) / self.fairness_tau_s))

            # Extremeness: distance from centroid (penalize large distances)
            extremeness = 0.0
            if centroid is not None and v is not None:
                cent_sim = sum(float(a) * float(b) for a, b in zip(centroid, v))
                extremeness = 1.0 - float(cent_sim)

            score = (
                (self.sim_weight * sim_component)
                + (self.fair_weight * fairness)
                - (self.extremeness_penalty * extremeness)
                + (random.random() * 1e-3)  # tiny tie-break
            )
            scored.append((agent_id, score, sim, fairness, extremeness))

        # Softmax sample
        agent_ids = [s[0] for s in scored]
        scores = [s[1] for s in scored]
        picked = self._softmax_pick(agent_ids, scores, temperature=self.temperature)

        # Persist bookkeeping/debug meta
        try:
            top = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
            meta = {
                "mode": "echo" if wants_echo else "contrast",
                "reference": reference_agent_id,
                "exclude": exclude_agent_id,
                "picked": picked,
                "top": [
                    {
                        "agent_id": a,
                        "score": float(sc),
                        "sim": float(si),
                        "fair": float(fa),
                        "extreme": float(ex),
                    }
                    for a, sc, si, fa, ex in top
                ],
            }
            self._record_selected(picked, meta=meta)
        except Exception:
            self._record_selected(picked, meta=None)

        return picked

    def _pick_by_fairness(self, agent_ids: list[str]) -> str:
        if not agent_ids:
            raise ValueError("No agents available")
        now = time.time()
        weights = []
        for aid in agent_ids:
            last_ts = self._get_last_selected_ts(aid)
            if last_ts is None or self.fairness_tau_s <= 0:
                w = 1.0
            else:
                w = max(0.05, min(1.0, (now - last_ts) / self.fairness_tau_s))
            weights.append(w)
        return random.choices(agent_ids, weights=weights, k=1)[0]

    @staticmethod
    def _softmax_pick(agent_ids: list[str], scores: list[float], *, temperature: float) -> str:
        if not agent_ids:
            raise ValueError("No agents available")
        if len(agent_ids) == 1:
            return agent_ids[0]

        t = temperature if temperature and temperature > 0 else 1.0
        scaled = [s / t for s in scores]
        m = max(scaled)
        exps = [pow(2.718281828, (s - m)) for s in scaled]
        total = sum(exps)
        if total <= 0:
            return random.choice(agent_ids)
        probs = [e / total for e in exps]
        return random.choices(agent_ids, weights=probs, k=1)[0]

    def get_last_publisher(self, stream_name: str) -> Optional[str]:
        """Return the sender_id of the most recent message in the stream, or None."""
        try:
            entries = self.stream_client.redis.xrevrange(stream_name, count=1)
            if entries:
                _, data = entries[0]
                sender = data.get("sender_id") if isinstance(data, dict) else None
                if isinstance(sender, bytes):
                    return sender.decode()
                return sender
        except Exception:
            pass
        return None

    async def select_and_store_next_responder(self, exclude_agent_id: Optional[str] = None) -> str:
        """Select the next responder, store it in Redis, and return the agent id.

        Called by the publishing agent *before* publishing so other agents
        can see who is designated to respond next.
        """
        # Neighbor restriction: if a connection graph exists, we select the next responder
        # from the neighbors of the publishing agent (exclude_agent_id).
        candidate_agent_ids: Optional[set[str]] = None
        if exclude_agent_id and self.connection_graph:
            neigh = self.get_neighbors(exclude_agent_id)
            if neigh:
                candidate_agent_ids = set(neigh)

        if self.ordering_mode == "topology" and self.profile_store is not None:
            next_agent_id = await self._select_topology_weighted(
                reference_agent_id=exclude_agent_id,
                exclude_agent_id=exclude_agent_id,
                candidate_agent_ids=candidate_agent_ids,
            )
        else:
            if candidate_agent_ids is None:
                next_agent_id = self.get_random_agent(exclude_agent_id=exclude_agent_id)
            else:
                candidates = [a for a in self.agents if a.id in candidate_agent_ids and a.id != exclude_agent_id]
                if not candidates:
                    candidates = [a for a in self.agents if a.id != exclude_agent_id]
                next_agent_id = random.choice(candidates).id

        # Store in Redis so all agents see the same value
        self.stream_client.redis.set(NEXT_RESPONDER_KEY, next_agent_id)
        return next_agent_id

    def get_designated_responder(self) -> Optional[str]:
        """Get the currently designated next responder from Redis.
        
        Returns None if no responder is designated.
        """
        try:
            value = self.stream_client.redis.get(NEXT_RESPONDER_KEY)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode()
            return str(value)
        except Exception:
            return None

    def clear_designated_responder(self) -> None:
        """Clear the designated responder (called after an agent responds)."""
        try:
            self.stream_client.redis.delete(NEXT_RESPONDER_KEY)
        except Exception:
            pass

    def is_my_turn(self, agent_id: str) -> bool:
        """Check if the given agent is the designated next responder."""
        designated = self.get_designated_responder()
        return designated == agent_id

    # Legacy method - now just reads from Redis instead of computing
    def get_next_agent(self, stream_name: str) -> Optional[str]:
        """Get the designated next responder (reads from Redis, does not select new one)."""
        return self.get_designated_responder()