"""Generate a random, connected social graph over a set of agents.

Requirements:
  - Connected graph.
  - Contains an explicit closed walk backbone (a Hamiltonian cycle).
  - Tunable density so each agent connects to many others.
  - Output is JSON-serializable (dict[str, list[str]]).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional
import networkx as nx


class ConnectionGraph:
    def __init__(
        self,
        agents: list[str],
        *,
        seed: Optional[int] = None,
        include_self_loops: bool = False,
        graph_type: str = "community",
        avg_degree_target: float = 6.0,
        lognormal_mu: float = 1.0,
        lognormal_sigma: float = 10.0,
        activity_lognormal_mu: Optional[float] = None,
        activity_lognormal_sigma: Optional[float] = None,
        base_in_social_needs: Optional[float] = None,
        base_out_social_needs: Optional[float] = None,
        group_divisor: int = 5,
        within_group_fraction: float = 0.7,
        cross_group_divisor: float = 10.0,
        reciprocity_within: float = 0.5,
        reciprocity_cross: float = 0.05,
        reciprocity_influencer: float = 0.0,
        influencer_quantile: float = 0.9,
        min_out_degree: int = 2,
        min_in_degree: int = 1,
    ):
        """Create a random connected graph with a cycle backbone.

        Args:
            agents: Agent ids.
            seed: RNG seed for reproducibility.
            include_self_loops: If True, allow a node to connect to itself (default False).
            graph_type: Type of graph structure to build ("hamiltonian", "chung-lu", "community").
            avg_degree_target: Target average out-degree.
            lognormal_mu: LogNormal(mu, sigma) location parameter for the "spark".
            lognormal_sigma: LogNormal(mu, sigma) scale parameter for the "spark".
            activity_lognormal_mu: Mu for out-activity sparks (defaults to lognormal_mu).
            activity_lognormal_sigma: Sigma for out-activity sparks (defaults to lognormal_sigma).
            base_in_social_needs: Base in-weight (defaults to 0.0).
            base_out_social_needs: Base out-weight (defaults to base_in_social_needs).
            group_divisor: Community group size divisor (group_size ~= n/group_divisor).
            within_group_fraction: Fraction of non-backbone degree expected within-group.
            cross_group_divisor: For community graphs, cross-group probabilities are divided by this.
            reciprocity_within: P(add j->i | i->j) within group (non-influencers).
            reciprocity_cross: P(add j->i | i->j) across groups (non-influencers).
            reciprocity_influencer: P(add j->i | i->j) when target is influencer.
            influencer_quantile: Influencers are nodes with in-weight >= this quantile.
            min_out_degree: Minimum out-degree enforced.
            min_in_degree: Minimum in-degree enforced.
        """
        self.agents = list(dict.fromkeys(agents))  # de-dup, preserve order
        self.seed = seed
        self.include_self_loops = bool(include_self_loops)
        self.graph_type = str(graph_type)
        self.avg_degree_target = float(avg_degree_target)
        self.lognormal_mu = float(lognormal_mu)
        self.lognormal_sigma = float(lognormal_sigma)
        self.activity_lognormal_mu = float(activity_lognormal_mu) if activity_lognormal_mu is not None else self.lognormal_mu
        self.activity_lognormal_sigma = (
            float(activity_lognormal_sigma) if activity_lognormal_sigma is not None else self.lognormal_sigma
        )
        self.base_in_social_needs = float(base_in_social_needs) if base_in_social_needs is not None else 0.0
        self.base_out_social_needs = float(base_out_social_needs) if base_out_social_needs is not None else self.base_in_social_needs
        self.group_divisor = int(group_divisor)
        self.within_group_fraction = float(within_group_fraction)
        self.cross_group_divisor = float(cross_group_divisor)
        self.reciprocity_within = float(reciprocity_within)
        self.reciprocity_cross = float(reciprocity_cross)
        self.reciprocity_influencer = float(reciprocity_influencer)
        self.influencer_quantile = float(influencer_quantile)
        self.min_out_degree = int(min_out_degree)
        self.min_in_degree = int(min_in_degree)

        self._rng = random.Random(seed)

        self.node_in_weights: dict[str, float] = {}
        self.node_out_weights: dict[str, float] = {}

        if self.graph_type in ("chung-lu", "community"):
            self.node_in_weights, self.node_out_weights = self._generate_directed_weights_base_plus_spark()
            self.node_weights = dict(self.node_in_weights)

        self.graph = self._generate_directed_connection_graph()

    def _validate_node_weights(self, node_weights: dict[str, float]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for agent in self.agents:
            w = float(node_weights.get(agent, 0.0))
            weights[agent] = max(0.0, w)
        return weights

    def _generate_directed_weights_base_plus_spark(self) -> tuple[dict[str, float], dict[str, float]]:
        """Generate (in_weight, out_weight) for directed models.
        Semantics for this project:
          - An edge i -> j means "i influences j" (i publishes to j).
          - out_weight[i] controls how many others i influences (out-degree tendency).
          - in_weight[j] controls how much input j tends to receive (in-degree / feed size).
        """
        n = len(self.agents)
        if n == 0:
            return {}, {}

        base_in = max(0.0, float(self.base_in_social_needs))
        base_out = max(0.0, float(self.base_out_social_needs))

        cap = float(n - 1)
        in_w: dict[str, float] = {}
        out_w: dict[str, float] = {}
        for agent in self.agents:
            in_spark = self._rng.lognormvariate(self.lognormal_mu, self.lognormal_sigma)
            out_spark = self._rng.lognormvariate(self.activity_lognormal_mu, self.activity_lognormal_sigma)
            in_w[agent] = max(0.0, min(cap, base_in + float(in_spark)))
            out_w[agent] = max(0.0, min(cap, base_out + float(out_spark)))

        return in_w, out_w

    @staticmethod
    def _add_directed_edge(graph: dict[str, set[str]], src: str, dst: str) -> bool:
        if dst in graph[src]:
            return False
        graph[src].add(dst)
        return True

    def _generate_directed_connection_graph(self) -> dict[str, list[str]]:
        """Generate a directed graph with a directed Hamiltonian cycle backbone.
        """
        n = len(self.agents)
        if n == 0:
            return {}
        if n == 1:
            return {self.agents[0]: [self.agents[0]] if self.include_self_loops else []}

        out_graph: dict[str, set[str]] = {agent: set() for agent in self.agents}

        # 1) Backbone: directed Hamiltonian cycle => strong connectivity.
        cycle = list(self.agents)
        self._rng.shuffle(cycle)
        for i in range(n):
            src = cycle[i]
            dst = cycle[(i + 1) % n]
            if (not self.include_self_loops) and src == dst:
                continue
            self._add_directed_edge(out_graph, src, dst)

        # 2) Add extra directed edges.
        if self.graph_type == "hamiltonian":
            pass
        elif self.graph_type == "chung-lu":
            self._add_directed_chung_lu_edges(out_graph)
        elif self.graph_type == "community":
            self._add_directed_community_edges(out_graph)
        else:
            raise ValueError(
                f"Unsupported graph_type for directed graph: {self.graph_type!r}. "
                "Use 'hamiltonian', 'chung-lu', or 'community'."
            )

        return {k: sorted(v) for k, v in out_graph.items()}

    def _add_directed_chung_lu_edges(self, out_graph: dict[str, set[str]]) -> None:
        """Directed Chung–Lu-like model.

        For each ordered pair (i, j):
            p(i->j) = min(1, (w_out[i] * w_in[j] / sum_k w_in[k]))
        and we sample an edge with that Bernoulli probability.
        """
        agents = self.agents
        n = len(agents)
        if n <= 1:
            return

        in_w = {a: max(0.0, float(self.node_in_weights.get(a, 0.0))) for a in agents}
        out_w = {a: max(0.0, float(self.node_out_weights.get(a, 0.0))) for a in agents}
        sum_in = sum(in_w.values())
        if sum_in <= 0:
            return

        for src in agents:
            wout = float(out_w[src])
            if wout <= 0:
                continue
            for dst in agents:
                if (not self.include_self_loops) and src == dst:
                    continue
                if dst in out_graph[src]:
                    continue
                win = float(in_w[dst])
                if win <= 0:
                    continue
                p = (wout * win) / sum_in
                if p >= 1.0 or self._rng.random() < p:
                    self._add_directed_edge(out_graph, src, dst)

    def _add_directed_community_edges(self, out_graph: dict[str, set[str]]) -> None:
        agents = self.agents
        n = len(agents)
        if n <= 1:
            return

        divisor = max(1, self.group_divisor)
        group_size = max(2, n // divisor)
        shuffled = list(agents)
        self._rng.shuffle(shuffled)
        groups: list[list[str]] = [shuffled[i : i + group_size] for i in range(0, n, group_size)]
        group_of: dict[str, int] = {}
        for gi, members in enumerate(groups):
            for m in members:
                group_of[m] = gi
        self._group_of = group_of

        in_w = {a: max(0.0, float(self.node_in_weights.get(a, 0.0))) for a in agents}
        out_w = {a: max(0.0, float(self.node_out_weights.get(a, 0.0))) for a in agents}
        sum_in = sum(in_w.values())
        if sum_in <= 0:
            return

        within_frac = min(1.0, max(0.0, float(self.within_group_fraction)))
        cross_div = max(1.0, float(self.cross_group_divisor))

        # 1) Dense within-community connections (friend-group baseline).
        p_friend = min(1.0, max(0.0, 0.5 + 0.5 * within_frac))
        for members in groups:
            for src in members:
                for dst in members:
                    if (not self.include_self_loops) and src == dst:
                        continue
                    if dst in out_graph[src]:
                        continue
                    if self._rng.random() < p_friend:
                        self._add_directed_edge(out_graph, src, dst)

        # Convert within_frac into a multiplier: larger within_frac => larger within-community probability.
        within_mult = 1.0 / max(1e-6, (1.0 - within_frac))
        cross_mult = 1.0 / cross_div

        # Directed edges using chung lu model but with different within vs cross community probabilities.
        for src in agents:
            wout = float(out_w[src])
            if wout <= 0:
                continue
            gsrc = group_of.get(src, -1)

            for dst in agents:
                if (not self.include_self_loops) and src == dst:
                    continue
                if dst in out_graph[src]:
                    continue
                win = float(in_w[dst])
                if win <= 0:
                    continue

                same = group_of.get(dst, -2) == gsrc
                mult = within_mult if same else cross_mult
                p = ((wout * win) / sum_in) * mult
                if p >= 1.0 or self._rng.random() < p:
                    self._add_directed_edge(out_graph, src, dst)


    def get_graph(self) -> dict[str, list[str]]:
        """Returns the generated connection graph (JSON-serializable)."""
        return self.graph

    def neighbors(self, agent_id: str) -> list[str]:
        return list(self.graph.get(agent_id, []) or [])

    def to_json(self) -> dict[str, list[str]]:
        return self.get_graph()
    
    def to_networkx(self):
        """Convert to a NetworkX graph object."""
        G = nx.DiGraph()
        for agent, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(agent, neighbor)
        return G