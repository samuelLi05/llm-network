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
        min_degree: int = 2,
        degree_exponent: float = 0.5,
        degree_factor: float = 2.0,
        degree_offset: float = 0.0,
        max_degree: Optional[int] = None,
        include_self_loops: bool = False,
        directed: bool = False,
        graph_type: str = "community",
        avg_degree_target: float = 6.0,
        base_social_needs: float = 4.0,
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
        node_weights: Optional[dict[str, float]] = None,
    ):
        """Create a random connected graph with a cycle backbone.

        Args:
            agents: Agent ids.
            seed: RNG seed for reproducibility.
            min_degree: Minimum degree per node (>=2 recommended).
            degree_exponent: Controls scaling with N (default sqrt(N)).
            degree_factor: Multiplier applied to N**degree_exponent.
            degree_offset: Additive offset to target degree.
            max_degree: Cap per-node degree. Defaults to (N-1).
            include_self_loops: If True, allow a node to connect to itself (default False).
            directed: If True, produce a directed graph (out-adjacency list).
            graph_type: Type of graph structure to build ("hamiltonian", "chung-lu", "community").
            avg_degree_target: Target average degree. If directed=True, this is avg out-degree.
            base_social_needs: Minimum functional weight for every node.
            lognormal_mu: LogNormal(mu, sigma) location parameter for the "spark".
            lognormal_sigma: LogNormal(mu, sigma) scale parameter for the "spark".
            activity_lognormal_mu: For directed graphs, mu for out-activity sparks (defaults to lognormal_mu).
            activity_lognormal_sigma: For directed graphs, sigma for out-activity sparks (defaults to lognormal_sigma).
            base_in_social_needs: For directed graphs, base in-weight (defaults to 0.0).
            base_out_social_needs: For directed graphs, base out-weight (defaults to base_social_needs).
            group_divisor: Community group size divisor (group_size ~= n/group_divisor).
            within_group_fraction: Fraction of non-backbone degree expected within-group.
            cross_group_divisor: For community graphs, cross-group probabilities are divided by this.
            reciprocity_within: For directed community graphs, P(add j->i | i->j) within group (non-influencers).
            reciprocity_cross: For directed community graphs, P(add j->i | i->j) across groups (non-influencers).
            reciprocity_influencer: For directed community graphs, P(add j->i | i->j) when target is influencer.
            influencer_quantile: Influencers are nodes with in-weight >= this quantile.
            min_out_degree: Minimum out-degree enforced in directed graphs.
            min_in_degree: Minimum in-degree enforced in directed graphs.
            node_weights: Optional explicit weights per agent (overrides generated weights).
        """
        self.agents = list(dict.fromkeys(agents))  # de-dup, preserve order
        self.seed = seed
        self.min_degree = int(min_degree) # min degree for vertex
        self.degree_exponent = float(degree_exponent)
        self.degree_factor = float(degree_factor)
        self.degree_offset = float(degree_offset)
        self.max_degree = max_degree # max degree for vertex
        self.include_self_loops = bool(include_self_loops)
        self.directed = bool(directed)
        self.graph_type = str(graph_type)
        self.avg_degree_target = float(avg_degree_target)
        self.base_social_needs = float(base_social_needs)
        self.lognormal_mu = float(lognormal_mu)
        self.lognormal_sigma = float(lognormal_sigma)
        self.activity_lognormal_mu = float(activity_lognormal_mu) if activity_lognormal_mu is not None else self.lognormal_mu
        self.activity_lognormal_sigma = (
            float(activity_lognormal_sigma) if activity_lognormal_sigma is not None else self.lognormal_sigma
        )
        self.base_in_social_needs = float(base_in_social_needs) if base_in_social_needs is not None else 0.0
        self.base_out_social_needs = float(base_out_social_needs) if base_out_social_needs is not None else self.base_social_needs
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

        if self.directed and self.graph_type in ("chung-lu", "community"):
            self.node_in_weights, self.node_out_weights = self._generate_directed_weights_base_plus_spark()
            # Back-compat: expose "node_weights" as influence weights.
            self.node_weights = dict(self.node_in_weights)
        else:
            if node_weights is not None:
                self.node_weights = self._validate_node_weights(node_weights)
            elif self.graph_type in ("chung-lu", "community"):
                self.node_weights = self._generate_weights_base_plus_spark()
            else:
                self.node_weights = {agent: 1.0 for agent in self.agents}  # uniform weights

        self.graph = self._generate_connection_graph()

    def _target_degree(self, n: int) -> int:
        if n <= 1:
            return 0
        cap = int(self.max_degree) if self.max_degree is not None else (n - 1)
        cap = max(0, min(n - 1, cap))
        base = int(round((n ** self.degree_exponent) * self.degree_factor + self.degree_offset))
        return max(0, min(cap, max(int(self.min_degree), base)))

    def _validate_node_weights(self, node_weights: dict[str, float]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for agent in self.agents:
            w = float(node_weights.get(agent, 0.0))
            weights[agent] = max(0.0, w)
        return weights

    def _generate_weights_base_plus_spark(self) -> dict[str, float]:
        """Generate weights using the Base + Spark method.

        We generate
            w_i = base_social_needs + alpha * LogNormal(mu, sigma)

        and choose alpha so that sum(w_i) ~= n * (avg_degree_target - 2).
        The "-2" accounts for the guaranteed Hamiltonian cycle backbone.
        """
        n = len(self.agents)
        if n == 0:
            return {}

        base = max(0.0, self.base_social_needs)
        # Keep avg_degree_target sane.
        k_total = max(0.0, self.avg_degree_target)
        k_remaining = max(0.0, k_total - 2.0)
        if base > k_remaining and k_remaining > 0:
            # Still allow it, but it will push degrees above the target.
            base = k_remaining

        sparks = [self._rng.lognormvariate(self.lognormal_mu, self.lognormal_sigma) for _ in range(n)]
        spark_sum = sum(sparks)
        target_sum = n * k_remaining

        if spark_sum <= 0 or target_sum <= n * base:
            alpha = 0.0
        else:
            alpha = (target_sum - n * base) / spark_sum

        cap = float(n - 1)
        weights: dict[str, float] = {}
        for agent, spark in zip(self.agents, sparks):
            w = base + alpha * float(spark)
            weights[agent] = max(0.0, min(cap, w))
        return weights

    def _generate_directed_weights_base_plus_spark(self) -> tuple[dict[str, float], dict[str, float]]:
        """Generate (in_weight, out_weight) for directed models.

        Semantics for this project:
          - An edge i -> j means "i influences j" (i publishes to j).
          - out_weight[i] controls how many others i influences (out-degree tendency).
          - in_weight[j] controls how much input j tends to receive (in-degree / feed size).

        We sample both using a simple Base + LogNormal Spark model. We do NOT enforce
        an exact average degree; the graph density is controlled by these distributions
        plus the within/cross community multipliers.
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

    @staticmethod
    def _add_undirected_edge(graph: dict[str, set[str]], a: str, b: str) -> bool:
        if b in graph[a]:
            return False
        graph[a].add(b)
        graph[b].add(a)
        return True

    def _generate_connection_graph(self) -> dict[str, list[str]]:
        """Generate a connected random graph with a guaranteed cycle (closed walk)."""
        if self.directed:
            return self._generate_directed_connection_graph()

        n = len(self.agents)
        if n == 0:
            return {}
        if n == 1:
            return {self.agents[0]: [self.agents[0]] if self.include_self_loops else []}

        graph: dict[str, set[str]] = {agent: set() for agent in self.agents}

        # 1) Backbone: random Hamiltonian cycle => connected + closed walk.
        cycle = list(self.agents)
        self._rng.shuffle(cycle)
        for i in range(n):
            a = cycle[i]
            b = cycle[(i + 1) % n]
            self._add_undirected_edge(graph, a, b)

        # 2) Add extra edges based on graph type.
        if self.graph_type == "chung-lu":
            self._add_chung_lu_edges_networkx(graph)
            self._top_up_low_degree_first(graph)
        elif self.graph_type == "community":
            self._add_community_edges(graph)
            self._top_up_low_degree_first(graph)
        else:  # hamiltonian random augmented edges
            target = self._target_degree(n)
            if target > 2:
                # Attempts are bounded to avoid infinite loops when near-complete.
                max_attempts = max(1000, n * n * 10)
                attempts = 0

                def degree(a: str) -> int:
                    return len(graph[a])

                while attempts < max_attempts:
                    attempts += 1
                    low = [a for a in self.agents if degree(a) < target]
                    if not low:
                        break

                    a = self._rng.choice(low)

                    # Prefer to connect to another node that is also under target.
                    candidates = [
                        b
                        for b in self.agents
                        if (self.include_self_loops or b != a)
                        and b not in graph[a]
                        and degree(b) < target
                    ]
                    if not candidates:
                        # Relax: allow connecting to any not-already-connected node.
                        candidates = [
                            b
                            for b in self.agents
                            if (self.include_self_loops or b != a)
                            and b not in graph[a]
                        ]
                    if not candidates:
                        # Node is saturated.
                        continue

                    b = self._rng.choice(candidates)
                    self._add_undirected_edge(graph, a, b)

        # Finalize to JSON-serializable adjacency list.
        final_graph = {k: sorted(v) for k, v in graph.items()}
        # Best-effort sanity check (cycle guarantees connectedness, but keep it safe).
        if n > 1 and not self._is_connected(final_graph):
            # Should not happen; fall back to the cycle-only graph.
            cycle_only: dict[str, set[str]] = {agent: set() for agent in self.agents}
            for i in range(n):
                a = cycle[i]
                b = cycle[(i + 1) % n]
                self._add_undirected_edge(cycle_only, a, b)
            final_graph = {k: sorted(v) for k, v in cycle_only.items()}

        return final_graph

    def _generate_directed_connection_graph(self) -> dict[str, list[str]]:
        """Generate a directed graph with a directed Hamiltonian cycle backbone.

        Output adjacency is out-neighbors.
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
        # Each ordered within-group pair gets an edge with high probability,
        # so members receive/send most inputs within their friend group.
        # Use `within_group_fraction` as the density knob.
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
        # (This keeps your config surface minimal: one knob for "stickiness".)
        within_mult = 1.0 / max(1e-6, (1.0 - within_frac))
        cross_mult = 1.0 / cross_div

        # 2) Weighted directed edges (DCSBM / Chung–Lu style) layered on top.
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

    def _ensure_min_degrees(self, out_graph: dict[str, set[str]]) -> None:
        """Enforce min in/out degree in directed graphs without flattening degree variance.

        NOTE: We intentionally do NOT top up to an exact average out-degree here;
        the expected average is controlled by the weight scaling + edge sampling.
        """
        agents = self.agents
        n = len(agents)
        if n <= 1:
            return

        min_out = max(0, int(self.min_out_degree))
        min_in = max(0, int(self.min_in_degree))

        in_w = {a: float(self.node_in_weights.get(a, 1.0)) for a in agents}
        out_w = {a: float(self.node_out_weights.get(a, 1.0)) for a in agents}
        if sum(in_w.values()) <= 0:
            in_w = {a: 1.0 for a in agents}
        if sum(out_w.values()) <= 0:
            out_w = {a: 1.0 for a in agents}

        group_of = getattr(self, "_group_of", {}) if self.graph_type == "community" else {}
        cross_div = max(1.0, float(self.cross_group_divisor))

        def out_deg(a: str) -> int:
            return len(out_graph[a])

        def in_degrees() -> dict[str, int]:
            indeg = {a: 0 for a in agents}
            for src in agents:
                for dst in out_graph[src]:
                    if dst in indeg:
                        indeg[dst] += 1
            return indeg

        def pick_target(src: str, *, prefer_invisible: Optional[set[str]] = None) -> Optional[str]:
            candidates = [
                b
                for b in agents
                if (self.include_self_loops or b != src) and b not in out_graph[src]
            ]
            if prefer_invisible:
                invis = [b for b in candidates if b in prefer_invisible]
                if invis:
                    candidates = invis
            if not candidates:
                return None

            if self.graph_type == "community":
                gsrc = group_of.get(src, -1)
                weights = []
                for b in candidates:
                    factor = 1.0 if group_of.get(b, -2) == gsrc else (1.0 / cross_div)
                    weights.append(max(0.0, in_w[b] * factor))
            else:
                weights = [max(0.0, in_w[b]) for b in candidates]

            if sum(weights) <= 0:
                weights = [1.0 for _ in candidates]
            return self._rng.choices(candidates, weights=weights, k=1)[0]

        # A) Ensure minimum out-degree (no empty feeds).
        max_attempts = max(10_000, n * n * 30)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            lows = [a for a in agents if out_deg(a) < min_out]
            if not lows:
                break
            src = self._rng.choice(lows)
            dst = pick_target(src)
            if dst is None:
                break
            self._add_directed_edge(out_graph, src, dst)

        # B) Ensure minimum in-degree (no invisible users).
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            indeg = in_degrees()
            invis = {a for a in agents if indeg[a] < min_in}
            if not invis:
                break

            # Prefer to satisfy invisibility from within the same group when possible.
            dst = self._rng.choice(list(invis))
            if self.graph_type == "community":
                gdst = group_of.get(dst, -2)
                src_candidates = [
                    a
                    for a in agents
                    if (self.include_self_loops or a != dst)
                    and dst not in out_graph[a]
                    and group_of.get(a, -1) == gdst
                ]
                if not src_candidates:
                    src_candidates = [a for a in agents if (self.include_self_loops or a != dst) and dst not in out_graph[a]]
            else:
                src_candidates = [a for a in agents if (self.include_self_loops or a != dst) and dst not in out_graph[a]]

            if not src_candidates:
                break

            src_weights = [max(0.0, out_w.get(a, 1.0)) for a in src_candidates]
            src = self._rng.choices(src_candidates, weights=src_weights, k=1)[0]
            self._add_directed_edge(out_graph, src, dst)

    def _add_chung_lu_edges_networkx(self, graph: dict[str, set[str]]) -> None:
        """Add edges using NetworkX's Chung–Lu graph generator.

        Expects node_weights to represent expected degrees for the non-backbone edges.
        """
        n = len(self.agents)
        if n <= 1:
            return

        expected = [max(0.0, float(self.node_weights.get(a, 0.0))) for a in self.agents]
        G = nx.expected_degree_graph(expected, seed=self.seed, selfloops=self.include_self_loops)
        mapping = {i: agent for i, agent in enumerate(self.agents)}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            if (not self.include_self_loops) and u == v:
                continue
            self._add_undirected_edge(graph, str(u), str(v))

    def _top_up_low_degree_first(self, graph: dict[str, set[str]]) -> None:
        """Low-degree-first top-up to reach avg_degree_target.

        This acts as the "discovery" / fairness rule: prioritize connecting
        the loneliest nodes when filling remaining edges.
        """
        agents = self.agents
        n = len(agents)
        if n <= 1:
            return

        target_avg = max(0.0, float(self.avg_degree_target))
        if target_avg <= 0:
            return

        def current_avg() -> float:
            return sum(len(graph[a]) for a in agents) / n

        weights = {a: max(0.0, float(self.node_weights.get(a, 1.0))) for a in agents}
        if sum(weights.values()) <= 0:
            weights = {a: 1.0 for a in agents}

        avg = current_avg()
        max_attempts = max(10_000, n * n * 30)
        attempts = 0
        while avg < target_avg and attempts < max_attempts:
            attempts += 1
            # Pick from the lowest-degree bucket.
            degrees = {a: len(graph[a]) for a in agents}
            min_deg = min(degrees.values())
            low_bucket = [a for a, d in degrees.items() if d == min_deg]
            a = self._rng.choice(low_bucket)

            candidates = [
                b
                for b in agents
                if (self.include_self_loops or b != a)
                and b not in graph[a]
            ]
            if not candidates:
                break

            # Weight partner choice by influence + (for community) by group factor.
            if self.graph_type == "community":
                group_of = getattr(self, "_group_of", {})
                ga = group_of.get(a, -1)
                cross_div = max(1.0, float(self.cross_group_divisor))
                cand_weights = []
                for b in candidates:
                    gb = group_of.get(b, -2)
                    factor = 1.0 if ga == gb else (1.0 / cross_div)
                    cand_weights.append(max(0.0, weights[b] * factor))
            else:
                cand_weights = [max(0.0, weights[b]) for b in candidates]

            if sum(cand_weights) <= 0:
                cand_weights = [1.0 for _ in candidates]
            b = self._rng.choices(candidates, weights=cand_weights, k=1)[0]
            self._add_undirected_edge(graph, a, b)
            avg = current_avg()

    def _add_community_edges(self, graph: dict[str, set[str]]) -> None:
        """Add edges with a simple degree-corrected SBM flavor.

        - Assigns groups of size ~= n/group_divisor.
        - Uses Chung–Lu weighting within groups.
        - Uses Chung–Lu weighting across groups, but divided by cross_group_divisor.
        """
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

        # Store for low-degree-first top-up.
        self._group_of = group_of

        # Degree-corrected SBM (DCSBM-like):
        #   p_ij = min(1, (w_i w_j / sum_w) * B_{g_i,g_j})
        # where B=1 within-group and B=1/cross_group_divisor across-group.
        weights = self.node_weights
        sum_w = sum(float(weights[a]) for a in agents)
        if sum_w <= 0:
            return

        cross_div = max(1.0, float(self.cross_group_divisor))
        frac_in = min(1.0, max(0.0, self.within_group_fraction))
        # Scale within-group probability up so we can hit the within_group_fraction mix.
        # This is a simple heuristic multiplier.
        in_multiplier = 1.0 / max(1e-9, frac_in) if frac_in > 0 else 1.0
        in_multiplier = min(10.0, max(1.0, in_multiplier))

        for i in range(n):
            a = agents[i]
            wa = float(weights[a])
            ga = group_of.get(a, -1)
            for j in range(i + 1, n):
                b = agents[j]
                if b in graph[a]:
                    continue
                wb = float(weights[b])
                gb = group_of.get(b, -2)
                if ga == gb:
                    B = in_multiplier
                else:
                    B = 1.0 / cross_div
                p = (wa * wb / sum_w) * B
                if p >= 1.0 or self._rng.random() < p:
                    self._add_undirected_edge(graph, a, b)

    @staticmethod
    def _is_connected(graph: dict[str, list[str]]) -> bool:
        if not graph:
            return True
        start = next(iter(graph.keys()))
        seen: set[str] = set([start])
        q: deque[str] = deque([start])
        while q:
            cur = q.popleft()
            for nxt in graph.get(cur, []) or []:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return len(seen) == len(graph)

    def get_graph(self) -> dict[str, list[str]]:
        """Returns the generated connection graph (JSON-serializable)."""
        return self.graph

    def neighbors(self, agent_id: str) -> list[str]:
        return list(self.graph.get(agent_id, []) or [])

    def to_json(self) -> dict[str, list[str]]:
        return self.get_graph()
    
    def to_networkx(self):
        """Convert to a NetworkX graph object."""
        G = nx.DiGraph() if self.directed else nx.Graph()
        for agent, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(agent, neighbor)
        return G