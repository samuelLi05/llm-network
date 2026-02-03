"""Generate a random, connected social graph over a set of agents.

Requirements:
  - Connected graph.
  - Contains an explicit closed walk backbone (a Hamiltonian cycle).
  - Tunable density so each agent connects to many others.
  - Output is JSON-serializable (dict[str, list[str]]).
"""

from __future__ import annotations

import math
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
        """
        self.agents = list(dict.fromkeys(agents))  # de-dup, preserve order
        self.seed = seed
        self.min_degree = int(min_degree) # min degree for vertex
        self.degree_exponent = float(degree_exponent)
        self.degree_factor = float(degree_factor)
        self.degree_offset = float(degree_offset)
        self.max_degree = max_degree # max degree for vertex
        self.include_self_loops = bool(include_self_loops)

        self._rng = random.Random(seed)
        self.graph = self._generate_connection_graph()

    def _target_degree(self, n: int) -> int:
        if n <= 1:
            return 0
        cap = int(self.max_degree) if self.max_degree is not None else (n - 1)
        cap = max(0, min(n - 1, cap))
        base = int(round((n ** self.degree_exponent) * self.degree_factor + self.degree_offset))
        return max(0, min(cap, max(int(self.min_degree), base)))

    @staticmethod
    def _add_undirected_edge(graph: dict[str, set[str]], a: str, b: str) -> bool:
        if b in graph[a]:
            return False
        graph[a].add(b)
        graph[b].add(a)
        return True

    def _generate_connection_graph(self) -> dict[str, list[str]]:
        """Generate a connected random graph with a guaranteed cycle (closed walk)."""
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

        # 2) Add extra random edges to reach a target degree.
        target = self._target_degree(n)
        if target <= 2:
            return {k: sorted(v) for k, v in graph.items()}

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
        G = nx.Graph()
        for agent, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(agent, neighbor)
        return G