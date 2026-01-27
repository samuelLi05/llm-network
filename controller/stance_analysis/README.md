# `controller/stance_analysis/`

Embedding-based stance representation, agent profiles, rolling corpus indexing, and topology snapshots.

This folder is the “math core” of the project.

## Files

### `vector_ops.py`
Small numerical helpers used across the recommender/profile code.

Core functions
- `to_np(vec: list[float] | np.ndarray, *, dtype=np.float32) -> np.ndarray`
- `mean_vector(vectors: Iterable[list[float] | np.ndarray]) -> list[float]`
- `l2_normalize(vec: list[float] | np.ndarray, *, eps: float = 1e-12) -> list[float]`
- `dot_similarity_normalized(a, b) -> float` (assumes inputs already normalized)

Normalization
- For a vector $v$:

$$\hat{v} = \frac{v}{\lVert v \rVert_2}$$

If vectors are unit-length, cosine similarity equals dot product:

$$\cos(\hat{a}, \hat{b}) = \hat{a} \cdot \hat{b}$$

---

### `embedding_analyzer.py`
Defines a topic-specific embedding “frame” using anchor groups (pro/anti/neutral). Produces:
- `topic_similarity`
- `stance_score` (projection onto a pro-vs-anti axis)
- `strength` (how opinionated + on-topic)

Key class
- `class EmbeddingAnalyzer:`
  - `__init__(topic: str, local_llm: Optional[LocalLLM] = None, llm_service: Optional[LLMService] = None)`
  - `await embed_and_score(prompt: str, *, include_vector: bool = False) -> Optional[dict]`
  - `await score_vector(vector: list[float], *, include_vector: bool = False) -> Optional[dict]`

Parameters (env)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `EMBEDDING_SIM_TEMPERATURE` (softmax temperature used by `_softmax`, if you use it)

Anchor construction
- For each anchor group (pro/anti/neutral), embed its texts and compute the centroid:

$$c_g = \frac{1}{|G|} \sum_{x \in G} \hat{e}(x)$$

Then normalize:

$$\hat{c}_g = \frac{c_g}{\lVert c_g \rVert_2}$$

Stance axis
- Compute a pro-vs-anti direction:

$$a = \widehat{(\hat{c}_{pro} - \hat{c}_{anti})}$$

Then project an embedded prompt vector $\hat{p}$:

Let $s$ denote the stance score (corresponds to `stance_score` in code):

$$s = \hat{p} \cdot a$$

Topic similarity

Let $\tau$ denote topic similarity (corresponds to `topic_similarity` in code):

$$\tau = \hat{p} \cdot \hat{t}$$

Where $\hat{t}$ is the normalized embedding of the topic string.

Strength heuristic

Let $r$ denote strength (corresponds to `strength` in code):

$$r = \max(0, \tau) \cdot (1 - \max(0, \hat{p} \cdot \hat{c}_{neutral}))$$

Where the second term penalizes “neutral-ish” vectors.

---

### `agent_profile_store.py`
Maintains a **sliding-window** per-agent embedding profile (seed prompt + interactions).

Data model
- `@dataclass AgentProfile:`
  - `agent_id: str`
  - `vector: list[float]` (normalized)
  - `seed_vector: Optional[list[float]]`
  - `window: list[dict]` where each entry contains `{vector, weight, type, ts, meta}`

Key class
- `class AgentProfileStore:`
  - `__init__(redis: Optional[AsyncRedis] = None, *, window_size: int = 200, seed_weight: float = 5.0, authored_weight: float = 1.0, consumed_weight: float = 0.7, key_prefix: str = "agent_profile:")`
  - `await ensure_initialized(agent_id: str, *, seed_text: str, topic_for_embedding: str) -> AgentProfile`
  - `await add_interaction_vector(agent_id: str, *, vector: list[float] | np.ndarray, interaction_type: str, ts: Optional[float] = None, metadata: Optional[dict] = None) -> AgentProfile`
  - `await add_interaction(agent_id: str, *, text: str, interaction_type: str, topic: str, ts: Optional[float] = None, metadata: Optional[dict] = None) -> AgentProfile`
  - `await get_agent_topic_view(agent_id: str, *, topic: str) -> Optional[dict]`

Profile equation
Let:
- $s$ be the normalized seed vector
- $w_s$ be `seed_weight`
- Window entries $i=1..N$ have normalized vectors $x_i$ and weights $w_i$:
  - authored: $w_i = w_a$ (corresponds to `authored_weight`)
  - consumed: $w_i = w_c$ (corresponds to `consumed_weight`)

Maintain a running weighted sum:

$$S = \sum_{i=1}^{N} w_i x_i$$

Then the profile vector is:

$$v = \widehat{(S + w_s s)}$$

Sliding-window update
- When the window grows beyond `window_size`, the oldest entry is subtracted from $S$.

Where parameters come from
- `seed_weight`, `window_size` are set when constructing `AgentProfileStore` (in `main.py`).
- `authored_weight`, `consumed_weight` use defaults unless overridden in `AgentProfileStore(...)`.

Where it is used
- `NetworkAgent.start()` seeds the profile via `ensure_initialized(...)`.
- `NetworkAgent` updates profiles on consumed/published messages (fast-path uses `add_interaction_vector`).
- `OrderManager` uses profile vectors for topology-based next-responder selection.

---

### `rolling_embedding_store.py`
Stores embedded posts (rolling corpus) and performs nearest-neighbor style retrieval.

Data model
- `@dataclass(frozen=True) EmbeddedPost:` includes `vector`, `topic_similarity`, `stance_score`, `strength`, and `metadata`.

Key class
- `class RollingEmbeddingStore:`
  - `__init__(topic: str, analyzer: EmbeddingAnalyzer, redis_cache: Optional[RedisCache] = None, redis_key: Optional[str] = None, max_items: int = 2000)`
  - `await add(text: str, *, id: Optional[str] = None, created_at: Optional[float] = None, metadata: Optional[dict] = None, persist: bool = True) -> EmbeddedPost`
  - `await add_scored_vector(*, id: str, text: str, vector: list[float] | np.ndarray, scored: dict, created_at: Optional[float] = None, metadata: Optional[dict] = None, persist: bool = True) -> EmbeddedPost`
  - `await load_from_redis(last_n: int = 500) -> int`
  - `await recommend(query_text: str, *, top_k: int = 10, min_topic_similarity: float = 0.15, min_strength: float = 0.05, exclude_sender_id: Optional[str] = None, alpha: float = 1.0, beta: float = 0.5, gamma: float = 1.0) -> list[dict]`
  - `await recommend_for_agent_vector(*, agent_vector: list[float], agent_stance_score: float, agent_strength: float, top_k: int = 10, ...) -> list[dict]`

Composite distance function
For a query $q$ and item $i$:
- $\Delta stance = |s_q - s_i|$
- $\Delta strength = |r_q - r_i|$ (strength)
- semantic term uses cosine distance for normalized vectors: $1 - (v_q \cdot v_i)$

$$d(q,i) = \alpha \cdot |s_q - s_i| + \beta \cdot |r_q - r_i| + \gamma \cdot (1 - v_q \cdot v_i)$$

Lower $d$ means more similar.

Candidate filtering
- Items must satisfy:
  - `topic_similarity >= min_topic_similarity`
  - `strength >= min_strength`
  - optionally `sender_id != exclude_sender_id`

Where parameters come from
- Defaults live in the method signatures.
- `main.py` wires `CONTEXT_TOP_K` into `NetworkAgent(context_top_k=...)`.

Where it is used
- `NetworkAgent._build_context(...)` calls `recommend_for_agent_vector(...)` using its stored profile vector.

---

### `network_topology.py`
Produces periodic snapshots of the agent network as a graph.

- `class NetworkTopologyTracker:`
  - `__init__(*, topic: str, profile_store: AgentProfileStore, redis_cache: Optional[RedisCache] = None, redis_key: Optional[str] = None, min_edge_similarity: float = 0.15, update_interval_s: float = 10.0)`
  - `await snapshot(agent_ids: list[str]) -> dict`
  - `await maybe_update(agent_ids: list[str], *, force: bool = False) -> Optional[dict]`

Topology math
- Stack agent vectors into a matrix $M \in \mathbb{R}^{N \times D}$.
- Similarity matrix:

$$W = M M^T$$

Edges are emitted for pairs $(i,j)$ where $W_{ij} \ge \theta$, where $\theta$ corresponds to `min_edge_similarity`.

Node features
- Each node’s `stance_score`, `strength`, and `topic_similarity` is computed by scoring its profile vector through `EmbeddingAnalyzer.score_vector(...)`.

---

### `baseline_analyzer.py`
LLM-based stance labeling + SBERT similarity.

- Used by `StanceWorker` and `tests/stance_test.py`.
- This is separate from the embedding-based recommender; it’s “labeling/analysis”, not the retrieval frame.
