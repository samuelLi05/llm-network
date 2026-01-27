# `controller/`

Orchestration layer: time gating, turn selection, and optional stance worker.

## Files

### `time_manager.py`
Global publish rate limiting.

- `class TimeManager:`
  - `__init__(global_interval: float = 3.0, default_interval: float = 3.0, intervals: Optional[dict[str, float]] = None)`
  - `publish_lock(agent_id: str) -> TimeManager.PublishLock`

Concept
- Enforces a **global** minimum time between publishes across *all* agents.
- Used by `NetworkAgent.publish_message(...)`:
  - `async with time_manager.publish_lock(agent_id): ...`

Time equation
- Let $t$ be current monotonic time and $t_g$ be the last global publish time.

$$\text{wait} = \max(0, \text{global\_interval} - (t - t_g))$$

When the publish lock exits, the manager records the new $t_g$.

### `order_manager.py`
Selects the next agent to respond and stores that decision in Redis.

- `class OrderManager:`
  - `__init__(agents: list[NetworkAgent], message_cache: RedisCache, profile_store: Optional[AgentProfileStore] = None, redis_host: str = "localhost", redis_port: int = 6379)`
  - `async select_and_store_next_responder(exclude_agent_id: Optional[str] = None) -> str`
  - `get_designated_responder() -> Optional[str]`
  - `clear_designated_responder() -> None`
  - `is_my_turn(agent_id: str) -> bool`

Redis coordination
- The next responder is written to Redis key `NEXT_RESPONDER_KEY` (see source).
- Each consumer reads this key to decide whether it is allowed to respond.

Selection modes
- `ordering_mode = "random"`:
  - Chooses uniformly at random from candidates.
- `ordering_mode = "topology"` (requires `profile_store`):
  - Scores candidates using agent profile-vector similarity to the reference agent.

Topology-weighted scoring
Let:
- $s_i$ = similarity between candidate $i$ and reference agent (dot product of normalized vectors)
- $f_i$ = fairness term in $[0,1]$ based on recency
- $e_i$ = extremeness term (distance from centroid), typically $e_i = 1 - \cos(\text{centroid}, v_i)$

With weights (`sim_weight`, `fair_weight`, `extremeness_penalty`) the score is:

$$\text{score}_i = w_s \cdot s_i' + w_f \cdot f_i - w_e \cdot e_i$$

Where $s_i'$ is either $s_i$ (echo reply) or $-s_i$ (contrast reply), controlled by `echo_probability`.

Softmax sampling
Given scores $z_i$ and temperature $T$:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Then choose next responder by sampling from $p$.

Where parameters come from
- In this repo version, several knobs are hard-coded defaults in `__init__` (see file).
- Earlier versions may read these from environment variables; check the file if you change this.

### `stance_worker.py`
Optional background worker for stance classification + similarity batching.

Behavior
- Consumes the same Redis stream as agents (separate consumer group).
- Writes analysis results into Redis with prefix `stance:` (via `RedisCache`).
- Runs per-message stance classification (OpenAI/local).
- Batches similarity computation every `batch_size` messages or `batch_interval` seconds.

Related modules
- `controller/stance_analysis/` contains the embedding-based recommender and topology tracker.
