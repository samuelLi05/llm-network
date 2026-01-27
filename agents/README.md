# `agents/`

Agent runtime and prompt/LLM wiring.

## Files

### `network_agent.py`
Primary runtime agent. Each agent:
- Consumes messages from a Redis Stream consumer group
- Chooses whether to respond (via `OrderManager` turn selection)
- Builds a generation context (either embedding-based recommendations or cache fallback)
- Generates a post via OpenAI **or** a local HuggingFace model
- Publishes the post back to Redis and appends to the message cache

Key class

- `class NetworkAgent:`
  - `__init__(
      self,
      id: str,
      init_prompt: str,
      topic: Optional[str] = None,
      stream_name: Optional[str] = None,
      stream_group: Optional[str] = None,
      redis_host: str = "localhost",
      redis_port: int = 6379,
      seed: Optional[int] = None,
      time_manager: Optional[TimeManager] = None,
      order_manager: Optional[OrderManager] = None,
      message_cache: Optional[RedisCache] = None,
      logger: Optional[Logger] = None,
      local_llm: Optional[HuggingFaceLLM] = None,
      llm_service: Optional[LLMService] = None,
      rolling_store: Optional[RollingEmbeddingStore] = None,
      profile_store: Optional[AgentProfileStore] = None,
      topology_tracker: Optional[NetworkTopologyTracker] = None,
      analysis_lock: Optional[asyncio.Lock] = None,
      context_top_k: int = 8,
    )`

Lifecycle
- `await start()`
  - Creates the Redis consumer group.
  - Seeds `AgentProfileStore` from `init_prompt` if embedding/profiles are enabled.
  - Spawns the background consumer loop (`consume_stream`).
- `await stop()`
  - Cancels the background consumer task (important for Jupyter reruns).

Message flow
- `async def consume_stream(self) -> None`
  - Reads from Redis stream.
  - Skips own messages.
  - Uses `OrderManager` (if configured) to decide whether it is allowed to respond.
  - Updates the agent profile with *consumed* messages (fast-path when vectors are available).
  - Calls `generate_response(...)` and publishes.

Generation
- `async def generate_response(self, prompt: Optional[str] = None) -> str`
  - Builds context via `_build_context(...)`:
    - Preferred: embedding-based feed built from rolling store + agent profile vector.
    - Fallback: last-N cache dump per agent.
  - Calls `_generate_from_messages(...)` which routes to:
    - OpenAI (if no local LLM configured)
    - `LLMService.generate(...)` (if local queued)
    - `HuggingFaceLLM.generate(...)` (if local direct)

Repetition guard
- Uses a similarity ratio vs. the agent’s recent posts and may retry generation if too similar.

### `llm_service.py`
Async queue/worker wrapper around the local model to avoid blocking the event loop.

- `class LLMService:`
  - `__init__(self, llm: Optional[HuggingFaceLLM] = None)`
  - `await start()` / `await stop()`
  - `await generate(messages: Sequence[dict], **kwargs) -> str`
  - `await score_label_logprob(messages: Sequence[dict], label: str, **kwargs) -> dict`
  - `await classify_stance(system_prompt: str, user_prompt: str, topic: Optional[str] = None) -> dict`

The worker loop offloads heavy model calls using `asyncio.to_thread(...)`.

### `local_llm.py`
Local HuggingFace model wrapper.

- `class HuggingFaceLLM:`
  - `__init__(..., model_name: Optional[str] = None, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95, quantize: bool = True, use_4bit: bool = True, trust_remote_code: Optional[bool] = None)`
  - `generate(prompt: str | Sequence[dict], max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, stop: Optional[list[str]] = None) -> str`
  - `score_label_logprob(prompt: str | Sequence[dict], label: str, normalize: bool = True) -> dict[str, float]`
  - `classify_stance(system_prompt: str, user_prompt: str, topic: Optional[str] = None) -> dict`

Environment knobs
- `LOCAL_MODEL`: model repo/path
- `HF_TRUST_REMOTE_CODE`: whether to allow custom model code

### `prompt_configs/`
Prompt and topic generation.

- `generate_prompt.py`: `PromptGenerator` selects a shared topic and generates per-agent stance sentences.
- `random_prompt.json`: data file with `topics`, `templates`, and substitution lists.

## Notes on async (Jupyter)
- Prefer top-level `await agent.start()` over `asyncio.run(...)`.
- Always call `await agent.stop()` before rerunning cells to avoid duplicate consumers.
