# agents/prompt_configs/

Prompt/topic configuration used to initialize agents.

This folder is responsible for:
- picking a **shared discussion topic** for a run
- generating **distinct agent prompts** that all refer to that same topic

The generated prompts are typically consumed by the agent wiring in `main.py` / the notebook runner.

## Files

### `random_prompt.json`
A structured JSON “prompt bank”.

Key fields:
- `topics: list[str]` — candidate global topics (one chosen per run unless forced).
- `templates: list[str]` — prompt templates. Templates may contain placeholders like `{topic}` and category placeholders.
- Category word lists (e.g. `adjective`, `nouns_technology`, …): lists of replacement tokens.

Placeholder rules (as used by `PromptGenerator`):
- `{topic}` is always replaced with the chosen topic.
- For any other key `k` in the JSON with a list value, `{k}` in a template may be replaced with a random element from that list.

Practical tip:
- If you add a new placeholder to `templates`, you must also add a matching key list in the JSON for it to get substituted.

### `generate_prompt.py`
Defines `PromptGenerator`, which loads `random_prompt.json` and produces per-agent prompts.

#### Class: `PromptGenerator`

```python
class PromptGenerator:
    def __init__(self, topic: str | None = None): ...

    def generate_single_prompt(self) -> str: ...
    def generate_multiple_prompts(self, n: int) -> list[str]: ...
    def get_topic(self) -> str: ...
```

Behavior:
- If `topic` is not provided, a random one is chosen from `topics`.
- `generate_single_prompt()` picks a random template and performs placeholder substitution.
- `generate_multiple_prompts(n)` produces `n` prompts that share the same `topic`.

Parameter provenance:
- `topic` may be forced by the runner (e.g., for experiments/tests) or left random.

## How it fits into the system
- The **topic** chosen here is threaded through:
  - agent prompt text
  - embedding analyzers (`EmbeddingAnalyzer(topic)`) and rolling stores (`RollingEmbeddingStore(topic=topic, ...)`)
  - stance and topic similarity scoring

Keeping the topic consistent across components is what makes the “stance axis” and topology meaningful.
