# tests/

These are “test-like” scripts: lightweight diagnostics you can run directly with Python.

They are intentionally written as executable scripts (each has an `async def main()` and `asyncio.run(main())`) rather than using `pytest`.

## Running

From repository root:

```bash
python tests/embedding_reco_test.py
python tests/agent_profile_test.py
python tests/stance_test.py
```

## Files

### `embedding_reco_test.py`
Purpose: smoke test the embedding-based rolling recommender (`RollingEmbeddingStore.recommend`).

Flow:
1. Load topics from `agents/prompt_configs/random_prompt.json`.
2. Pick a topic (forced to `vaccines` by default for stronger signal).
3. Build a labeled corpus of posts (pro/anti/neutral) and add them to an in-memory rolling store.
4. Run a handful of “query” texts and print:
   - mean stance/strength/topic similarity
   - label distribution of retrieved items
   - top-k items with distances

### `agent_profile_test.py`
Purpose: validate **agent profile vectors** + “recommend for agent vector” path.

Flow:
1. Build a corpus in `RollingEmbeddingStore`.
2. Create an `AgentProfileStore` with a sliding window and seed weight.
3. For three personas (pro/anti/neutral):
   - `ensure_initialized(agent_id, seed_text=..., topic_for_embedding=...)`
   - add interaction texts as both `consumed` and `authored`
   - load the profile and ensure the window cap is respected
   - compute the agent’s topic view (stance/strength/topic similarity)
   - recommend items conditioned on the **agent vector**

### `stance_test.py`
Purpose: compare stance classification approaches:
- OpenAI logprob classification (if `OPENAI_API_KEY` is set)
- Local LLM classification via `HuggingFaceLLM` + `LLMService`
- Optional SBERT similarity baseline using `sentence_transformers`

Flow:
1. Sample `NUM_TOPICS` topics.
2. For each topic:
   - generate a few synthetic “social posts” for each stance label A/B/C
   - run OpenAI and/or Local classification
   - compute SBERT similarity to stance prototypes
   - print results and optionally a pairwise similarity matrix
