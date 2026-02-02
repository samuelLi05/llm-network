# LLM Network

## Project Summary

This project provides a small multi-agent LLM network that uses Redis streams for messaging between agents. Each `NetworkAgent` listens on a Redis stream, generates responses using the OpenAI client, and publishes messages back to the stream. The repository includes a simple runner script and a Jupyter notebook to start and inspect conversations.

## Requirements

- Python 3.10+ recommended
- Docker & Docker Compose

Install Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(See [requirements.txt](requirements.txt))

## Start Redis (Docker)

A Docker Compose file for Redis is included under the `network` folder. From the repository root, start the Redis service:

```bash
# Start Redis in the background (run from repository root)
cd network
docker compose up -d
```

Note: If you prefer a single container run without compose, you can start Redis directly:

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

## Run the project

There are two primary ways to run a conversation:

- Script: run the main script

```bash
python main.py
```

## Tunable Parameters

The following are tunable parameters for the LLM-network. 
#### Network Configuration in [main.py](./main.py)
- `NUM_AGENTS` - Number of agents to instantiate in the network (controls scale).
- `STREAM_NAME` - Redis stream key used for agent publish/consume.
- `REDIS_HOST` - Host address for the Redis server.
- `REDIS_PORT` - Port for the Redis server.
- `RUN_DURATION_SECONDS` - How long the demo run should sleep before shutdown.
- `USE_LOCAL_LLM` - If true, use the local HuggingFace LLM queue instead of OpenAI.

#### LLM Based Stance Analysis in [main.py](./main.py)
(Not critical for the network to run; off by default)
- `ENABLE_STANCE_WORKER` - Toggle the background stance-labeling batch worker.
- `STANCE_BATCH_SIZE` - Number of items processed per stance-worker batch.
- `STANCE_BATCH_INTERVAL` - Seconds between stance-worker batch runs.

#### Recommendation System Configurations in [main.py](./main.py)
- `ENABLE_EMBEDDING_CONTEXT` - Enable embedding-based context/recommender (requires `OPENAI_API_KEY`).
- `ROLLING_STORE_MAX_ITEMS` - Max items to retain in the rolling embedded store.
- `CONTEXT_TOP_K` - Number of top-k retrieved items to include as generation context for agent (recommendation feed).
- `PROFILE_WINDOW_SIZE` - Sliding window size for agent interaction history used to build profiles.
- `PROFILE_SEED_WEIGHT` - Weight applied to the agent's init prompt when computing profile vectors.
- `TOPOLOGY_LOG_INTERVAL` - Seconds between topology snapshots written to logs.

#### Agent parameters (defaults in `agents/network_agent.py`)
- `log_recommendations` - Enable logging of top-k recommendations used for context.
- `log_reco_debug` - Extra verbose debug logging for recommendation internals.
- `log_reco_max_items` - Max number of recommendation entries to include in debug logs.
- `regen_on_repeat` - Retry generation if output is too similar to recent posts.
- `regen_max_attempts` - Maximum retries when `regen_on_repeat` is enabled.
- `regen_similarity_threshold` - Similarity threshold above which output is considered a repeat.
- `regen_history_last_n` - Number of recent posts used to check for repetition.

#### Latent representation / recommend() knobs (`controller/stance_analysis/rolling_embedding_store.py`)
- `top_k` - How many nearest neighbors to return for recommendations.
- `min_topic_similarity` - Minimum topic-similarity score to consider an item relevant.
- `min_strength` - Minimum opinion strength to include an item in candidates.
- `exclude_sender_id` - Sender id to exclude from recommendations (e.g., current author) or blocking.
- `alpha` - Weight for stance-distance term in composite distance.
- `beta` - Weight for strength-distance term in composite distance.
- `gamma` - Weight for semantic distance (embedding cosine) in composite distance.

#### Agent ordering configs (`controller/order_manager.py`)
- `ordering_mode` - Selection policy: `random` or `topology` (profile-based).
- `echo_probability` - Probability to choose a similar (echo) responder vs. contrasting one.
- `fairness_tau_s` - Time constant (seconds) used to prefer agents that haven't spoken recently.
- `cooldown_s` - Per-agent cooldown (seconds) to avoid immediate re-selection.
- `temperature` - Softmax temperature for probabilistic selection among scored candidates.
- `sim_weight` - Weight applied to similarity when scoring candidates.
- `fair_weight` - Weight given to fairness/recency in the selection score.
- `extremeness_penalty` - Penalty for agents far from the population centroid (discourages extremes).
- `explore_epsilon` - Small probability of pure random exploration when picking next agent.

## Notes on Jupyter & Python versions

- You may need to install `ipykernel` to run the notebook with your environment:

```bash
pip install ipykernel
python -m ipykernel install --user --name=llm-network-env
```

- Depending on your Python distribution and packaging, you might need `setuptools`:

```bash
pip install setuptools
```

## Cleanup

When stopping, tear down the Redis container (if started with compose):

```bash
cd network
docker compose down
```

## Repository Structure

Top-level files and folders (click to open):

- [agents/](agents/) : Agent implementations and prompt generation.
	- `network_agent.py`: Agent class that listens on the Redis stream and publishes responses.
	- `llm_service.py` / `local_llm.py`: Local LLM wrapper and async queuing helper.
	- `prompt_configs/`: Templates and topic data used to generate agent prompts.

- [controller/](controller/) : Orchestration and analysis controllers.
	- `time_manager.py`: Rate-limiting and publish locks.
	- `order_manager.py`: Logic to choose which agent replies next.
	- `stance_worker.py`: Optional batch worker for stance labeling.
	- `stance_analysis/`: Embedding analyzer, rolling store, profile store, topology tracking, and vector ops.

- [network/](network/) : Redis helpers and compose files.
	- `cache.py`: Async Redis cache wrapper for storing agent message histories.
	- `stream.py`: Redis stream helper used by agents for pub/sub and cleanup.
	- `docker-compose.yml`: Quick way to run Redis locally for demos.

- [logs/](logs/) : Runtime logs and topology outputs.
	- `network_logs/`, `topology_logs/`, `stance_logs/`, `agent_config_logs/` for collected artifacts.

- [tests/](tests/) : Diagnostic and integration-style tests you can run as scripts.
	- `agent_profile_test.py`, `embedding_reco_test.py`, `stance_test.py` — test with more in-depth examples to gauge stance analysis.

- `main.py` : The primary script that mirrors the notebook runner (full end-to-end run).
- `llm_network.ipynb` : Notebook version of the runner (interactive, split into steps for inspection).
- `requirements.txt` : Python dependencies.
- `README.md` : This file.


