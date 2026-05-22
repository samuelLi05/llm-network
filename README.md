# LLM Network

## Project Summary

This project investigates whether classical opinion-dynamics models can reproduce and forecast stance trajectories that emerge when LLM agents interact in social-media-style networks. The repository includes a configurable multi-agent social network simulator for generating realistic conversational data, along with implementations of classical opinion-dynamics models such as DeGroot, Friedkin–Johnsen, and homophily-based variants for fitting and forecasting agent behavior. We find that simple extensions, particularly incorporating agent bias toward innate opinions, substantially improve prediction accuracy across discussion topics and network structures.

## Requirements

- Python 3.10+ (recommended)
- Docker & Docker Compose (optional; required to run the included Redis compose file).

Quick setup:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for exact dependency versions and optional extras.

## Start Redis

A Docker Compose configuration for Redis is provided under the `network` folder. From the repository root, start the Redis service with:

```bash
cd network
docker compose up -d
```

## Run the project

- Script: run the main script

```bash
python main.py
```

## Embeddings and Data Collection

By default the code uses the OpenAI Embeddings API (set `USE_OPENAI_EMBEDDINGS=True` and provide `OPENAI_API_KEY`). A local sentence-transformers option is available (`USE_OPENAI_EMBEDDINGS=False`). 

Embedded posts and related artifacts are written to Redis during runs and persisted as pipeline logs under `logs/`.

Log folders and contents:
- **logs/network_logs/**: Raw network event records including agent posts and embedding metadata for each post
- **logs/topology_logs/**: Topology snapshots including network graph structuer and agent opinions. 
- **logs/stance_logs/**: Agent initialization stance labels
- **logs/agent_config_logs/**: Agent prompt initializations

## Tunable parameters

The detailed, up-to-date tunable parameters live in the source files (these are the canonical locations):

- Network and global flags: [main.py](main.py)
- Agent configuration: [agents/network_agent.py](agents/network_agent.py)
- Embedding and recommendation knobs: [controller/stance_analysis/rolling_embedding_store.py](controller/stance_analysis/rolling_embedding_store.py)
- Ordering configuration: [controller/order_manager.py](controller/order_manager.py)
- Agent post timing configurations: [controller/time_manager.py](controller/time_manager.py)

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
	- `time_manager.py`: Wait times for agent posts.
	- `order_manager.py`: Logic to choose which agent replies next.
	- `stance_worker.py`: Optional batch worker for stance labeling.
	- `stance_analysis/`: Embedding analyzer, rolling store, profile store, and topology tracking.

- [network/](network/) : Redis helpers and compose files.
	- `cache.py`: Async Redis cache wrapper for storing agent message histories.
	- `stream.py`: Redis stream helper used by agents for pub/sub and cleanup.
	- `docker-compose.yml`: Quick way to run Redis locally for demos.

- [modeling/](modeling/) : Data preparation, model fitters, ranking scripts, and plotting notebooks.
	- `models/`: Model implementations (fixed-graph and adjacency-based fitters) and helpers.
	- `generate_model_rankings.py`: Batch runner for fitting and exporting model rankings.
	- `plots.ipynb`: Notebook for visualizing fit results.
	- `plot_utils.py`: Helper scripts for plotting.

- [logs/](logs/) : Runtime logs and topology outputs.
	- `network_logs/`, `topology_logs/`, `stance_logs/`, `agent_config_logs/` for collected artifacts.

- [tests/](tests/) : Diagnostic and integration-style tests you can run as scripts.
	- `agent_profile_test.py`, `embedding_reco_test.py`, `stance_test.py` — tests to gauge stance analysis.

- `main.py` : The primary script that mirrors the notebook runner (full end-to-end run).
- `requirements.txt` : Python dependencies.


