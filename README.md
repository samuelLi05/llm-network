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

## Validating the embedding-based recommender (agent profiles + rolling store)

When enabled, agents build their generation context from the rolling embedded corpus
using their precomputed agent profile vectors (sliding window + seed prompt).

### Enable proof logging

```bash
export OPENAI_API_KEY=...   # required for embeddings
export LOG_RECOMMENDATIONS=1
export TOPOLOGY_LOG_INTERVAL_S=10
python main.py
```

What you should see:

- Console logs like: `Agent agent_3 using reco feed: {...}` showing the top-k retrieved items
	(ids/distances/sender_ids) used to build the FEED context.
- A JSONL file created under `logs/topology_logs/` containing periodic network topology snapshots
	(nodes=agents with stance metrics, edges=similarity links). This is intended for later modeling.

If `OPENAI_API_KEY` is missing, the system automatically falls back to the older cache-based
"last N messages" context builder.

### Local LLM (Quantized HF) Option

To isolate GPU inference and avoid blocking the asyncio event loop, this project can route all local model calls through a shared `LLMService` queue.

Set environment variables before running:

```bash
export USE_LOCAL_LLM=true
export LOCAL_MODEL="Qwen/Qwen3-VL-8B-Instruct"
export HF_TRUST_REMOTE_CODE=true
```

Notes:
- The local LLM uses 4-bit quantization by default when `bitsandbytes` is installed.
- If you need to inspect weights, you can load the model via the in-process `HuggingFaceLLM` class.
- For higher concurrency, consider running a separate inference server (vLLM/TGI) and swapping the client call.

- Notebook: open the Jupyter notebook and run the cells

Open [llm_network.ipynb](llm_network.ipynb) in Jupyter or VS Code and run the cells. The notebook uses nested event loop handling via `nest_asyncio` so it can be run inside Jupyter.

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
