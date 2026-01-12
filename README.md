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

- Notebook: open the Jupyter notebook and run the cells

Open [main_runner.ipynb](main_runner.ipynb) in Jupyter or VS Code and run the cells. The notebook uses nested event loop handling via `nest_asyncio` so it can be run inside Jupyter.

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