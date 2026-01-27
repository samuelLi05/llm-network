# logs/

Runtime artifacts produced by the network runner.

There are two logging “planes” in this project:
- **Console logs** (Python `logging` via `console_logger`) for status/debug.
- **File logs** (threaded writers) for publish events, agent configs, stances, and topology snapshots.

Because the network is async and long-running, file logging is done on background threads to avoid blocking the event loop.

## Folders

### `logs/network_logs/`
Plain-text `.log` files containing agent publish events.

Typical line format:

```
[YYYY-mm-dd HH:MM:SS] PUBLISH agent=agent_3 | <message>
```

These logs are good for answering: “what did the conversation look like over time?”

### `logs/agent_config_logs/`
Plain-text `.log` files that capture each agent’s initial configuration/prompt at startup.

This is useful for reproducibility: it makes it possible to reconstruct *exact* agent settings for a given run.

### `logs/stance_logs/`
Plain-text `.log` files used by the optional stance worker (`controller/stance_worker.py`) to record stance analysis artifacts.

### `logs/topology_logs/`
JSONL files (`.jsonl`) of topology snapshots.

Each line is one JSON object (a snapshot). The intended usage is:
- load the file later
- reconstruct a similarity graph over time
- verify that the recommender/profile pipeline is producing meaningful structure

## Files

### `logger.py`
Defines the console logger and the main publish/config file logger.

#### Console logger

```python
console_logger = logging.getLogger("llm_network")
```

- Used by multiple modules for informational output.
- Configured with a stream handler and a simple formatter.

#### Class: `Logger`
Handles **publish events** and **config snapshots**.


**Threading model**
- A daemon thread (`LoggerThread`) consumes a `queue.Queue` and writes to a single file handle.
- `async_log_publish()` and `async_put()` use `loop.run_in_executor(...)` to avoid blocking the async loop.

**File naming**
- `network_logs/log_YYYYmmdd-HHMMSS_{num_agents}.log`
- `agent_config_logs/log_YYYYmmdd-HHMMSS_{num_agents}_agent_configs.log`
- `stance_logs/log_YYYYmmdd-HHMMSS_{num_agents}_stance_configs.log`

Parameter provenance:
- `num_agents` comes from `NUM_AGENTS` in `main.py` / notebook config.

### `topology_logger.py`
Defines `TopologyLogger`, a threaded JSONL writer.

**Threading model**
- Similar to `Logger`, but writes JSON objects (one per line).

**File naming**
- `topology_logs/topology_YYYYmmdd-HHMMSS.jsonl`

## How to interpret topology JSONL
The exact snapshot schema is produced by `controller/stance_analysis/network_topology.py` (and whatever calls it), but conceptually snapshots include:
- nodes: agents with stance/topic metrics
- edges: similarity links between agent vectors
- timestamp / step counters