# network/

Redis connectivity and messaging utilities.

This folder is the “infra glue” of the project:
- `stream.py` handles Redis Streams publish/consume + consumer group cleanup.
- `cache.py` provides an async Redis-backed cache for per-agent history and recent global messages.
- `docker-compose.yml` starts a local Redis service for demos.

## Files

### `docker-compose.yml`
Starts a Redis 7 instance (AOF enabled) bound to `localhost:6379`.

Common lifecycle:

```bash
cd network
docker compose up -d
# ... run main.py / notebook ...
docker compose down
```

### `stream.py`
Defines `RedisStream`, a thin wrapper around `redis.Redis` (sync client) but exposed via `async def` methods for compatibility with the project’s async code.

#### Class: `RedisStream`

```python
class RedisStream:
    def __init__(self, host='localhost', port=6379, db=0): ...

    async def create_consumer_group(self, stream_name, group_name): ...
    async def publish_message(self, stream_name, message_data: dict): ...
    async def consume_messages(self, stream_name, group_name, consumer_name): ...

    async def cleanup_stream(
        self,
        stream_name: str,
        num_groups=None,
        group_prefix: str = 'group_',
    ): ...
```

#### Redis Stream model
- Stream key: `stream_name` (e.g. `STREAM_NAME` in `main.py`).
- Consumer group key: `group_name` (e.g. `group_1`, `group_2`, …).
- Consumer name: `consumer_name` (per-agent name).

Consumption uses:
- `XREADGROUP group consumer {stream: '>'} COUNT 1 BLOCK 1000`
- `XACK stream group message_id` after yielding the message

#### Cleanup semantics
`cleanup_stream()` attempts to:
1. Destroy consumer groups (either by enumerating with `XINFO GROUPS` or by constructing `{group_prefix}{i}` when `num_groups` is known).
2. Delete the stream key.

This is primarily useful for reruns in notebooks/tests to avoid a buildup of stale groups.

#### Important async note
Even though methods are `async`, the underlying Redis client is synchronous. In practice this is usually OK for a small demo, but if you see event loop stalls under load, the next step would be switching `RedisStream` to `redis.asyncio`.

### `cache.py`
Defines `RedisCache`, an **async** cache backed by `redis.asyncio.Redis`.

This cache is used as a lightweight persistence layer for:
- per-agent response history lists
- a “global recency” view via a sorted set

#### Class: `RedisCache`

```python
class RedisCache:
    def __init__(self, host: str='localhost', port: int=6379, db: int=1, prefix: str='cache:'): ...

    async def get_responses(self, agent_id: str, last_n: int | None = None) -> list[Any]: ...
    async def get_random_responses(self, sample_size: int) -> list[Any]: ...
    async def get_last_responses(self, sample_size: int) -> list[Any]: ...

    async def append_response(self, agent_id: str, value: Any, expire: int | None = None): ...

    async def clear(self, agent_id: str) -> None: ...
    async def clear_all(self) -> None: ...
    async def close(self) -> None: ...
```

#### Key layout
With `prefix='cache:'`:
- Per-agent list key: `cache:{agent_id}`
  - `RPUSH` appends JSON-serialized items (dict/list) or `str(value)`.
  - `LRANGE` retrieves full history or last `N` items.
- Global ordered set: `cache:timestamps`
  - `ZADD` stores `{payload: timestamp}`.
  - `ZREVRANGE` retrieves most recent payloads.

#### Parameter provenance
- `host`, `port`: come from `REDIS_HOST`, `REDIS_PORT` in `main.py` / notebook config.
- `db`: defaults to `1` (separate from stream DB=0), but is configurable.
- `prefix`: defaults to `cache:`.

## How it fits into the system
- Agents publish to a Redis Stream for “live” message passing.
- Agents also append outputs to the Redis cache for:
- Cache stores all relevant agent data as a global data structure (emebdding store, past messages, etc)