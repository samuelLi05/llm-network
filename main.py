"""Main runner for the LLM Network.

Initializes multiple NetworkAgents with shared topic prompts, integrates
TimeManager, OrderManager, RedisCache, and Logger for coordinated messaging.
"""

import asyncio
import os
from agents.network_agent import NetworkAgent
from agents.llm_service import LLMService
from agents.local_llm import HuggingFaceLLM
from agents.prompt_configs.generate_prompt import PromptGenerator
from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.rolling_embedding_store import RollingEmbeddingStore
from controller.stance_analysis.agent_profile_store import AgentProfileStore
from controller.stance_analysis.network_topology import NetworkTopologyTracker
from controller.time_manager import TimeManager
from controller.order_manager import OrderManager
from controller.stance_worker import StanceWorker
from network.cache import RedisCache
from network.stream import RedisStream
from logs.logger import Logger, console_logger
from logs.topology_logger import TopologyLogger

# Configuration
NUM_AGENTS = 10
STREAM_NAME = "agent_stream"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
RUN_DURATION_SECONDS = 500
USE_LOCAL_LLM = True
ENABLE_STANCE_WORKER = False
STANCE_BATCH_SIZE = 5
STANCE_BATCH_INTERVAL = 30

ENABLE_EMBEDDING_CONTEXT = True
ROLLING_STORE_MAX_ITEMS = 2000
CONTEXT_TOP_K = 8
PROFILE_WINDOW_SIZE = 50
PROFILE_SEED_WEIGHT = 5.0

TOPOLOGY_LOG_INTERVAL_S = 5.0

initial_prompt_template = (
   "You are participating in a social-media-style discussion about {topic}." \
   "The sentence {unique_prompt} is your fixed stance and is authoritative and exhaustive. Write entirely from the worldview, assumptions, tone, values, and constraints it defines; it fully determines what you believe, how you speak, and what claims you are willing to make." \
   "Produce a short, attention-grabbing post that hooks readers, makes a clear and strong claim aligned with that grounding, and invites engagement (likes, replies, shares)." \
   "Be concise, bold, and evocative. Use a distinct memorable opening line, assertive language, and a direct call-to-action every time. Emulate authentic social media posts." \
   "Make sure posts are distinct, do not copy formatting and language of previous posts, instead contradict any claims that oppose your fixed stance"
   "Do not introduce outside viewpoints, neutral framing, balance, or meta-commentary. Do not soften or qualify claims unless explicitly required by the authoritative sentence. Never refer to yourself as an agent, AI, or participant in a debate."
)

async def main():
    console_logger.info("Starting LLM Network...")

    llm_service = None
    if USE_LOCAL_LLM:
        console_logger.info("Using local LLM service (quantized HF model).")
        local_llm = HuggingFaceLLM()
        llm_service = LLMService(local_llm)
        await llm_service.start()
    else:
        local_llm = None

    # 1. Initialize shared components
    prompt_generator = PromptGenerator()
    topic = prompt_generator.get_topic()
    console_logger.info(f"Shared discussion topic: {topic}")

    # Generate unique prompts for each agent (same topic, different wording)
    agent_prompts = prompt_generator.generate_multiple_prompts(NUM_AGENTS)

    # Create dictionary of agents and their initial prompts for logging
    agent_configs = {}
    for i in range(NUM_AGENTS):
        agent_id = f"agent_{i+1}"
        init_prompt = initial_prompt_template.format(
            topic=topic,
            unique_prompt=agent_prompts[i]
        )
        agent_configs[agent_id] = init_prompt

    # Initialize TimeManager with 3-second global interval
    time_manager = TimeManager(global_interval=3.0)

    # Initialize RedisCache for storing agent message histories
    message_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT)

    # Embedding-based stores (optional)
    embed_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT, prefix="embed:")
    topology_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT, prefix="topology:")

    rolling_store = None
    profile_store = None
    topology_tracker = None
    analysis_lock = asyncio.Lock()
    topology_logger = None

    if ENABLE_EMBEDDING_CONTEXT and bool(os.getenv("OPENAI_API_KEY")):
        analyzer = EmbeddingAnalyzer(topic)
        rolling_store = RollingEmbeddingStore(
            topic=topic,
            analyzer=analyzer,
            redis_cache=embed_cache,
            max_items=ROLLING_STORE_MAX_ITEMS,
        )
        # Load previous corpus (if any)
        loaded = await rolling_store.load_from_redis(last_n=min(500, ROLLING_STORE_MAX_ITEMS))
        console_logger.info(f"Loaded {loaded} embedded posts from Redis.")

        # Cold-start: seed the latent space with strong opposing posts
        if loaded == 0:
            seed_texts: list[tuple[str, str]] = []  # (side, text)
            # Use the analyzer's anchor groups as stable reference seeds
            for side, texts in analyzer.anchor_groups.items():
                for t in texts:
                    seed_texts.append((side, t))

            # Add a few extra high-contrast, attention-grabbing seeds
            seed_texts.extend(
                [
                    ("pro", f"Enough dithering. {topic} is non-negotiable — we should expand it now."),
                    ("pro", f"If you're against {topic}, you're choosing stagnation. Push it through."),
                    ("anti", f"Wake up: {topic} is a harmful mistake. Stop pretending it's 'progress'."),
                    ("anti", f"{topic} is a disaster in slow motion. Reject it before it spreads."),
                ]
            )

            for i, (side, text) in enumerate(seed_texts):
                await rolling_store.add(
                    text,
                    id=f"seed:{i}",
                    metadata={"sender_id": "__seed__", "seed": True, "side": side},
                    persist=True,
                )
            console_logger.info(f"Seeded rolling store with {len(seed_texts)} synthetic posts.")

        profile_store = AgentProfileStore(
            redis=message_cache.redis,
            window_size=PROFILE_WINDOW_SIZE,
            seed_weight=PROFILE_SEED_WEIGHT,
        )
        topology_tracker = NetworkTopologyTracker(
            topic=topic,
            profile_store=profile_store,
            redis_cache=topology_cache,
            redis_key=f"snapshot:{topic}",
        )
        topology_logger = TopologyLogger()
        console_logger.info(f"Topology snapshots will be written to: {topology_logger.file_path}")
    else:
        console_logger.info("Embedding-based context disabled (missing OPENAI_API_KEY or ENABLE_EMBEDDING_CONTEXT=0).")
    # Redis stream helper (used for cleanup)
    redis_stream = RedisStream(host=REDIS_HOST, port=REDIS_PORT)

    # Initialize Logger for recording agent publishes
    logger = Logger(num_agents=NUM_AGENTS)
    logger.log_agent_configs(agent_configs)
    console_logger.info(f"Logging publishes to: {logger.file_path}")

    # 2. Create agents
    agents = []
    for i in range(NUM_AGENTS):
        agent_id = f"agent_{i+1}"
        init_prompt = initial_prompt_template.format(
            topic=topic,
            unique_prompt=agent_prompts[i]
        )
        agent = NetworkAgent(
            id=agent_id,
            init_prompt=init_prompt,
            topic=topic,
            stream_name=STREAM_NAME,
            stream_group=f"group_{i+1}",
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            time_manager=time_manager,
            order_manager=None,  # Will set after all agents are created
            message_cache=message_cache,
            logger=logger,
            llm_service=llm_service,
            rolling_store=rolling_store,
            profile_store=profile_store,
            topology_tracker=topology_tracker,
            analysis_lock=analysis_lock,
            context_top_k=CONTEXT_TOP_K,
        )
        agents.append(agent)

    # 3. Initialize OrderManager with all agents
    order_manager = OrderManager(
        agents=agents,
        message_cache=message_cache,
        profile_store=profile_store,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
    )

    # Inject OrderManager into each agent
    for agent in agents:
        agent.order_manager = order_manager

    # 4. Start stance worker (optional)
    stance_worker = None
    if ENABLE_STANCE_WORKER:
        stance_worker = StanceWorker(
            topic=topic,
            stream_name=STREAM_NAME,
            group_name="stance_group",
            consumer_name="stance_consumer",
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            batch_size=STANCE_BATCH_SIZE,
            batch_interval=STANCE_BATCH_INTERVAL,
            local_llm=local_llm,
            llm_service=llm_service,
            use_openai=bool(os.getenv("OPENAI_API_KEY")),
        )
        await stance_worker.start()
        console_logger.info("StanceWorker started.")

    # 5. Start all agents
    console_logger.info("Starting agents...")
    for agent in agents:
        await agent.start()

    console_logger.info(f"All {NUM_AGENTS} agents are running.")

    # Periodic topology logging (proof of network structure over time)
    topo_task = None
    if topology_tracker and topology_logger:
        async def _topology_loop():
            while True:
                try:
                    agent_ids = [a.id for a in agents]
                    snap = await topology_tracker.maybe_update(agent_ids, force=True)
                    if snap is not None:
                        topology_logger.log_snapshot(snap)
                except Exception as exc:
                    console_logger.info(f"Topology snapshot failed (continuing): {exc}")
                await asyncio.sleep(TOPOLOGY_LOG_INTERVAL_S)

        topo_task = asyncio.create_task(_topology_loop())

    # 6. Kick off the conversation with an initial message from the first agent
    initial_message = await agents[0].generate_response(
        "Write the first viral post that kicks off a heated comment thread about this topic. "
        "Make a strong claim, then invite replies."
    )
    console_logger.info(f"Agent 1 starting conversation:")
    await agents[0].publish_message(initial_message)

    # 7. Run for the specified duration
    console_logger.info(f"Running for {RUN_DURATION_SECONDS} seconds...")
    await asyncio.sleep(RUN_DURATION_SECONDS)

    # 8. Cleanup
    console_logger.info("Shutting down...")
    # Stop agents first to ensure they don't publish after we delete the stream/groups
    console_logger.info("Stopping agents...")
    for agent in agents:
        try:
            await agent.stop()
        except Exception as e:
            console_logger.info(f"Error stopping agent {agent.id}: {e}")

    await logger.async_stop()

    if topo_task:
        topo_task.cancel()
        try:
            await topo_task
        except asyncio.CancelledError:
            pass

    if topology_logger:
        topology_logger.stop()

    if stance_worker:
        await stance_worker.stop()

    # Destroy consumer groups and remove the stream key
    try:
        await redis_stream.cleanup_stream(STREAM_NAME, num_groups=NUM_AGENTS)
    except Exception:
        console_logger.info("Redis stream cleanup encountered an error (continuing shutdown).")

    await message_cache.clear_all()
    await message_cache.close()

    if rolling_store is not None:
        await embed_cache.close()
    await topology_cache.close()

    if llm_service:
        await llm_service.stop()
    console_logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
