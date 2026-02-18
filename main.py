"""Main runner for the LLM Network.

Initializes multiple NetworkAgents with shared topic prompts, integrates
TimeManager, OrderManager, RedisCache, and Logger for coordinated messaging.
"""

import asyncio

import os
import random
import time
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
from controller.connection_graph import ConnectionGraph
from controller.stance_worker import StanceWorker
from network.cache import RedisCache
from network.stream import RedisStream
from logs.logger import Logger, console_logger
from logs.topology_logger import TopologyLogger

# Configuration
NUM_AGENTS = 30
STREAM_NAME = "agent_stream"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
RUN_DURATION_SECONDS = 10800 # 3 hours
USE_LOCAL_LLM = True
ENABLE_STANCE_WORKER = False
STANCE_BATCH_SIZE = 5
STANCE_BATCH_INTERVAL = 30

# Embedding configuration
# If True: use stance-aware SBERT for topic/stance/strength scoring, while still
# preferring OpenAI embeddings for cosine similarity between vectors.
# If False: use OpenAI embeddings for both semantic vectors and scoring.
USE_LOCAL_EMBEDDING_MODEL = True

# How to score stance/topic/strength from social-media-style posts.
# Options: "heading" (first line), "weighted" (softmax-weighted mean over spans), "best_span" (single best span), "full" (full post)
SCORE_SPAN_MODE = "full"

LOCAL_EMBEDDING_MODEL = "all-mpnet-base-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Startup concurrency (bounded to avoid overwhelming Redis / threadpool)
STARTUP_CONCURRENCY = 25

# Stance statement for intializeing with a baseline stance
USE_BASELINE_STATEMENT = True
BASELINE_STATEMENT = "Vaccines cause autism"
BASELINE_TOPIC = "vaccine safety and autism"

# Seed for assigning baseline stances to agents
BASELINE_ASSIGNMENT_SEED = 1234

# Connection graph, define the size of the random hamiltonian graph
GRAPH_SEED = 42
GRAPH_MIN_DEGREE = 8

# Computing the degree as a function of the number of agents n: ie degree = (n ** exponent) * factor + offset
GRAPH_DEGREE_EXPONENT = 0.5
GRAPH_DEGREE_FACTOR = 2.0
GRAPH_DEGREE_OFFSET = 0.0
GRAPH_MAX_DEGREE = 15

ENABLE_EMBEDDING_CONTEXT = True
ROLLING_STORE_MAX_ITEMS = 2000
CONTEXT_TOP_K = 10
PROFILE_WINDOW_SIZE = 50
PROFILE_SEED_WEIGHT = 5.0

TOPOLOGY_LOG_INTERVAL_S = 5.0

initial_prompt_template = (
    "You are a social media user who posts about {topic}. There are other users on the network who have different perspectives on this topic. "
    "The sentence \"{unique_prompt}\" reflects your stable perspective on this topic. "
    "Write from this perspective consistently.\n\n"
    
    "Produce short standalone posts that expresses your current view and provides reasoning through made up examples (news, studies, numbers, stats, trusted sources, etc). "
    "Do everything you can to advance your argument and make it convincing, even if that means making up extreme claims, eye-catching details, and sensational examples. "
    "Come up with brand new unique examples or reasoning every post to further your argument to beat out other arguments on the feed. "
    "Do not frame it as a rebuttal or direct response to anyone. "
    "Do not quote or reference opposing claims.\n\n"
    
    "Be concise and write like a real person sharing a thought on their timeline. "
    "Avoid engagement-bait call-to-actions (e.g., 'like/share/reply'), and avoid hashtag or emoji spam. "
    "Do not refer to yourself as an agent, AI, or participant in a debate."
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
    agent_configs: dict[str, str] = {}
    agent_stance_weights: dict[str, float] = {}

    if USE_BASELINE_STATEMENT:
        topic = BASELINE_TOPIC
        console_logger.info(f"Baseline mode enabled: topic='{topic}' stance='{BASELINE_STATEMENT}'")

        # Build a pool of fixed stance sentences mapped to stance weights.
        fixed_pool = prompt_generator.generate_fixed_opinions(
            BASELINE_STATEMENT,
            weighted_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        fixed_items = list(fixed_pool.items())  # (stance_sentence, weight)
        if not fixed_items:
            raise RuntimeError("No fixed templates found for baseline mode")

        # Assign each agent a fixed stance sentence + associated stance weight.
        # We cycle through the pool to guarantee coverage, then shuffle for variety.
        rng = random.Random(BASELINE_ASSIGNMENT_SEED)
        assignments = []
        while len(assignments) < NUM_AGENTS:
            assignments.extend(fixed_items)
        assignments = assignments[:NUM_AGENTS]
        rng.shuffle(assignments)

        agent_prompts = [s for (s, _w) in assignments]
        for i in range(NUM_AGENTS):
            agent_id = f"agent_{i+1}"
            stance_sentence, stance_weight = assignments[i]
            init_prompt = initial_prompt_template.format(
                topic=topic,
                unique_prompt=stance_sentence,
            )
            agent_configs[agent_id] = init_prompt
            agent_stance_weights[agent_id] = float(stance_weight)
    else:
        topic = prompt_generator.get_topic()
        console_logger.info(f"Shared discussion topic: {topic}")

        # Generate unique prompts for each agent (same topic, different wording)
        agent_prompts = prompt_generator.generate_multiple_prompts(NUM_AGENTS)

        for i in range(NUM_AGENTS):
            agent_id = f"agent_{i+1}"
            init_prompt = initial_prompt_template.format(
                topic=topic,
                unique_prompt=agent_prompts[i],
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

    if ENABLE_EMBEDDING_CONTEXT:
        analyzer = EmbeddingAnalyzer(
            BASELINE_STATEMENT if USE_BASELINE_STATEMENT else topic,
            use_local_embedding_model=USE_LOCAL_EMBEDDING_MODEL,
            use_baseline_statement=USE_BASELINE_STATEMENT,
            score_span_mode=SCORE_SPAN_MODE,
            openai_embedding_model=OPENAI_EMBEDDING_MODEL,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
        )
        rolling_store = RollingEmbeddingStore(
            topic=BASELINE_STATEMENT if USE_BASELINE_STATEMENT else topic,
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

            # Add a few extra high-contrast, attention-grabbing seeds.
            # In baseline mode, keep seeds about the *topic* and refer to the
            # baseline statement as a claim (without asserting it as fact).
            if USE_BASELINE_STATEMENT:
                pass
                # seed_texts.extend(
                #     [
                #         ("pro", f"We need a calm, evidence-first conversation about {topic}."),
                #         ("pro", f"If you care about public health, take {topic} seriously and stop platforming fear."),
                #         ("anti", f"People keep repeating the claim '{BASELINE_STATEMENT}'. Either show evidence or stop spreading it."),
                #         ("anti", f"Public trust collapses when misinformation about {topic} goes unchallenged."),
                #         ("neutral", f"On {topic}, I want clear data, not vibes. What studies are people citing — if any?"),
                #     ]
                # )
            else:
                pass
                # seed_texts.extend(
                #     [
                #         ("pro", f"Enough dithering. {topic} is non-negotiable — we should expand it now."),
                #         ("pro", f"If you're against {topic}, you're choosing stagnation. Push it through."),
                #         ("anti", f"Wake up: {topic} is a harmful mistake. Stop pretending it's 'progress'."),
                #         ("anti", f"{topic} is a disaster in slow motion. Reject it before it spreads."),
                #     ]
                # )

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
            use_local_embedding_model=USE_LOCAL_EMBEDDING_MODEL,
            use_baseline_statement=USE_BASELINE_STATEMENT,
            openai_embedding_model=OPENAI_EMBEDDING_MODEL,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
        )
        topology_tracker = NetworkTopologyTracker(
            topic=topic,
            profile_store=profile_store,
            redis_cache=topology_cache,
            redis_key=f"snapshot:{topic}",
            use_local_embedding_model=USE_LOCAL_EMBEDDING_MODEL,
            use_baseline_statement=USE_BASELINE_STATEMENT,
            openai_embedding_model=OPENAI_EMBEDDING_MODEL,
            local_embedding_model=LOCAL_EMBEDDING_MODEL,
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

    # Persist baseline stance weights for analysis (optional)
    if USE_BASELINE_STATEMENT and agent_stance_weights:
        try:
            logger.log_stance_configs(agent_stance_weights, header=f"Baseline stance weights for: {BASELINE_STATEMENT}")
            console_logger.info(f"Logging stance configs to: {logger.stance_config_path}")
        except Exception as exc:
            console_logger.info(f"Failed to log stance configs (continuing): {exc}")

        try:
            # Store as a Redis hash for easy retrieval in analysis tools.
            await message_cache.redis.hset("llm_network:agent_stance_weights", mapping=agent_stance_weights)
        except Exception as exc:
            console_logger.info(f"Failed to persist stance weights to Redis (continuing): {exc}")

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

    # 2.5 Build a connected random social graph and inject into ordering/routing
    agent_ids = [a.id for a in agents]
    graph = ConnectionGraph(
        agent_ids,
        seed=42,
        graph_type="community",
        avg_degree_target=6.0,  # density knob (not an exact constraint)
        base_in_social_needs=0.5,
        base_out_social_needs=2.0,
        lognormal_mu=1.0,
        lognormal_sigma=1.0,
        activity_lognormal_mu=1.0,
        activity_lognormal_sigma=1.8,
        group_divisor=5,
        within_group_fraction=0.85,
        cross_group_divisor=5.0,
        reciprocity_within=0.4,
        reciprocity_cross=0.05,
        reciprocity_influencer=0.0,
        influencer_quantile=0.9,
        min_out_degree=2,
        min_in_degree=1,
    ).get_graph()
    avg_degree = (sum(len(v) for v in graph.values()) / max(1, len(graph)))
    console_logger.info(
        f"Connection graph built: n={len(agent_ids)} avg_degree="
        f"{avg_degree:.2f}"
    )

    # Persist the initial connection graph to the topology log for debugging.
    if topology_logger:
        try:
            topology_logger.log_snapshot(
                {
                    "t": "connection_graph",
                    "ts": time.time(),
                    "n": len(agent_ids),
                    "ad": float(avg_degree),
                    "g": graph,
                }
            )
        except Exception as exc:
            console_logger.info(f"Failed to log connection graph snapshot (continuing): {exc}")

    # 3. Initialize OrderManager with all agents
    order_manager = OrderManager(
        agents=agents,
        message_cache=message_cache,
        profile_store=profile_store,
        connection_graph=graph,
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
    sem = asyncio.Semaphore(max(1, int(STARTUP_CONCURRENCY)))

    async def _start_one(a: NetworkAgent) -> None:
        async with sem:
            await a.start()

    await asyncio.gather(*(_start_one(a) for a in agents))

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
        "Write your first social media post. Follow your fixed stance and provide reasoning. Keep it concise and natural, like a real person."
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
