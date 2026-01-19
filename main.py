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
from controller.time_manager import TimeManager
from controller.order_manager import OrderManager
from controller.stance_worker import StanceWorker
from network.cache import RedisCache
from network.stream import RedisStream
from logs.logger import Logger, console_logger

# Configuration
NUM_AGENTS = 10
STREAM_NAME = "agent_stream"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
RUN_DURATION_SECONDS = 100
USE_LOCAL_LLM = True
ENABLE_STANCE_WORKER = True
STANCE_BATCH_SIZE = int(os.getenv("STANCE_BATCH_SIZE", "5"))
STANCE_BATCH_INTERVAL = float(os.getenv("STANCE_BATCH_INTERVAL", "30"))

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
            stream_name=STREAM_NAME,
            stream_group=f"group_{i+1}",
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            time_manager=time_manager,
            order_manager=None,  # Will set after all agents are created
            message_cache=message_cache,
            logger=logger,
            llm_service=llm_service,
        )
        agents.append(agent)

    # 3. Initialize OrderManager with all agents
    order_manager = OrderManager(
        agents=agents,
        message_cache=message_cache,
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

    if stance_worker:
        await stance_worker.stop()

    # Destroy consumer groups and remove the stream key
    try:
        await redis_stream.cleanup_stream(STREAM_NAME, num_groups=NUM_AGENTS)
    except Exception:
        console_logger.info("Redis stream cleanup encountered an error (continuing shutdown).")

    await message_cache.clear_all()
    await message_cache.close()

    if llm_service:
        await llm_service.stop()
    console_logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
