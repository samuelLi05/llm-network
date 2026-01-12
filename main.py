"""Main runner for the LLM Network.

Initializes multiple NetworkAgents with shared topic prompts, integrates
TimeManager, OrderManager, RedisCache, and Logger for coordinated messaging.
"""

import asyncio
from agents.network_agent import NetworkAgent
from agents.prompt_configs.generate_prompt import PromptGenerator
from controller.time_manager import TimeManager
from controller.order_manager import OrderManager
from network.cache import RedisCache
from logs.logger import Logger, console_logger

# Configuration
NUM_AGENTS = 3
STREAM_NAME = "agent_stream"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
RUN_DURATION_SECONDS = 60  # How long to run the demo

initial_prompt_template = (
    "You are agent {agent_id} in a multi-agent social network debate about '{topic}'. " \
    "You strongly believe in the following statement: {unique_prompt} " \
    "Defend this statement in your responses. Prove your point is correct compared to others." \
    "Respond thoughtfully and concisely, keeping the conversation going."
)

converstation_starter = (
    "Let's begin our discussion about {topic}. " \
    "What are your initial thoughts on this subject?"
)


async def main():
    console_logger.info("Starting LLM Network...")

    # 1. Initialize shared components
    prompt_generator = PromptGenerator()
    topic = prompt_generator.get_topic()
    console_logger.info(f"Shared discussion topic: {topic}")

    # Generate unique prompts for each agent (same topic, different wording)
    agent_prompts = prompt_generator.generate_multiple_prompts(NUM_AGENTS)

    # Initialize TimeManager with 3-second global interval
    time_manager = TimeManager(global_interval=3.0)

    # Initialize RedisCache for storing agent message histories
    message_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT)

    # Initialize Logger for recording agent publishes
    logger = Logger(num_agents=NUM_AGENTS)
    console_logger.info(f"Logging publishes to: {logger.file_path}")

    # 2. Create agents
    agents = []
    for i in range(NUM_AGENTS):
        agent_id = f"agent_{i+1}"
        init_prompt = initial_prompt_template.format(
            agent_id=i+1,
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

    # 4. Start all agents
    console_logger.info("Starting agents...")
    for agent in agents:
        await agent.start()

    console_logger.info(f"All {NUM_AGENTS} agents are running.")

    # 5. Kick off the conversation with an initial message from the first agent
    initial_message = converstation_starter.format(topic=topic)
    console_logger.info(f"Agent 1 starting conversation:")
    await agents[0].publish_message(initial_message)

    # 6. Run for the specified duration
    console_logger.info(f"Running for {RUN_DURATION_SECONDS} seconds...")
    await asyncio.sleep(RUN_DURATION_SECONDS)

    # 7. Cleanup
    console_logger.info("Shutting down...")

    await logger.async_stop()
    await message_cache.close()
    console_logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
