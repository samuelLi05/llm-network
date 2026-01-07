import asyncio
from agents.network_agent import NetworkAgent

async def main():
    # Create two agents
    agent1 = NetworkAgent(
        id="agent1",
        init_prompt="You are a helpful AI assistant.",
        stream_name="agent_stream",
        stream_group="group1"
    )

    agent2 = NetworkAgent(
        id="agent2",
        init_prompt="You are an unhelpful AI assistant.",
        stream_name="agent_stream",
        stream_group="group2"
    )

    # Start the agents
    await agent1.start()
    await agent2.start()

    # Test message from agent1 to agent2
    await agent1.publish_message("Hello from Agent 1!")

    # Keep the main function running to allow agents to process messages
    await asyncio.sleep(60)  # Run for 30 seconds for demonstration


if __name__ == "__main__":
    asyncio.run(main())
