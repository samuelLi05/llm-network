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
        stream_group="group1"
    )

    # Start the agents
    #await agent1.start()
    #await agent2.start()

if __name__ == "__main__":
    asyncio.run(main())
