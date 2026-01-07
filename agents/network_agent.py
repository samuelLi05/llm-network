import os
import asyncio
from dotenv import load_dotenv
import openai
import redis
import os
import asyncio
from dotenv import load_dotenv
import openai
from network.network import RedisStream

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class NetworkAgent:
    def __init__(self, id: str, init_prompt: str, stream_name: str, stream_group: str, redis_host: str = 'localhost', redis_port: int = 6379, seed=None):
        self.id = id
        self.init_prompt = init_prompt
        self.stream_client = RedisStream(host=redis_host, port=redis_port)
        self.stream_name = stream_name
        self.stream_group = stream_group
        self.seed = seed
        self.consumer_name = f"{stream_group}-consumer-{id}"

    async def start(self):
        """
        Initializes the agent by creating the consumer group and starting the message consumption loop.
        """
        await self.stream_client.create_consumer_group(self.stream_name, self.stream_group)
        asyncio.create_task(self.consume_stream())
        print(f"Agent {self.id} started and is listening for messages.")

    async def consume_stream(self):
        """
        Consumes messages from the Redis stream and processes them.
        This is a long-running task that should be started with the agent.
        """
        print(f"Agent {self.id} is consuming from stream '{self.stream_name}' as part of group '{self.stream_group}'.")
        async for message_id, message_data in self.stream_client.consume_messages(self.stream_name, self.stream_group, self.consumer_name):
            if message_data.get('sender_id') != self.id:  # Avoid processing its own messages
                print(f"Agent {self.id} received message {message_id}: {message_data}")
                prompt = message_data.get('content', '')
                if prompt:
                    response = await self.generate_response(prompt)
                    await self.publish_message(response)


    async def publish_message(self, message: str):
        """
        Publishes a message to the Redis stream.
        """
        message_data = {'sender_id': self.id, 'content': message}
        await self.stream_client.publish_message(self.stream_name, message_data)

    async def generate_response(self, prompt: str) -> str:
        """
        Generates a response using the OpenAI API.
        """
        pass
