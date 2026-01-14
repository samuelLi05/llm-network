import os
import asyncio
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv
from openai import OpenAI
from network.stream import RedisStream

from logs.logger import console_logger

if TYPE_CHECKING:
    from controller.time_manager import TimeManager
    from controller.order_manager import OrderManager
    from network.cache import RedisCache
    from logs.logger import Logger

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class NetworkAgent:
    """An agent that listens to a Redis stream, generates responses, and publishes back.

    Integrates with TimeManager for rate limiting, OrderManager for turn selection,
    RedisCache for message history, and Logger for publish event logging.
    """

    def __init__(
        self,
        id: str,
        init_prompt: str,
        stream_name: str,
        stream_group: str,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        seed: Optional[int] = None,
        time_manager: Optional["TimeManager"] = None,
        order_manager: Optional["OrderManager"] = None,
        message_cache: Optional["RedisCache"] = None,
        logger: Optional["Logger"] = None,
    ):
        self.id = id
        self.init_prompt = init_prompt
        self.stream_client = RedisStream(host=redis_host, port=redis_port)
        self.stream_name = stream_name
        self.stream_group = stream_group
        self.seed = seed
        self.consumer_name = f"{self.stream_group}-consumer"
        self.is_consuming = asyncio.Event()

        # Controller integrations
        self.time_manager = time_manager
        self.order_manager = order_manager
        self.message_cache = message_cache
        self.logger = logger
        self._publish_lock = asyncio.Lock()
        self._consumer_task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize consumer group and start listening for messages."""
        await self.stream_client.create_consumer_group(self.stream_name, self.stream_group)
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self.consume_stream())
        await self.is_consuming.wait()
        console_logger.info(f"Agent {self.id} started and is listening for messages.")

    async def stop(self):
        """Stop background consumption (useful for notebook reruns/cleanup)."""
        task = self._consumer_task
        self._consumer_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def consume_stream(self):
        """Consume messages from the Redis stream and process them."""
        console_logger.info(f"Agent {self.id} is consuming from stream '{self.stream_name}' as part of group '{self.stream_group}'.")
        self.is_consuming.set()
        async for message_id, message_data in self.stream_client.consume_messages(
            self.stream_name, self.stream_group, self.consumer_name
        ):
            sender_id = message_data.get('sender_id')
            if sender_id == self.id:
                # Ignore own messages
                console_logger.info(f"Agent {self.id} ignored its own message {message_id}.")
                await asyncio.sleep(0.1)
                continue

            # console_logger.info(f"Agent {self.id} received message {message_id}: {message_data}")
            incoming_post = message_data.get('content', '')
            if not incoming_post:
                continue

            # Check if this agent is the designated responder (OrderManager logic)
            if self.order_manager:
                if not self.order_manager.is_my_turn(self.id):
                    # Not our turn, skip responding
                    designated = self.order_manager.get_designated_responder()
                    console_logger.info(f"Agent {self.id} skipped (designated responder is {designated}).")
                    continue
                # Clear the designation since we're responding
                self.order_manager.clear_designated_responder()

            # Generate and publish response
            wrapped_prompt = (
                "React to the post below in your own voice. " \
                "Do NOT copy or paraphrase it line-by-line. " \
                "Write a NEW, distinct social-media-style post that reflects your fixed stance and worldview, " \
                "and directly contradict at least one implied assumption in the original post. " \
                "Keep it attention-grabbing, concise, and assertive.\n" \
                "POST TO REACT TO:\n" \
                f"{incoming_post}"
            )
            response = await self.generate_response(wrapped_prompt)
            await self.publish_message(response)

    async def publish_message(self, message: str):
        """Publish a message to the Redis stream with rate limiting and logging."""
        # Wait for TimeManager permission if available
        if self.time_manager:
            async with self.time_manager.publish_lock(self.id):
                await self._do_publish(message)
        else:
            await self._do_publish(message)

    async def _do_publish(self, message: str):
        """Internal publish: designate next responder FIRST, then send to stream, cache, and log."""
        async with self._publish_lock:
            message_data = {'sender_id': self.id, 'content': message}

            # Designate the next responder BEFORE publishing the message
            if self.order_manager:
                next_responder = self.order_manager.select_and_store_next_responder(exclude_agent_id=self.id)
                console_logger.info(f"Agent {self.id} designating next responder: {next_responder}")

            # Now publish the message (other agents will see the designated responder)
            await self.stream_client.publish_message(self.stream_name, message_data)

        # Append to RedisCache (per-agent message history)
        if self.message_cache:
            await self.message_cache.append_response(self.id, message_data)

        # Log the publish event
        if self.logger:
            await self.logger.async_log_publish(self.id, message)

            console_logger.debug(f"Agent {self.id} published message.")

    async def generate_response(self, prompt: Optional[str] = None) -> str:
        """Generate a response using the OpenAI API."""
        console_logger.info(f"Agent {self.id} is generating response")
        try:
            last_message = (prompt or "").strip()
            if not last_message:
                # Helps avoid an empty user message on the first post.
                last_message = "Write your next post."

            # Build assistant context from recent messages across all agents
            recent_messages = ""
            if self.message_cache:
                agents_list = None
                if self.order_manager and getattr(self.order_manager, "agents", None):
                    agents_list = self.order_manager.agents
                else:
                    # fallback to only this agent
                    agents_list = [self]

                parts = []
                for a in agents_list:
                    try:
                        # console_logger.debug(f"Agent {self.id} fetching cache for {a.id}...")
                        items = await self.message_cache.get_responses(a.id, last_n=5)
                    except Exception:
                        console_logger.error(f"Error fetching messages for agent {a.id} from cache.")
                        items = []
                    # normalize items to strings
                    normalized = [str(item) for item in items]
                    if normalized:
                        parts.append(f"{a.id}: " + " | ".join(normalized))

                recent_messages = "\n".join(parts)
            # console_logger.info(f"Agent {self.id} recent messages for context: {recent_messages}")
            console_logger.debug(f"Agent {self.id} calling OpenAI responses API...")
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.responses.create,
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": self.init_prompt},
                        {"role": "user", "content": last_message},
                        {"role": "assistant", "content": recent_messages},
                    ],
                    temperature=0.7,
                    max_output_tokens=300,
                ),
                timeout=60,
            )
            # console_logger.info(f"Agent {self.id} generated response: {response.output_text}")
            return response.output_text
        except asyncio.TimeoutError:
            console_logger.error(f"Timed out generating response for agent {self.id} (Redis/OpenAI timeout).")
            return "I'm sorry, I timed out while generating a response."
        except Exception as e:
            console_logger.error(f"Error generating response for agent {self.id}: {e}")
            return "I'm sorry, I couldn't process that."
        
