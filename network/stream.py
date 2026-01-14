import redis
import asyncio

from network.cache import RedisCache
from logs.logger import console_logger

class RedisStream:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True) 

    async def create_consumer_group(self, stream_name, group_name):
        try:
            self.redis.xgroup_create(stream_name, group_name, id='0', mkstream=True)
            console_logger.info(f"Consumer group '{group_name}' created for stream '{stream_name}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                console_logger.info(f"Consumer group '{group_name}' already exists for stream '{stream_name}'.")
            else:
                raise

    async def publish_message(self, stream_name, message_data : dict):
        """
        Publishes a message to the specified Redis stream.
        """
        message_id = self.redis.xadd(stream_name, message_data)
        console_logger.info(f"Message {message_id} published to stream '{stream_name}'.")
        return message_id

    async def consume_messages(self, stream_name, group_name, consumer_name):
        """
        A generator that yields messages from the stream for a specific consumer.
        """
        while True:
            messages = self.redis.xreadgroup(group_name, consumer_name, {stream_name: '>'}, count=1, block=1000)
            if messages:
                for stream, message_list in messages:
                    for message_id, message_data in message_list:
                        # console_logger.info(f"Consumer '{consumer_name}' received message {message_id} from stream '{stream_name}': {message_data}")
                        yield message_id, message_data
                        self.redis.xack(stream_name, group_name, message_id)
            await asyncio.sleep(0.1) # Prevent busy-waiting

    async def cleanup_stream(self, stream_name: str, num_groups=None, group_prefix: str = 'group_'):
        """
        Destroy consumer groups and delete the stream key.

        - If `num_groups` is provided, attempts to destroy groups named
          `{group_prefix}1`..`{group_prefix}{num_groups}`.
        - Otherwise enumerates groups via XINFO GROUPS and destroys them.
        - Finally deletes the stream key.
        """
        try:
            if num_groups is not None:
                for i in range(num_groups):
                    group = f"{group_prefix}{i+1}"
                    try:
                        self.redis.xgroup_destroy(stream_name, group)
                        console_logger.info(f"Destroyed consumer group '{group}' on stream '{stream_name}'.")
                    except redis.exceptions.ResponseError as e:
                        console_logger.info(f"Could not destroy group '{group}': {e}")
            else:
                try:
                    groups = self.redis.xinfo_groups(stream_name)
                    for g in groups:
                        group_name = g.get('name') if isinstance(g, dict) else g['name']
                        try:
                            self.redis.xgroup_destroy(stream_name, group_name)
                            console_logger.info(f"Destroyed consumer group '{group_name}' on stream '{stream_name}'.")
                        except Exception as ex:
                            console_logger.info(f"Could not destroy group '{group_name}': {ex}")
                except redis.exceptions.ResponseError as e:
                    console_logger.info(f"No groups found for stream '{stream_name}': {e}")

            try:
                deleted = self.redis.delete(stream_name)
                console_logger.info(f"Deleted stream '{stream_name}' (deleted={deleted}).")
            except Exception as e:
                console_logger.info(f"Failed to delete stream '{stream_name}': {e}")

        except Exception as e:
            console_logger.info(f"Unexpected error during cleanup of stream '{stream_name}': {e}")