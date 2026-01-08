import redis
import asyncio

from network.cache import RedisCache

class RedisStream:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True) 

    async def create_consumer_group(self, stream_name, group_name):
        try:
            self.redis.xgroup_create(stream_name, group_name, id='0', mkstream=True)
            print(f"Consumer group '{group_name}' created for stream '{stream_name}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                print(f"Consumer group '{group_name}' already exists for stream '{stream_name}'.")
            else:
                raise

    async def publish_message(self, stream_name, message_data : dict):
        """
        Publishes a message to the specified Redis stream.
        """
        message_id = self.redis.xadd(stream_name, message_data)
        print(f"Message {message_id} published to stream '{stream_name}'.")
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
                        print (f"Consumer '{consumer_name}' received message {message_id} from stream '{stream_name}': {message_data}")
                        yield message_id, message_data
                        self.redis.xack(stream_name, group_name, message_id)
            await asyncio.sleep(0.1) # Prevent busy-waiting