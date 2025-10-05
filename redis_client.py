import json
import asyncio
import os
from typing import List, Dict, Any
import redis.asyncio as aioredis
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = None

    async def connect(self):
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def close(self):
        if self.redis:
            await self.redis.close()

    async def ping(self):
        return await self.redis.ping()

    async def cache_message(self, message_data: Dict[str, Any]):
        try:
            user_key = f"user:{message_data['receiver_id']}:recent_messages"
            await self.redis.lpush(user_key, json.dumps(message_data))
            await self.redis.ltrim(user_key, 0, 99)
            await self.redis.expire(user_key, 24 * 60 * 60)
        except Exception as e:
            logger.error(f"Message caching failed: {e}")

    async def get_recent_messages(self, user_id: int) -> List[Dict]:
        try:
            user_key = f"user:{user_id}:recent_messages"
            messages_raw = await self.redis.lrange(user_key, 0, 49)

            messages = []
            for msg_raw in messages_raw:
                try:
                    messages.append(json.loads(msg_raw))
                except json.JSONDecodeError:
                    continue

            return messages
        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}")
            return []

    async def publish_message(self, message_data: Dict[str, Any]):
        try:
            channel = f"user:{message_data['receiver_id']}:messages"
            await self.redis.publish(channel, json.dumps(message_data))
        except Exception as e:
            logger.error(f"Message publishing failed: {e}")

    async def subscribe_to_user_messages(self, user_id: int, websocket: WebSocket):
        try:
            pubsub = self.redis.pubsub()
            channel = f"user:{user_id}:messages"
            await pubsub.subscribe(channel)

            logger.info(f"Subscribed to channel: {channel}")

            while True:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        await websocket.send_text(message["data"])
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Subscription error: {e}")
                    break
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
