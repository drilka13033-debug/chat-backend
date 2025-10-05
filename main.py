import asyncio
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import redis.asyncio as aioredis
import logging

from database import get_db, engine
from models import Base, User, Message
from schemas import UserCreate, UserResponse, MessageCreate, MessageResponse
from redis_client import RedisClient

Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mini Chat API",
    description="Real-time chat service with WebSocket, Redis pub/sub and PostgreSQL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = RedisClient()


class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected via WebSocket")

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected")

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
                return True
            except:
                self.disconnect(user_id)
                return False
        return False


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    await redis_client.connect()
    logger.info("Application started")


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
    logger.info("Application shutdown")


@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Mini Chat Backend API",
        "version": "1.0.0",
        "endpoints": {
            "users": "/users/",
            "messages": "/messages/",
            "websocket": "/ws/{user_id}",
            "docs": "/docs"
        }
    }


@app.post("/users/", response_model=UserResponse, tags=["Users"])
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    db_user = User(
        username=user.username,
        display_name=user.display_name or user.username
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    logger.info(f"Created user: {db_user.username} (ID: {db_user.id})")
    return db_user


@app.get("/users/", response_model=List[UserResponse], tags=["Users"])
async def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/messages/", response_model=MessageResponse, tags=["Messages"])
async def send_message(message: MessageCreate, db: Session = Depends(get_db)):
    sender = db.query(User).filter(User.id == message.sender_id).first()
    if not sender:
        raise HTTPException(status_code=404, detail="Sender not found")

    receiver = db.query(User).filter(User.id == message.receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")

    db_message = Message(
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        content=message.content,
        timestamp=datetime.utcnow()
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)

    message_data = {
        "id": db_message.id,
        "sender_id": db_message.sender_id,
        "sender_username": sender.username,
        "receiver_id": db_message.receiver_id,
        "content": db_message.content,
        "timestamp": db_message.timestamp.isoformat(),
        "type": "message"
    }

    await redis_client.cache_message(message_data)
    sent_via_ws = await manager.send_personal_message(message_data, message.receiver_id)
    await redis_client.publish_message(message_data)

    logger.info(f"Message sent from {message.sender_id} to {message.receiver_id}, WebSocket: {sent_via_ws}")

    return {
        **message_data,
        "delivered_via_websocket": sent_via_ws
    }


@app.get("/messages/", response_model=List[MessageResponse], tags=["Messages"])
async def get_messages(
    user_id: int,
    chat_with: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(Message).filter(
        (Message.sender_id == user_id) | (Message.receiver_id == user_id)
    )

    if chat_with:
        query = query.filter(
            ((Message.sender_id == user_id) & (Message.receiver_id == chat_with)) |
            ((Message.sender_id == chat_with) & (Message.receiver_id == user_id))
        )

    messages = query.order_by(desc(Message.timestamp)).offset(skip).limit(limit).all()

    result = []
    for msg in messages:
        sender = db.query(User).filter(User.id == msg.sender_id).first()
        result.append({
            "id": msg.id,
            "sender_id": msg.sender_id,
            "sender_username": sender.username if sender else "Unknown",
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        })

    return result


@app.get("/messages/recent/{user_id}", tags=["Messages"])
async def get_recent_messages(user_id: int):
    messages = await redis_client.get_recent_messages(user_id)
    return {"messages": messages}


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        await websocket.close(code=4004, reason="User not found")
        return

    await manager.connect(websocket, user_id)

    try:
        await websocket.send_text(json.dumps({
            "type": "system",
            "message": f"Connected to chat as {user.username}",
            "timestamp": datetime.utcnow().isoformat()
        }))

        pubsub_task = asyncio.create_task(
            redis_client.subscribe_to_user_messages(user_id, websocket)
        )

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        if 'pubsub_task' in locals():
            pubsub_task.cancel()
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)
        if 'pubsub_task' in locals():
            pubsub_task.cancel()


@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "OK"
    except Exception as e:
        db_status = f"ERROR: {e}"

    try:
        await redis_client.ping()
        redis_status = "OK"
    except Exception as e:
        redis_status = f"ERROR: {e}"

    return {
        "status": "OK" if db_status == "OK" and redis_status == "OK" else "ERROR",
        "database": db_status,
        "redis": redis_status,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
