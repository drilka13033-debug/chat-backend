# Создаю чистый профессиональный проект без комментариев о правках

# main.py
main_py = '''import asyncio
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
'''

# models.py
models_py = '''from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    display_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.receiver_id", back_populates="receiver")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")
'''

# database.py
database_py = '''import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://chat_user:chat_password@localhost:5432/chat_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

# redis_client.py
redis_client_py = '''import json
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
'''

# schemas.py
schemas_py = '''from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    username: str
    display_name: Optional[str] = None
    
    @validator('username')
    def username_must_be_valid(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not v.isalnum():
            raise ValueError('Username must contain only letters and numbers')
        return v


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    sender_id: int
    receiver_id: int
    content: str
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message content is too long (max 1000 characters)')
        return v.strip()
    
    @validator('receiver_id')
    def receiver_must_be_different(cls, v, values):
        if 'sender_id' in values and v == values['sender_id']:
            raise ValueError('Cannot send message to yourself')
        return v


class MessageResponse(BaseModel):
    id: int
    sender_id: int
    sender_username: str
    receiver_id: int
    content: str
    timestamp: str
    delivered_via_websocket: Optional[bool] = None
    
    class Config:
        from_attributes = True
'''

# requirements.txt
requirements_txt = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis[hiredis]==5.0.1
pydantic==2.5.0
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
'''

# docker-compose.yml
docker_compose_yml = '''version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: chat_db
      POSTGRES_USER: chat_user
      POSTGRES_PASSWORD: chat_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U chat_user -d chat_db"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  chat-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://chat_user:chat_password@postgres:5432/chat_db
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - .:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
'''

# Dockerfile
dockerfile = '''FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# test_api.py
test_api_py = '''import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from database import get_db
from models import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestUsers:
    def test_create_user(self):
        response = client.post(
            "/users/",
            json={"username": "testuser", "display_name": "Test User"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["display_name"] == "Test User"
        assert "id" in data
    
    def test_create_duplicate_user(self):
        client.post("/users/", json={"username": "duplicate", "display_name": "First"})
        response = client.post("/users/", json={"username": "duplicate", "display_name": "Second"})
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_get_users(self):
        client.post("/users/", json={"username": "listuser", "display_name": "List User"})
        response = client.get("/users/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_get_user_by_id(self):
        create_response = client.post("/users/", json={"username": "getuser", "display_name": "Get User"})
        user_id = create_response.json()["id"]
        
        response = client.get(f"/users/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == "getuser"


class TestMessages:
    def setup_method(self):
        self.user1_response = client.post("/users/", json={"username": "sender", "display_name": "Sender"})
        self.user2_response = client.post("/users/", json={"username": "receiver", "display_name": "Receiver"})
        
        self.user1_id = self.user1_response.json()["id"]
        self.user2_id = self.user2_response.json()["id"]
    
    def test_send_message(self):
        response = client.post(
            "/messages/",
            json={
                "sender_id": self.user1_id,
                "receiver_id": self.user2_id,
                "content": "Hello, World!"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sender_id"] == self.user1_id
        assert data["receiver_id"] == self.user2_id
        assert data["content"] == "Hello, World!"
        assert "id" in data
        assert "timestamp" in data
    
    def test_send_message_to_nonexistent_user(self):
        response = client.post(
            "/messages/",
            json={
                "sender_id": self.user1_id,
                "receiver_id": 99999,
                "content": "Hello, Ghost!"
            }
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_messages(self):
        client.post(
            "/messages/",
            json={
                "sender_id": self.user1_id,
                "receiver_id": self.user2_id,
                "content": "Test message"
            }
        )
        
        response = client.get(f"/messages/?user_id={self.user2_id}")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["content"] == "Test message"


class TestAPI:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data


@pytest.mark.asyncio
async def test_websocket_connection():
    user_response = client.post("/users/", json={"username": "wsuser", "display_name": "WebSocket User"})
    user_id = user_response.json()["id"]
    
    with client.websocket_connect(f"/ws/{user_id}") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "system"
        assert "Connected to chat" in data["message"]


if __name__ == "__main__":
    pytest.main([__file__])
'''

# README.md
readme_md = '''# Mini Chat Backend

Real-time chat service built with FastAPI, WebSocket, Redis pub/sub and PostgreSQL.

## Features

- REST API for user and message management
- WebSocket for real-time messaging
- Redis pub/sub for scalable notifications
- PostgreSQL for persistent data storage
- Docker Compose for easy deployment
- Comprehensive test coverage

## Tech Stack

- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Relational database
- **Redis** - Caching and pub/sub messaging
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation
- **WebSocket** - Real-time communication
- **Docker** - Containerization
- **Pytest** - Testing framework

## Quick Start

### Using Docker Compose

```bash
git clone <repository-url>
cd mini-chat-backend
docker-compose up --build
```

Service will be available at: http://localhost:8000

### Development Setup

```bash
# Start PostgreSQL and Redis
docker-compose up postgres redis

# Set environment variables
export DATABASE_URL="postgresql://chat_user:chat_password@localhost:5432/chat_db"
export REDIS_URL="redis://localhost:6379"

# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn main:app --reload
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Users
- `POST /users/` - Create user
- `GET /users/` - List users
- `GET /users/{user_id}` - Get user by ID

### Messages
- `POST /messages/` - Send message
- `GET /messages/` - Get message history
- `GET /messages/recent/{user_id}` - Get recent cached messages

### WebSocket
- `WS /ws/{user_id}` - Real-time chat connection

### System
- `GET /health` - Health check

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest test_api.py::TestUsers::test_create_user
```

## Usage Examples

### Create Users

```bash
curl -X POST "http://localhost:8000/users/" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "alice", "display_name": "Alice Smith"}'
```

### Send Message

```bash
curl -X POST "http://localhost:8000/messages/" \\
  -H "Content-Type: application/json" \\
  -d '{"sender_id": 1, "receiver_id": 2, "content": "Hello!"}'
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/1');

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('New message:', message);
};

ws.send(JSON.stringify({type: 'ping'}));
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   PostgreSQL     │    │     Redis       │
│                 │    │                  │    │                 │
│ • REST API      │    │ • Users          │    │ • Message Cache │
│ • WebSocket     │────│ • Messages       │────│ • Pub/Sub       │
│ • Pub/Sub       │    │ • History        │    │ • Sessions      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Environment Variables

```bash
DATABASE_URL=postgresql://chat_user:chat_password@localhost:5432/chat_db
REDIS_URL=redis://localhost:6379
```

## License

MIT License
'''

# Создаем проект
import os
os.makedirs('chat-backend', exist_ok=True)

project_files = {
    'main.py': main_py,
    'models.py': models_py,
    'database.py': database_py,
    'redis_client.py': redis_client_py,
    'schemas.py': schemas_py,
    'requirements.txt': requirements_txt,
    'docker-compose.yml': docker_compose_yml,
    'Dockerfile': dockerfile,
    'test_api.py': test_api_py,
    'README.md': readme_md
}

for filename, content in project_files.items():
    filepath = os.path.join('chat-backend', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Professional chat backend project created:")
print("\nchat-backend/")
for filename in project_files.keys():
    print(f"├── {filename}")

print("\nTo run:")
print("cd chat-backend")
print("docker-compose up --build")