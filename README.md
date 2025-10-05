# Mini Chat Backend

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
curl -X POST "http://localhost:8000/users/" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "display_name": "Alice Smith"}'
```

### Send Message

```bash
curl -X POST "http://localhost:8000/messages/" \
  -H "Content-Type: application/json" \
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
