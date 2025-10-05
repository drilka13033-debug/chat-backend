import pytest
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
