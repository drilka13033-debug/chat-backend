# Mini Chat Backend

Сервис обмена сообщениями в реальном времени на FastAPI, WebSocket, Redis pub/sub и PostgreSQL.

## Особенности

- REST API для управления пользователями и сообщениями
- WebSocket для обмена сообщениями в реальном времени
- Redis pub/sub для масштабируемых уведомлений
- PostgreSQL для постоянного хранения данных
- Docker Compose для удобного деплоя
- Полное покрытие тестами

## Технологический стек

- **FastAPI** — современный Python-веб-фреймворк
- **PostgreSQL** — реляционная база данных
- **Redis** — кеширование и pub/sub-уведомления
- **SQLAlchemy** — ORM для работы с базой данных
- **Pydantic** — валидация данных
- **WebSocket** — обмен сообщениями в реальном времени
- **Docker** — контейнеризация
- **Pytest** — фреймворк для тестирования

## Быстрый старт

### Через Docker Compose

```bash
git clone <repository-url>
cd mini-chat-backend
docker-compose up --build
```

Сервис будет доступен по адресу: http://localhost:8000

### Для разработки

```bash
# Запустить PostgreSQL и Redis
docker-compose up postgres redis

# Установить переменные окружения
export DATABASE_URL="postgresql://chat_user:chat_password@localhost:5432/chat_db"
export REDIS_URL="redis://localhost:6379"

# Установить зависимости
pip install -r requirements.txt

# Запустить приложение
uvicorn main:app --reload
```

## Документация API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Эндпоинты API

### Пользователи
- `POST /users/` — создать пользователя
- `GET /users/` — получить список пользователей
- `GET /users/{user_id}` — получить пользователя по ID

### Сообщения
- `POST /messages/` — отправить сообщение
- `GET /messages/` — получить историю сообщений
- `GET /messages/recent/{user_id}` — получить последние сообщения из кеша

### WebSocket
- `WS /ws/{user_id}` — соединение для чата в реальном времени

### Система
- `GET /health` — проверка работоспособности сервиса

## Тестирование

```bash
# Запуск всех тестов
pytest

# Запуск с подробным выводом
pytest -v

# Запуск конкретного теста
pytest test_api.py::TestUsers::test_create_user
```

## Примеры использования

### Создание пользователя

```bash
curl -X POST "http://localhost:8000/users/" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "display_name": "Alice Smith"}'
```

### Отправка сообщения

```bash
curl -X POST "http://localhost:8000/messages/" \
  -H "Content-Type: application/json" \
  -d '{"sender_id": 1, "receiver_id": 2, "content": "Привет!"}'
```

### WebSocket-соединение

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/1');

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Новое сообщение:', message);
};

ws.send(JSON.stringify({type: 'ping'}));
```

## Архитектура

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   PostgreSQL     │    │     Redis       │
│                 │    │                  │    │                 │
│ • REST API      │    │ • Users          │    │ • Message Cache │
│ • WebSocket     │────│ • Messages       │────│ • Pub/Sub       │
│ • Pub/Sub       │    │ • History        │    │ • Sessions      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Переменные окружения

```bash
DATABASE_URL=postgresql://chat_user:chat_password@localhost:5432/chat_db
REDIS_URL=redis://localhost:6379
```

## Лицензия

MIT License
