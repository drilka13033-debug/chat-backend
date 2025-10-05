from pydantic import BaseModel, validator
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
