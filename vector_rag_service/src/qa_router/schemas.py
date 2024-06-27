from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, field_validator


class Role(str, Enum):
    USER = "user"
    AI = "ai"


class Message(BaseModel):
    role: Role
    content: str


class Input(BaseModel):
    vault_id: str = Field(examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"])
    query: str = Field(description="User's new message")
    history: List[dict] = Field(examples=[[{"role": "user", "content": "Hello"}]], description="Chat history")


class SearchResult(BaseModel):
    document_id: UUID
    document_name: str
    information: str


class Answer(BaseModel):
    content: str
    traceback: List[SearchResult]
