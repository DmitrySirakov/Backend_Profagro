from pydantic import BaseModel, constr
from typing import Literal, List


class SearchRequest(BaseModel):
    query: str
    model: str
    company: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str


class AgentRequest(BaseModel):
    chat_history: List[ChatMessage]
    company: str
