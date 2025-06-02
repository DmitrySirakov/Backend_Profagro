from pydantic import BaseModel
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


class QueryExpansionRequest(BaseModel):
    chat_history: List[ChatMessage]
    company: str


class QueryExpansionResponse(BaseModel):
    expanded_questions: List[str]
