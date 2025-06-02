from pydantic import BaseModel
import asyncio
from project.models import SearchRequest
import json
import requests


class RequestToRAG(BaseModel):
    """
    Делает запрос в RAG (Retrieval Augmented Generation) [официальные руководства, технические пособия], если пользователь спрашивает информацию по селькохозяйственной технике / деталям / оборудованию и тд:

    Причем запрос нужно переформулировать, чтобы он был более понятен поисковой системе.
    Например,
    "Что делать, если система WindControl не компенсирует ветер?" -> RequestToRAG("система WindControl не компенсирует ветер"),
    "Как добавить новую карту поля в систему?" -> RequestToRAG("Как добавить новую карту поля в систему?")
    "Как изменить ширину захвата на ZG-TS?фыафыаdsfadsfsdav" -> RequestToRAG("Как изменить ширину захвата на ZG-TS?")
    "Что такое аукс н к аматрон 4??" -> RequestToRAG("Что такое AUX-N к AmaTron 4?")
    """

    user_message: str


def request_to_rag(user_message: str, company: str):

    url = "http://localhost:8200/api/retrieve"
    headers = {"Content-Type": "application/json"}
    payload = {
        "query": user_message,
        "company": company.lower(),
        "model": "kserve_baai_bge_m3",
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        search_response = response.json()
        top_k_documents = search_response.get("reranked", [])
        result = ["Релевантная информация для ответа на вопрос:"]
        metadata = []
        for idx, text in enumerate(top_k_documents):
            result.append(f"{idx + 1}: {text['page_content']}")
            metadata.append(text.get("metadata", {}))
        return "\n\n".join(result), metadata
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


map_name_func_to_func = {"RequestToRAG": request_to_rag}

from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Optional, Type, List


class RAGInput(BaseModel):
    query: str = Field(
        description="Запрос пользователя в базу знаний (поисковую систему)"
    )
    company: str = Field(
        description="Компания, технику которой пользуется пользователь"
    )


class RAGOutput(BaseModel):
    content: str = Field(description="Релевантная информация для ответа на вопрос:")
    metadata: List[str] = Field(description="Метаданные для этих документов")


class RequestToRAGTool(BaseTool):
    name: str = "RequestToRAG"
    description: str = """
    Делает запрос в RAG (Retrieval Augmented Generation) [официальные руководства, технические пособия], если пользователь спрашивает информацию по селькохозяйственной технике / деталям / оборудованию и тд:

    Причем запрос нужно переформулировать, чтобы он был более понятен поисковой системе.
    Например,
    "Что делать, если система WindControl не компенсирует ветер?" -> RequestToRAG("система WindControl не компенсирует ветер"),
    "Как добавить новую карту поля в систему?" -> RequestToRAG("Как добавить новую карту поля в систему?")
    "Как изменить ширину захвата на ZG-TS?фыафыаdsfadsfsdav" -> RequestToRAG("Как изменить ширину захвата на ZG-TS?")
    "Что такое аукс н к аматрон 4??" -> RequestToRAG("Что такое AUX-N к AmaTron 4?")
    """
    args_schema: Type[BaseModel] = RAGInput
    return_schema: Type[BaseModel] = RAGOutput
    response_format: str = "content_and_artifact"

    def _run(self, query: str, company: str):
        url = "http://localhost:8200/api/retrieve"
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "company": company.lower(),
            "model": "kserve_baai_bge_m3",
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            search_response = response.json()
            top_k_documents = search_response.get("reranked", [])
            result = ["Релевантная информация для ответа на вопрос:"]
            metadata = []
            for idx, text in enumerate(top_k_documents):
                result.append(f"{idx + 1}: {text['page_content']}")
                metadata.append(json.dumps(text.get("metadata", {})))
            return "\n\n".join(result), metadata
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
