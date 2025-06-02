import torch
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Dict, Any


def load_yaml_prompt(file_path: str, key: str) -> str:
    """Загружает определенный промпт из YAML файла."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return data.get(key, "")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    cluster: bool = Field(alias="CLUSTER", default=False)
    host: str = Field(alias="HOST", default="0.0.0.0")
    port: int = Field(alias="PORT", default=8400)
    workers: int = Field(alias="WORKERS", default=5)

    indexer_s3_bucket: str = Field(alias="INDEXER_S3_BUCKET")
    indexer_s3_access_key: str = Field(alias="INDEXER_S3_ACCESS_KEY")
    indexer_s3_secret_key: str = Field(alias="INDEXER_S3_SECRET_KEY")
    indexer_s3_endpoint: str = Field(alias="INDEXER_S3_ENDPOINT")

    embedders_endpoint: str = Field(alias="EMBEDDERS_ENDPOINT", default=None)
    reranker_endpoint: str = Field(alias="RERANKER_ENDPOINT", default=None)
    reranker_model_name: str = Field(alias="RERANKER_MODEL_NAME")
    reranker_top_n: int = Field(alias="RERANKER_TOP_N", default=4)

    milvus_endpoint: str = Field(alias="MILVUS_ENDPOINT")
    milvus_retriever_top_k: int = Field(alias="MILVUS_RETRIEVER_TOP_K", default=25)

    opensearch_host: str = Field(alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(alias="OPENSEARCH_PORT")
    opensearch_username: str = Field(alias="OPENSEARCH_USERNAME")
    opensearch_password: str = Field(alias="OPENSEARCH_PASSWORD")

    embedder_models: Dict[str, Dict[str, Any]] = {
        # "intfloat/multilingual-e5-large": {
        #     "model_kwargs": {
        #         "prompts": {"query": "query: ", "passage": "passage: "},
        #         "device": "cuda" if torch.cuda.is_available() else "cpu",
        #     },
        #     "encode_kwargs": {"normalize_embeddings": True, "prompt_name": "passage"},
        #     "bs": 32,
        # },
        # "intfloat/multilingual-e5-large-instruct": {
        #     "model_kwargs": {
        #         "prompts": {
        #             "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        #             "passage": "",
        #         },
        #         "device": "cuda" if torch.cuda.is_available() else "cpu",
        #     },
        #     "encode_kwargs": {"normalize_embeddings": True, "prompt_name": "passage"},
        #     "bs": 32,
        # },
        "BAAI/bge-m3": {
            "model_kwargs": {
                "prompts": {"query": "query: ", "passage": "passage: "},
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            "encode_kwargs": {"normalize_embeddings": True, "prompt_name": "passage"},
            "bs": 32,
        },
    }
    openai_key: str = Field(alias="OPENAI_KEY")
    proxy_url: str = Field(alias="PROXY_URL")
    openai_agent_model: str = Field(alias="OPENAI_AGENT_MODEL")
    agent_system_prompt: str = load_yaml_prompt(
        "./project/system_prompt.yaml", "assistant_profagro_prompt_v1"
    )
    prompt_template: str = Field(
        alias="PROMPT_TEMPLATE",
        default="""
Ты — ассистент ПрофАгро по технике Amazone (производитель сельскохозяйственных и коммунальных машин, поставщику системных решений для интеллектуального растениеводства) и не только.
Твоя задача — максимально точно и полно отвечать на вопросы пользователей, используя информацию из базы знаний и документации.

**Основные задачи:**
1. Отвечать на вопросы, используя контекст, содержащий информацию из базы знаний.
2. Предоставлять конкретные и структурированные ответы.
3. Отвечай развернуто и подробно. Уделяй особое внимание цифрам, моделям, используй их при ответе.

**Вопрос пользователя:**
{question}

**Контекст (информация из базы знаний):**
{context}

**Рекомендации:**
- Всегда говори конкретику, ссылайся на конкретные модели, на конкретные разделы / программы в документации.

Если ты не можешь дать ответ, скажи: "Эта информация отсутствует в базе знаний." Не придумывай ответов.
""",
    )
    number_of_messages_to_keep_in_chat_history: int = Field(
        alias="NUMBER_OF_MESSAGES_TO_KEEP_IN_CHAT_HISTORY", default=10
    )
    company_support: list[str] = ["amazone", "kverneland"]
    gigachat_cred: str = Field(alias="GIGACHAT_CREDENTIALS")


_settings = Settings()
