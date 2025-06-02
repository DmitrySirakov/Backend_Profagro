import torch

from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    # Use cluster setup or use models in this instance. For cluster setup endpoints in embedder_models must be set
    cluster: bool = Field(alias="CLUSTER", default=False)
    embedders_endpoint: str = Field(alias="EMBEDDERS_ENDPOINT", default=None)

    indexer_s3_bucket: str = Field(alias="INDEXER_S3_BUCKET")
    indexer_s3_access_key: str = Field(alias="INDEXER_S3_ACCESS_KEY")
    indexer_s3_secret_key: str = Field(alias="INDEXER_S3_SECRET_KEY")
    indexer_s3_endpoint: str = Field(alias="INDEXER_S3_ENDPOINT")

    milvus_endpoint: str = Field(alias="MILVUS_ENDPOINT")
    opensearch_host: str = Field(alias="OPENSEARCH_HOST")
    opensearch_port: str = Field(alias="OPENSEARCH_PORT")
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
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"


_settings = Settings()
