"""Wrapper around Elasticsearch vector database."""

from __future__ import annotations

import uuid
import requests
import logging
from typing import Any, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class OpenSearchBM25Retriever(BaseRetriever):
    client: Any
    index_name: str
    host: str
    port: int
    username: str
    password: str
    k1: float = 2.0
    b: float = 0.75
    synonyms: Optional[List[str]] = None

    @classmethod
    def create(
        cls,
        index_name: str,
        host: str,
        port: int,
        username: str,
        password: str,
        drop_old: bool = False,
        k1: float = 2.0,
        b: float = 0.75,
        synonyms: Optional[List[str]] = None,
    ) -> OpenSearchBM25Retriever:
        """
        Create an OpenSearchBM25Retriever instance.

        Args:
            index_name: Name of the index to use in OpenSearch.
            host: Hostname of the OpenSearch instance.
            port: Port number of the OpenSearch instance.
            username: Username for authentication.
            password: Password for authentication.
            k1: BM25 parameter k1. Default is 2.0.
            b: BM25 parameter b. Default is 0.75.
            synonyms: Optional list of synonyms to use in the search analyzer.

        Returns:
            An instance of OpenSearchBM25Retriever.
        """
        from opensearchpy import OpenSearch

        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

        # TODO: расхаркодить настройки
        settings = {
            "settings": {
                "index": {"number_of_shards": 2},
                "analysis": {
                    "analyzer": {
                        "default": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "russian_snowball"],
                        },
                        "default_search": {
                            "tokenizer": "standard",
                            "filter": (
                                [
                                    "lowercase",
                                    "russian_stop",
                                    "russian_snowball",
                                    "graph_synonyms",
                                ]
                                if synonyms is not None
                                else ["lowercase", "russian_stop", "russian_snowball"]
                            ),
                        },
                    },
                    "filter": {
                        "graph_synonyms": {
                            "type": "index_synonym_graph",
                            "index": ".synonyms",
                            "expand": True,
                            "lenient": False,
                            "username": username,
                            "password": password,
                        },
                        "russian_snowball": {"type": "snowball", "language": "Russian"},
                        "russian_stop": {"type": "stop", "stopwords": "_russian_"},
                    },
                },
            }
        }

        if not client.indices.exists(index=index_name):
            client.indices.create(
                index=index_name,
                body=settings,
            )
        elif drop_old:
            client.indices.delete(index=index_name)
            client.indices.create(
                index=index_name,
                body=settings,
            )
        else:
            logger.warning(f"Index {index_name} already exists.")

        if client.indices.exists(index=index_name):
            if drop_old:
                logger.info(f"Удаляем старый индекс {index_name}.")
                try:
                    client.indices.delete(index=index_name)
                    logger.info(f"Индекс {index_name} успешно удален.")
                except Exception as e:
                    logger.error(f"Ошибка при удалении индекса {index_name}: {e}")
            else:
                logger.warning(
                    f"Индекс {index_name} уже существует. Пропускаем создание."
                )
        else:
            logger.info(f"Индекс {index_name} не существует, создаем новый индекс.")
            try:
                client.indices.create(index=index_name, body=settings)
                logger.info(f"Индекс {index_name} успешно создан.")
            except Exception as e:
                logger.error(f"Ошибка при создании индекса {index_name}: {e}")
        obj = cls(
            client=client,
            index_name=index_name,
            host=host,
            port=port,
            username=username,
            password=password,
            k1=k1,
            b=b,
            synonyms=synonyms,
        )
        if synonyms is not None:
            obj._create_synonyms_index()
            obj._update_synonyms()
        return obj

    def _update_synonyms(self):
        # TODO: поискать в Python клиенте, как это делается
        response = requests.post(
            f"https://{self.host}:{self.port}/_plugins/_refresh_search_analyzers/{self.index_name}",
            auth=(self.username, self.password),
            verify=False,
        )
        self.client.indices.refresh(index=self.index_name)
        if response.status_code not in [200, 201]:
            logger.warning(
                f"Refresh search analyzers failed: {response.status_code} - {response.text}"
            )

    def _create_synonyms_index(self) -> None:
        response = requests.post(
            f"https://{self.host}:{self.port}/.synonyms/_doc/synonyms",
            json={"synonyms": self.synonyms},
            auth=(self.username, self.password),
            verify=False,
        )
        if response.status_code not in [200, 201]:
            logger.warning(
                f"Synonyms update failed: {response.status_code} - {response.text}"
            )

    def add_texts(
        self,
        texts: Iterable[str],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from opensearchpy.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import opensearch python package. "
                "Please install it with `pip install opensearch-py`."
            )
        requests = []
        ids = []
        for i, text in enumerate(texts):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        _, failures = bulk(self.client, requests)
        if failures:
            logger.warning(f"Failed to index documents: {failures}")

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def add_documents(
        self, documents: Iterable[Document], refresh_indices: bool = True
    ) -> List[str]:
        texts = [doc.page_content for doc in documents]

        if refresh_indices:
            try:
                logger.info(
                    f"Refreshing index {self.index_name} before adding documents."
                )
                self.client.indices.refresh(index=self.index_name)
            except Exception as e:
                logger.error(f"Failed to refresh index {self.index_name}: {e}")

        try:
            return self.add_texts(texts, refresh_indices)
        except Exception as e:
            logger.error(f"Failed to add documents to index {self.index_name}: {e}")
            raise

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_dict = {"query": {"match": {"content": query}}}
        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(Document(page_content=r["_source"]["content"]))
        return docs
