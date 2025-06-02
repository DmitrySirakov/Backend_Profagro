import re
import logging
import requests

from typing import List, Any, Optional, Union
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


from typing import Any, Dict, Tuple, List, Optional

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field


class KServeCrossEncoder(BaseModel, BaseCrossEncoder):
    endpoint: str
    model_name: str
    beauty_model_name: str
    bs: int = 32

    def __init__(self, endpoint: str, model_name: str, bs: int = 32) -> None:
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            beauty_model_name=re.sub(r"[/-]", "_", model_name).lower(),
            bs=bs,
        )

    class Config:
        extra = "forbid"

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a KServe CrossEncoder inference.

        Args:
            text_pairs: The list of text text_pairs to score the similarity.

        Returns:
            List of scores, one for each pair.
        """
        all_scores = []
        for i in range(0, len(text_pairs), self.bs):
            response = requests.post(
                url=f"{self.endpoint}/v1/models/{self.beauty_model_name}:predict",
                json={"texts": text_pairs[i : i + self.bs]},
            ).json()
            all_scores.extend(response["scores"])
        return all_scores


class KServeEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model_name: str, bs: int = 32) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.model_name = model_name
        self.beauty_model_name = re.sub(r"[/-]", "_", self.model_name).lower()
        self.bs = bs

    def embed_documents(
        self, texts: List[str], text_type: str = "passage"
    ) -> List[List[float]]:
        """Compute doc embeddings using a KServe model.

        Args:
            texts: The list of texts to embed.
            text_type: The type of text to embed. Can be "passage" or "query".

        Returns:
            List of embeddings, one for each text.
        """

        # This string is taken langchain_huggingface HuggingFaceEmbeddings class
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        if text_type == "query":
            texts = [("query", text) for text in texts]
        else:
            texts = [("passage", text) for text in texts]
        all_embeddings = []

        for i in range(0, len(texts), self.bs):
            response = requests.post(
                url=f"{self.endpoint}/v1/models/{self.beauty_model_name}:predict",
                json={"texts": texts[i : i + self.bs]},
            ).json()

            embeddings = response["embeddings"]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a KServe model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text], text_type="query")[0]


class RAGModels:
    def __init__(
        self,
        embedder: Embeddings,
        models_prefix: str,
        reranker: Optional[Union[BaseCrossEncoder, str]] = None,
    ) -> None:
        """Init RAG models.

        Args:
            embedder: Embedder model.
            reranker: Reranker model.
            models_prefix: Prefix for model names. Used in collection naming.
        """
        super().__init__()
        self.embedder_model_name = embedder.model_name
        self.beauty_embedder_model_name = embedder.beauty_model_name
        self.embedder_config = AutoConfig.from_pretrained(self.embedder_model_name)
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(
            self.embedder_model_name
        )
        self.embedder = embedder
        self.beauty_model_name = (
            f"{models_prefix}_{self.beauty_embedder_model_name}".lower()
        )

        if isinstance(reranker, str):
            self.reranker_model_name = reranker
            self.beauty_reranker_model_name = re.sub(
                r"[/-]", "_", self.reranker_model_name
            ).lower()
        else:
            self.reranker_model_name = reranker.model_name
            self.beauty_reranker_model_name = reranker.beauty_model_name
            self.reranker = reranker

        self.reranker_config = AutoConfig.from_pretrained(self.reranker_model_name)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_name
        )

    def length_function(self, text: str) -> int:
        max_tokens = max(
            len(
                self.embedder_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self.embedder_config.max_position_embeddings,
                )
            ),
            len(
                self.reranker_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self.reranker_config.max_position_embeddings,
                )
            ),
        )
        return max_tokens
