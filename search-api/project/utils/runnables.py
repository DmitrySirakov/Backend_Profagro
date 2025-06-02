from typing import List, Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableConfig
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables.utils import Input, Output


class RerankerRunnable(Runnable):
    def __init__(
        self,
        compressor: BaseDocumentCompressor,
    ):
        self.compressor = compressor

    def _remove_duplicates(
        self,
        retrieved_documents: List[Document],
    ):
        seen_page_contents = set()
        unique_documents = []
        for doc in retrieved_documents:
            if doc.page_content not in seen_page_contents:
                unique_documents.append(doc)
                seen_page_contents.add(doc.page_content)
        return unique_documents

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        milvus_retrieved_doc: List[Document] = input.get("milvus_retrieved_doc")
        bm25_retrieved_doc: List[Document] = input.get("bm25_retrieved_doc")
        query: str = input.get("query")
        unique_documents = self._remove_duplicates(
            milvus_retrieved_doc + bm25_retrieved_doc
        )
        result = self.compressor.compress_documents(unique_documents, query)

        return result
