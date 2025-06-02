import logging
import yaml
import json
from typing import List, Dict
from datetime import datetime, timedelta, time
from prefect import flow, serve
from prefect.client.schemas.schedules import IntervalSchedule

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from project.settings import _settings
from project.storage.s3 import S3Connector
from project.utils.models import RAGModels, KServeEmbeddings
from project.utils.retrievers import OpenSearchBM25Retriever

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

rag_models: Dict[str, RAGModels] = {}

# Init RAG models
for model_name, model_params in _settings.embedder_models.items():
    models_prefix = None
    if _settings.cluster:
        if _settings.embedders_endpoint is None:
            raise ValueError("You must set env `EMBEDDERS_ENDPOINT` in cluster setup!")
        embedder = KServeEmbeddings(
            endpoint=_settings.embedders_endpoint,
            model_name=model_name,
            bs=model_params["bs"],
        )
        models_prefix = "KServe"
    else:
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_params["model_kwargs"],
            encode_kwargs=model_params["encode_kwargs"],
        )
        models_prefix = "hf"

    rag_model = RAGModels(
        models_prefix=models_prefix,
        embedder=embedder,
        reranker=_settings.reranker_model_name,
    )
    rag_models[rag_model.beauty_model_name] = rag_model

sources_path = "sources.yml"
with open(sources_path, "r") as f:
    sources = yaml.safe_load(f)

profagro_params = sources["profagro"]
companies = profagro_params["company_support"]


@flow
def profagro_index_etl(model: str, company: str):
    # Read the sources.yml file
    logger.info(f"{_settings.indexer_s3_bucket}")
    logger.info(f"{_settings.indexer_s3_access_key}")
    logger.info(f"{_settings.indexer_s3_secret_key}")
    logger.info(f"{_settings.indexer_s3_endpoint}")
    rag_model = rag_models[model]

    sources_path = "sources.yml"
    with open(sources_path, "r") as f:
        sources = yaml.safe_load(f)

    profagro_params = sources["profagro"]

    storage = S3Connector(
        bucket_name=_settings.indexer_s3_bucket,
        access_key=_settings.indexer_s3_access_key,
        secret_key=_settings.indexer_s3_secret_key,
        endpoint_url=_settings.indexer_s3_endpoint,
    )

    opensearch = OpenSearchBM25Retriever.create(
        host=_settings.opensearch_host,
        port=_settings.opensearch_port,
        username=_settings.opensearch_username,
        password=_settings.opensearch_password,
        index_name=f"{profagro_params['base_index_name']}_{rag_model.beauty_model_name}_{company}",
        synonyms=profagro_params.get("synonyms", None),
        drop_old=True,
    )

    milvus_vectorstore = Milvus(
        embedding_function=rag_model.embedder,
        collection_name=f"{profagro_params['base_index_name']}_{rag_model.beauty_model_name}_{company}",
        connection_args={
            "uri": _settings.milvus_endpoint,
            "db_name": profagro_params["milvus_db_name"],
        },
        drop_old=True,
        auto_id=True,
    )

    # TODO: можно распаралелить обработку
    for filename, content in tqdm(
        storage.get_files(
            files=profagro_params["files"],
            reg_exp=rf"{profagro_params['reg_exp']}",
            recursive=profagro_params["recursive"],
        )
    ):
        json_data = json.loads(content.decode("utf-8"))
        company_name = json_data["metadata"]["collection_search"]
        if company != company_name:
            continue
        doc = Document(
            page_content=json_data["page_content"], metadata=json_data["metadata"]
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=profagro_params["chunk_size"],
            chunk_overlap=profagro_params["chunk_overlap"],
            length_function=rag_model.length_function,
            separators=profagro_params["separators"],
        )

        chunks = splitter.split_documents([doc])

        # TODO: добавляем название файла в текст чанка, мб не лучшая стратегия
        for chunk in chunks:
            chunk.page_content = f"{filename} {chunk.page_content}"

        milvus_vectorstore.add_documents(chunks)
        opensearch.add_documents(chunks)

    opensearch._update_synonyms()


if __name__ == "__main__":
    serve(
        *[
            profagro_index_etl.to_deployment(
                name=f"{rag_model_name}_{company}",
                parameters={"model": rag_model_name, "company": company},
                schedule=IntervalSchedule(
                    interval=timedelta(days=1),
                    anchor_date=datetime.combine(
                        (datetime.now() + timedelta(days=1)).date(), time(7, 0)
                    ),
                ),
            )
            for rag_model_name in rag_models.keys()
            for company in companies
        ]
    )
