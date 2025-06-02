import logging
import uvicorn

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import json
import uuid
from langchain_core.runnables.base import RunnableSequence
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from project.utils.models import RAGModels, KServeEmbeddings, KServeCrossEncoder
from project.utils.retrievers import OpenSearchBM25Retriever
from project.utils.runnables import RerankerRunnable
from project.models import (
    SearchRequest,
    AgentRequest,
    QueryExpansionRequest,
    QueryExpansionResponse,
)
from project.settings import _settings, load_yaml_prompt
from project.utils.agent import RequestToRAG, map_name_func_to_func, RequestToRAGTool
from project.utils.utils import format_message_for_logging
from project.models import ChatMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

search_chains: Dict[str, Dict[str, RunnableSequence]] = {}
retriever_chains: Dict[str, Dict[str, RunnableSequence]] = {}

logger = logging.getLogger(__name__)


def normalize_object_key(local_path: str) -> str:
    """
    Нормализует локальный путь к объекту, удаляя префикс "./", если он присутствует.
    """
    if local_path.startswith("./"):
        return local_path[2:]
    return local_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_chains
    global client_openai
    global tools
    global agent_executor
    client_openai = openai.OpenAI(
        api_key=_settings.openai_key, base_url=_settings.proxy_url
    )
    tools = [openai.pydantic_function_tool(RequestToRAG)]

    for model_name, model_params in _settings.embedder_models.items():
        models_prefix = None
        if _settings.cluster:
            if (
                _settings.embedders_endpoint is None
                or _settings.reranker_endpoint is None
            ):
                raise ValueError(
                    "You must set env `EMBEDDERS_ENDPOINT` in cluster setup!"
                )
            embedder = KServeEmbeddings(
                endpoint=_settings.embedders_endpoint,
                model_name=model_name,
                bs=model_params["bs"],
            )
            reranker = KServeCrossEncoder(
                endpoint=_settings.reranker_endpoint,
                model_name=_settings.reranker_model_name,
            )
            models_prefix = "KServe"

        else:
            embedder = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_params["model_kwargs"],
                encode_kwargs=model_params["encode_kwargs"],
            )
            reranker = HuggingFaceCrossEncoder(model_name=_settings.reranker_model_name)
            models_prefix = "hf"

        rag_model = RAGModels(
            models_prefix=models_prefix, embedder=embedder, reranker=reranker
        )
        reranker = RerankerRunnable(
            CrossEncoderReranker(model=reranker, top_n=_settings.reranker_top_n)
        )

        connectors = {}
        for company in _settings.company_support:
            opensearch = OpenSearchBM25Retriever.create(
                host=_settings.opensearch_host,
                port=_settings.opensearch_port,
                username=_settings.opensearch_username,
                password=_settings.opensearch_password,
                index_name=f"profagro_{rag_model.beauty_model_name}_{company}",
            )
            milvus_retriever = Milvus(
                embedding_function=rag_model.embedder,
                collection_name=f"profagro_{rag_model.beauty_model_name}_{company}",
                connection_args={
                    "uri": _settings.milvus_endpoint,
                    "db_name": "profagro",
                },
            ).as_retriever(search_kwargs={"k": _settings.milvus_retriever_top_k})

            if company not in connectors:
                connectors[company] = {}

            connectors[company]["opensearch"] = opensearch
            connectors[company]["milvus"] = milvus_retriever

            rag_prompt = PromptTemplate(
                template=_settings.prompt_template,
                input_variables=["context", "question"],
            )

            llm = ChatOpenAI(
                model="gpt-4o",
                streaming=False,
                api_key=_settings.openai_key,
                base_url=_settings.proxy_url,
                temperature=0.5,
            )

            hybrid_retriever = {
                "milvus_retrieved_doc": connectors[company]["milvus"],
                "bm25_retrieved_doc": connectors[company]["opensearch"],
                "query": RunnablePassthrough(),
            } | RunnablePassthrough()
            if company not in retriever_chains:
                retriever_chains[company] = {}

            if company not in search_chains:
                search_chains[company] = {}

            if rag_model.beauty_model_name not in retriever_chains:
                retriever_chains[rag_model.beauty_model_name] = {}

            if rag_model.beauty_model_name not in search_chains:
                search_chains[rag_model.beauty_model_name] = {}

            retriever_chains[rag_model.beauty_model_name][company] = (
                hybrid_retriever
                | {
                    "retriever": RunnablePassthrough(),
                    "reranked": reranker,
                }
            )

            search_chains[rag_model.beauty_model_name][company] = (
                {
                    "context": hybrid_retriever
                    | reranker
                    | (
                        lambda docs: "\n\n".join(
                            doc.page_content.replace("\r\n", " ") for doc in docs
                        )
                    ),
                    "question": RunnablePassthrough(),
                }
                | rag_prompt
                | llm
                | StrOutputParser()
            )
    llm_giga = GigaChat(
        credentials=_settings.gigachat_cred,
        model="GigaChat-Max",
        scope="GIGACHAT_API_CORP",
        verify_ssl_certs=False,
        temperature=0.5,
    )
    giga_tools = [RequestToRAGTool()]
    giga_with_functions = llm_giga.bind_functions(giga_tools)
    agent_executor = create_react_agent(giga_with_functions, giga_tools)

    yield


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

api_router = APIRouter()


@api_router.post("/search")
async def search(request: SearchRequest) -> Dict:
    if request.model not in search_chains:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    return {
        "answer": search_chains[request.model][request.company].invoke(request.query)
    }


@api_router.post("/retrieve")
async def retrieve(request: SearchRequest) -> Dict:
    if request.model not in retriever_chains:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    result = retriever_chains[request.model][request.company].invoke(request.query)
    return {
        "milvus_retrieved_doc": [
            doc.page_content for doc in result["retriever"]["milvus_retrieved_doc"]
        ],
        "bm25_retrieved_doc": [
            doc.page_content for doc in result["retriever"]["bm25_retrieved_doc"]
        ],
        "reranked": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["reranked"]
        ],
    }


@api_router.get("/list_available_models")
async def list_available_models() -> Dict:
    return {"models": list(search_chains.keys())}


async def parse_multi_hop(user_query: str) -> str:
    if user_query[-1] != "}":
        user_query = user_query + "}"
    if user_query[0] != "{":
        user_query = "{" + user_query
    return user_query


async def execute_tool_call(
    func_name: str, args: str, tools_map: dict, company: str
) -> str:
    if func_name.lower() == "requesttorag" and args.count("}{") >= 1:
        cnt_response = args.count("}{") + 1
        cnt_user_message = args.count('{"user_message":')
        if cnt_user_message == cnt_response:
            answers = []
            for user_query in args.split("}{"):
                final_query = await parse_multi_hop(user_query)
                logger.info(final_query)
                answers.append(tools_map[func_name](**json.loads(final_query)))

            return "\n\n".join(answers)

    args = json.loads(args)
    if func_name.lower() == "requesttorag":
        args["company"] = company
    return tools_map[func_name](**args)


@api_router.post("/agent")
async def tool_calling(request: AgentRequest) -> StreamingResponse:
    chat_history = [
        ChatMessage(role="system", content=_settings.agent_system_prompt),
        ChatMessage(
            role="user",
            content=f"Я у тебя спрашиваю про технику компании {request.company.lower()}",
        ),
    ]
    chat_history.extend(
        request.chat_history[-_settings.number_of_messages_to_keep_in_chat_history :]
    )
    tool_messages = []
    logger.info(await format_message_for_logging(chat_history))
    while True:
        response = client_openai.chat.completions.create(
            model=_settings.openai_agent_model,
            messages=chat_history,
            tools=tools,
            stream=True,
        )
        first_chunk = None

        for chunk in response:
            if (
                chunk.choices[0].delta.content is None
                and chunk.choices[0].delta.tool_calls is None
            ):
                continue
            first_chunk = chunk
            break

        logger.info(first_chunk.choices[0].delta)
        if first_chunk.choices[0].delta.content is None:
            if not first_chunk.choices[0].delta.tool_calls:
                continue
            function_name = first_chunk.choices[0].delta.tool_calls[0].function.name
            function_id = first_chunk.choices[0].delta.tool_calls[0].id
            collected_arguments_string = ""
            for chunk in response:
                if not chunk.choices:  # Проверка на пустые choices
                    continue
                logger.info(chunk)
                if (
                    chunk.choices[0].delta.tool_calls
                    and chunk.choices[0].delta.tool_calls[0].function.arguments
                ):
                    collected_arguments_string += (
                        chunk.choices[0].delta.tool_calls[0].function.arguments
                    )

            logger.info(f"{function_name.upper()}: {collected_arguments_string}")
            result, *metadata = await execute_tool_call(
                function_name,
                collected_arguments_string,
                map_name_func_to_func,
                request.company,
            )
            logger.info(f"{function_name.upper()} result: {result}")
            tool_message = {
                "role": "function",
                "tool_call_id": function_id,
                "content": result,
                "name": function_name,
                "metadata": metadata[0],
            }
            tool_messages.append(tool_message)
            tool_message = {
                "role": "function",
                "tool_call_id": function_id,
                "content": result,
                "name": function_name,
            }
            chat_history.append(tool_message)
        else:

            async def generate():
                try:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            try:
                                yield f"event: data\ndata: {json.dumps({'content': content, 'contentType': 'TEXT'}, ensure_ascii=False)}\n\n"
                            except Exception as e:
                                logging.error(f"Ошибка при отправке данных: {e}")
                                break
                    metadata = []
                    if tool_messages:
                        for msg in tool_messages:
                            for meta in msg.get("metadata", {}):
                                if meta:
                                    image_path = meta.get("image", "")
                                    if image_path:
                                        object_key = normalize_object_key(image_path)
                                        meta["image"] = object_key
                                    metadata.append(meta)
                    logger.info(f"Переданная метадата: {metadata}")
                    metadata = metadata[:9].copy()
                    try:
                        yield (
                            f"event: metadata\n"
                            f"data: {json.dumps({'tool_messages': metadata}, ensure_ascii=False)}\n\n"
                        )
                    except Exception as e:
                        logging.error(f"Ошибка при отправке tool_messages: {e}")
                except Exception as e:
                    logging.error(f"Ошибка при генерации данных: {e}")
                finally:
                    try:
                        yield f"event: done\ndata: {json.dumps({'content': '', 'contentType': 'TEXT'}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        logging.error(f"Ошибка при завершении стрима: {e}")

            return StreamingResponse(generate(), media_type="text/event-stream")


@api_router.post("/agent_gigachat")
async def agent_gigachat(request: AgentRequest) -> StreamingResponse:
    chat_history = [
        SystemMessage(content=_settings.agent_system_prompt),
        HumanMessage(
            content=f"Я у тебя спрашиваю про технику компании {request.company.lower()}. И буду спрашивать в дальнейшем только про неё"
        ),
    ]

    for msg in request.chat_history[
        -_settings.number_of_messages_to_keep_in_chat_history :
    ]:
        if msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))
        elif msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
    logging.info(chat_history)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    metadata = {"tool_message": []}

    async def generate():
        try:
            for chunk in agent_executor.stream({"messages": chat_history}, config):
                logging.info(f"CHINK: {chunk}")
                if "tools" in chunk:
                    metadata_raw = chunk["tools"]["messages"][0].artifact
                    for meta in metadata_raw:
                        if meta:
                            meta = json.loads(meta)
                            image_path = meta.get("image", "")
                            if image_path:
                                object_key = normalize_object_key(image_path)
                                meta["image"] = object_key
                            metadata["tool_message"].append(meta)
            logging.info(f"CHONK: {chunk}")
            if chunk["agent"]["messages"][0].content:
                content = chunk["agent"]["messages"][0].content
                logging.info(f"CONTENT: {content}")
                try:
                    yield f"event: data\ndata: {json.dumps({'content': content, 'contentType': 'TEXT'}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logging.error(f"Ошибка при отправке данных: {e}")

            if metadata["tool_message"]:
                try:
                    yield (
                        f"event: metadata\n"
                        f"data: {json.dumps({'tool_messages': metadata['tool_message']}, ensure_ascii=False)}\n\n"
                    )
                except Exception as e:
                    logging.error(f"Ошибка при отправке tool_messages: {e}")

        except Exception as e:
            logging.error(f"Ошибка при генерации данных: {e}")
        finally:
            try:
                yield f"event: done\ndata: {json.dumps({'content': '', 'contentType': 'TEXT'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logging.error(f"Ошибка при завершении стрима: {e}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@api_router.post("/query_expansion", response_model=QueryExpansionResponse)
async def query_expansion(request: QueryExpansionRequest) -> QueryExpansionResponse:
    messages = [
        {
            "role": "system",
            "content": load_yaml_prompt(
                "./project/system_prompt.yaml", "query_expansion_agro_prompt_v1"
            ),
        }
    ]
    for msg in request.chat_history:
        messages.append({"role": msg.role, "content": msg.content})

    client = openai.OpenAI(api_key=_settings.openai_key, base_url=_settings.proxy_url)
    completion = client.chat.completions.create(
        model=_settings.openai_agent_model,
        messages=messages,
        temperature=0.2,
    )
    raw = completion.choices[0].message.content.strip()
    expanded_questions = [q.strip() for q in raw.split("\n") if q.strip()]
    return QueryExpansionResponse(expanded_questions=expanded_questions)


app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, port=_settings.port, host=_settings.host)
