from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

from typing import Tuple


async def create_router_query_engine(
    document_fp: str,
    verbose: bool = True,
) -> RouterQueryEngine:
    # 加载pdf文档
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()

    # 切块，每块1024
    splitter = SentenceSplitter(chunk_size=1024)
    # 提取文档节点
    nodes = splitter.get_nodes_from_documents(documents)

    # LLM model
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    # embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    # 定义summary index工具
    summary_index = SummaryIndex(nodes)
    # vector store index
    vector_index = VectorStoreIndex(nodes)

    # summary query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # 定义vector query engine工具
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "用户生成摘要的工具"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "常规检索的工具"
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=verbose
    )

    return query_engine


async def create_doc_tools(
    document_fp: str,
    doc_name: str,
    verbose: bool = True,
) -> Tuple[QueryEngineTool, QueryEngineTool]:
    
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    # 创建数据节点
    nodes = splitter.get_nodes_from_documents(documents)

    # LLM model
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    # embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    # summary index
    summary_index = SummaryIndex(nodes)
    # vector store index
    vector_index = VectorStoreIndex(nodes)

    # summary query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # vector query engine
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        name=f"{doc_name}_summary_query_engine_tool",
        query_engine=summary_query_engine,
        description=(
            f"对与摘要问题相关的有用的 {doc_name}."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        name=f"{doc_name}_vector_query_engine_tool",
        query_engine=vector_query_engine,
        description=(
            f"对于上下文常规检索有用的 {doc_name}."
        ),
    )

    return vector_tool, summary_tool