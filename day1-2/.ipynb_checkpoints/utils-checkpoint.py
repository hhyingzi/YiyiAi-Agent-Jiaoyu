from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

async def create_router_query_engine(
    document_fp: str,
    verbose: bool = True,
) -> RouterQueryEngine:
    # 加载本地pdf数据
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()
    
    # 分文本块，每块1024
    splitter = SentenceSplitter(chunk_size=1024)
    # 创建文档节点
    nodes = splitter.get_nodes_from_documents(documents)
    
    # LLM model，这里用的是api
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
        query_engine=summary_query_engine,
        description=(
            "适用于生成与卖油翁课堂相关的摘要问题。"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "适用于检索卖油翁特定上下文的问题。"
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