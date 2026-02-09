INDEX_NAME = "tanishk-rag-index-1"
REGION = "us-east-2"
LLM_MODEL = "meta.llama3-3-70b-instruct-v1:0"
LLM_TEMPERATURE = 0.1
INGEST = False  # Set True to run ingestion

from opensearchpy import RequestsHttpConnection

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter
)
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline
)

from common.aws_setup import awsauth, embedding_function

# -------------------------
# AWS / OpenSearch Settings
# -------------------------
AOSS_URL = "https://hcv0472oypdyengtsl48.us-east-2.aoss.amazonaws.com"

# -------------------------
# Vector Stores
# -------------------------
def _build_vectorstore():
    return OpenSearchVectorSearch(
        opensearch_url=AOSS_URL,
        index_name=INDEX_NAME,
        embedding_function=embedding_function,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )

vectorstore = _build_vectorstore()

semantic_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 16,
        "fetch_k": 48
    }
)

# -------------------------
# Compressor (Late Stage)
# -------------------------
compressor = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsClusteringFilter(
            embeddings=embedding_function,
            num_clusters=6
        ),
        EmbeddingsRedundantFilter(
            embeddings=embedding_function
        )
    ]
)
