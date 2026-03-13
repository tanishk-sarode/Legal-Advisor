import os
from dotenv import load_dotenv

load_dotenv()  # load .env into os.environ (no-op if already loaded)

INDEX_NAME = os.getenv("INDEX_NAME", "tanishk-rag-index-async")
REGION = os.getenv("AWS_REGION", "us-east-2")
LLM_MODEL = os.getenv("LLM_MODEL", "meta.llama3-3-70b-instruct-v1:0")
LLM_TEMPERATURE = 0.1
INGEST = False  # Set True to run ingestion

from opensearchpy import RequestsHttpConnection

from langchain_community.vectorstores import OpenSearchVectorSearch

from common.aws_setup import awsauth, embedding_function, dynamodb

# -------------------------
# AWS Settings
# -------------------------
AOSS_URL = os.environ["AOSS_URL"]          # required – set in .env
S3_BUCKET = os.getenv("S3_BUCKET", "il-advisor")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "legal_advisor_chat")
CHAT_STORE_BACKEND = os.getenv("CHAT_STORE_BACKEND", "dynamodb")

# -------------------------
# Hybrid Retrieval Tunables
# -------------------------
HYBRID_K = 20                         # docs per hybrid call (kNN + BM25 each)
HYBRID_FALLBACK_THRESHOLD = 6         # trigger fallback leg if primary returns fewer docs
ENRICHMENT_CONFIDENCE_THRESHOLD = 0.5 # skip citation-fetch if trigger confidence below this

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

# -------------------------
# File DynamoDB Setup 
# -------------------------

def setup_dynamodb():
    dynamodb.create_table(
        TableName=DYNAMODB_TABLE,
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST"
    )
