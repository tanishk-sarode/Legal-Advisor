import os

INDEX_NAME = "tanishk-rag-index-async"
REGION = "us-east-2"
LLM_MODEL = "meta.llama3-3-70b-instruct-v1:0"
LLM_TEMPERATURE = 0.1
INGEST = False  # Set True to run ingestion

from opensearchpy import RequestsHttpConnection

from langchain_community.vectorstores import OpenSearchVectorSearch

from common.aws_setup import awsauth, embedding_function, dynamodb

# -------------------------
# AWS Settings
# -------------------------
AOSS_URL = "https://hcv0472oypdyengtsl48.us-east-2.aoss.amazonaws.com"
S3_BUCKET = "il-advisor"
DYNAMODB_TABLE = "legal_advisor_chat"
CHAT_STORE_BACKEND = os.getenv("CHAT_STORE_BACKEND", "dynamodb").strip().lower()
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat_memory.db")

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
