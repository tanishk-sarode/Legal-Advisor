import boto3
import os
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from requests_aws4auth import AWS4Auth

load_dotenv()  # load .env into os.environ (no-op if already loaded)

REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_PROFILE = os.getenv("AWS_PROFILE", "sandbox")
EMBED_MODEL = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")

session = boto3.Session(profile_name=AWS_PROFILE)

bedrock_client = session.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

embedding_function = BedrockEmbeddings(
    client=bedrock_client,
    model_id=EMBED_MODEL
)

credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "aoss",
    session_token=credentials.token
)

dynamodb = boto3.client("dynamodb", 
                        region_name=REGION, 
                        aws_access_key_id=credentials.access_key, 
                        aws_secret_access_key=credentials.secret_key, 
                        aws_session_token=credentials.token)


