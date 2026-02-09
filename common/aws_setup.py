import boto3
from common.config import REGION
from langchain_aws import BedrockEmbeddings
from requests_aws4auth import AWS4Auth

session = boto3.Session(profile_name="sandbox")

bedrock_client = session.client(
    service_name="bedrock-runtime",
    region_name=REGION
)

embedding_function = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"
)

credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "aoss",
    session_token=credentials.token
)
