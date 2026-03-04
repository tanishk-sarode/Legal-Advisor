from langchain_aws import ChatBedrockConverse
from common.aws_setup import bedrock_client


def get_answer_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
    client=bedrock_client,
    model="us.anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.4
    )
