import os
from langchain_aws import ChatBedrockConverse
from common.aws_setup import bedrock_client

_ANSWER_MODEL = os.getenv("ANSWER_MODEL", "us.anthropic.claude-3-haiku-20240307-v1:0")
_FILTER_MODEL = os.getenv("FILTER_MODEL", "us.anthropic.claude-3-haiku-20240307-v1:0")


def get_answer_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        client=bedrock_client,
        model=_ANSWER_MODEL,
        temperature=0.4,
    )


def get_source_filter_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        client=bedrock_client,
        model=_FILTER_MODEL,
        temperature=0.0,
    )
