from langchain_aws import ChatBedrockConverse
from common.aws_setup import bedrock_client
from common.config import LLM_MODEL, LLM_TEMPERATURE


def get_retriever_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        client=bedrock_client,
        model=LLM_MODEL,
        temperature=0
    )


def get_answer_llm() -> ChatBedrockConverse:
    return ChatBedrockConverse(
        client=bedrock_client,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )
