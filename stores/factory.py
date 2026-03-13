from __future__ import annotations

from pathlib import Path

from common.aws_setup import session as aws_session
from common.config import DYNAMODB_TABLE, REGION
from stores.chat_store import ChatStore
from stores.dynamo_chat_store import DynamoChatStore


def build_chat_store(project_root: Path) -> ChatStore:
    return DynamoChatStore(
        table_name=DYNAMODB_TABLE,
        region_name=REGION,
        session=aws_session,
    )
