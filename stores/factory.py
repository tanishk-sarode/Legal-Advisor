from __future__ import annotations

from pathlib import Path

from common.config import CHAT_DB_PATH, CHAT_STORE_BACKEND, DYNAMODB_TABLE, REGION
from stores.chat_store import ChatStore
from stores.dynamo_chat_store import DynamoChatStore
from stores.sqlite_chat_store import SQLiteChatStore


def build_chat_store(project_root: Path) -> ChatStore:
    if CHAT_STORE_BACKEND == "sqlite":
        return SQLiteChatStore(project_root / CHAT_DB_PATH)

    return DynamoChatStore(
        table_name=DYNAMODB_TABLE,
        region_name=REGION,
    )
