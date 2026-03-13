from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

from stores.chat_store import ChatStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DynamoChatStore(ChatStore):
    def __init__(
        self,
        table_name: str,
        region_name: str | None = None,
        session: boto3.Session | None = None,
    ):
        _boto_session: Any = session or boto3.Session()
        self._dynamodb: Any = _boto_session.resource("dynamodb", region_name=region_name)
        self._table_name = table_name
        self._ensure_table()
        self.table = self._dynamodb.Table(table_name)

    def _ensure_table(self) -> None:
        from botocore.exceptions import ClientError
        try:
            self._dynamodb.Table(self._table_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            self._dynamodb.create_table(
                TableName=self._table_name,
                AttributeDefinitions=[
                    {"AttributeName": "PK", "AttributeType": "S"},
                    {"AttributeName": "SK", "AttributeType": "S"},
                ],
                KeySchema=[
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "SK", "KeyType": "RANGE"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            self._dynamodb.Table(self._table_name).wait_until_exists()

    def _thread_pk(self, user_id: str) -> str:
        return f"USER#{user_id}"

    @staticmethod
    def _thread_sk(thread_id: str) -> str:
        return f"THREAD#{thread_id}"

    @staticmethod
    def _message_sk(created_at: str) -> str:
        return f"MSG#{created_at}"

    @staticmethod
    def _memory_sk() -> str:
        return "MEMORY#SUMMARY"

    def create_thread(
        self,
        user_id: str,
        thread_id: str,
        title: str = "New Chat",
        *,
        scope_act: str = "All",
    ) -> dict[str, Any]:
        now = utc_now_iso()
        item = {
            "PK": self._thread_pk(user_id),
            "SK": self._thread_sk(thread_id),
            "entity_type": "thread",
            "thread_id": thread_id,
            "user_id": user_id,
            "title": title,
            "scope_act": scope_act,
            "pinned": 0,
            "created_at": now,
            "updated_at": now,
        }
        self.table.put_item(Item=item)
        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "title": title,
            "scope_act": scope_act,
            "pinned": 0,
            "created_at": now,
            "updated_at": now,
        }

    def list_threads(self, user_id: str, search: str = "") -> list[dict[str, Any]]:
        response = self.table.query(
            KeyConditionExpression=Key("PK").eq(self._thread_pk(user_id)) & Key("SK").begins_with("THREAD#")
        )
        items = response.get("Items", [])
        filtered: list[dict[str, Any]] = []
        search_value = search.strip().lower()
        for item in items:
            title = item.get("title", "")
            if search_value and search_value not in title.lower():
                continue
            filtered.append(
                {
                    "thread_id": item.get("thread_id"),
                    "user_id": item.get("user_id"),
                    "title": title,
                    "scope_act": item.get("scope_act", "All"),
                    "pinned": int(item.get("pinned", 0)),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                }
            )

        return sorted(filtered, key=lambda row: (row.get("pinned", 0), row.get("updated_at", "")), reverse=True)

    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        response = self.table.get_item(
            Key={"PK": self._thread_pk(user_id), "SK": self._thread_sk(thread_id)}
        )
        item = response.get("Item")
        if not item:
            return None
        return {
            "thread_id": item.get("thread_id"),
            "user_id": item.get("user_id"),
            "title": item.get("title", "New Chat"),
            "scope_act": item.get("scope_act", "All"),
            "pinned": int(item.get("pinned", 0)),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
        }

    def set_thread_scope(self, user_id: str, thread_id: str, scope_act: str) -> None:
        self.table.update_item(
            Key={"PK": self._thread_pk(user_id), "SK": self._thread_sk(thread_id)},
            UpdateExpression="SET scope_act = :scope, updated_at = :updated",
            ExpressionAttributeValues={":scope": scope_act, ":updated": utc_now_iso()},
        )

    def set_thread_pinned(self, user_id: str, thread_id: str, pinned: bool) -> None:
        self.table.update_item(
            Key={"PK": self._thread_pk(user_id), "SK": self._thread_sk(thread_id)},
            UpdateExpression="SET pinned = :pinned",
            ExpressionAttributeValues={":pinned": 1 if pinned else 0},
        )

    def rename_thread(self, user_id: str, thread_id: str, title: str) -> None:
        self.table.update_item(
            Key={"PK": self._thread_pk(user_id), "SK": self._thread_sk(thread_id)},
            UpdateExpression="SET title = :title, updated_at = :updated",
            ExpressionAttributeValues={":title": title, ":updated": utc_now_iso()},
        )

    def delete_thread(self, user_id: str, thread_id: str) -> None:
        thread = self.get_thread(user_id=user_id, thread_id=thread_id)
        if not thread:
            return

        self.table.delete_item(Key={"PK": self._thread_pk(user_id), "SK": self._thread_sk(thread_id)})

        messages = self.get_messages(thread_id)
        for msg in messages:
            self.table.delete_item(
                Key={
                    "PK": f"THREAD#{thread_id}",
                    "SK": self._message_sk(msg.get("created_at", "")),
                }
            )

        self.table.delete_item(Key={"PK": f"THREAD#{thread_id}", "SK": self._memory_sk()})

    def touch_thread(self, thread_id: str) -> None:
        return None

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        created_at = utc_now_iso()
        item = {
            "PK": f"THREAD#{thread_id}",
            "SK": self._message_sk(created_at),
            "entity_type": "message",
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": created_at,
        }
        self.table.put_item(Item=item)

    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        response = self.table.query(
            KeyConditionExpression=Key("PK").eq(f"THREAD#{thread_id}") & Key("SK").begins_with("MSG#")
        )
        items = response.get("Items", [])
        messages: list[dict[str, Any]] = []
        for item in items:
            messages.append(
                {
                    "thread_id": thread_id,
                    "role": item.get("role", "assistant"),
                    "content": item.get("content", ""),
                    "sources": item.get("sources", []),
                    "created_at": item.get("created_at", ""),
                }
            )
        messages.sort(key=lambda row: row.get("created_at", ""))
        return messages

    def get_summary(self, thread_id: str) -> str:
        response = self.table.get_item(Key={"PK": f"THREAD#{thread_id}", "SK": self._memory_sk()})
        item = response.get("Item")
        return item.get("summary", "") if item else ""

    def set_summary(self, thread_id: str, summary: str) -> None:
        self.table.put_item(
            Item={
                "PK": f"THREAD#{thread_id}",
                "SK": self._memory_sk(),
                "entity_type": "memory",
                "thread_id": thread_id,
                "summary": summary,
                "updated_at": utc_now_iso(),
            }
        )

    def export_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        thread = self.get_thread(user_id=user_id, thread_id=thread_id)
        if not thread:
            return None
        return {
            "thread": thread,
            "summary": self.get_summary(thread_id),
            "messages": self.get_messages(thread_id),
        }