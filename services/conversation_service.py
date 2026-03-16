from __future__ import annotations

from typing import Any, Callable

from services.memory_service import MemoryService
from stores.chat_store import ChatStore


class ConversationService:
    """Orchestrates one conversation turn while keeping UI rendering separate."""

    def __init__(self, chain, chat_store: ChatStore, memory_service: MemoryService):
        self.chain = chain
        self.chat_store = chat_store
        self.memory_service = memory_service

    @staticmethod
    def _extract_source(source: Any) -> tuple[dict[str, Any], str]:
        if isinstance(source, dict):
            metadata = source.get("metadata", {}) or {}
            content = source.get("page_content", "")
            return metadata, content

        metadata = getattr(source, "metadata", {}) or {}
        content = getattr(source, "page_content", "")
        return metadata, content

    @classmethod
    def _serialize_sources(cls, sources: list[Any]) -> list[dict[str, Any]]:
        serialized = []
        for source in sources:
            metadata, content = cls._extract_source(source)
            serialized.append({"metadata": metadata, "page_content": content})
        return serialized

    @staticmethod
    def auto_title_from_query(query: str) -> str:
        return " ".join(query.split())[:60].strip() or "New Chat"

    def process_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        act_abbrev: str | None,
        request_id: str,
        on_token: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        active_thread = self.chat_store.get_thread(user_id=user_id, thread_id=thread_id)
        if active_thread and active_thread.get("title") == "New Chat":
            self.chat_store.rename_thread(
                user_id=user_id,
                thread_id=thread_id,
                title=self.auto_title_from_query(query),
            )

        self.chat_store.add_message(
            thread_id=thread_id,
            role="user",
            content=query,
            sources=[],
        )

        memory_context, prior_summary = self.memory_service.build_context(thread_id)

        answer_text = ""
        sources: list[Any] = []
        chunk_count = 0

        for event in self.chain.stream(
            {
                "query": query,
                "act": act_abbrev,
                "chat_history": memory_context,
                "request_id": request_id,
            }
        ):
            event_type = event.get("type")
            if event_type == "token":
                chunk_count += 1
                answer_text += event.get("content", "")
                if on_token is not None:
                    on_token(answer_text, chunk_count)
            elif event_type == "done":
                answer_text = event.get("content", answer_text)
                sources = event.get("sources", [])

        serialized_sources = self._serialize_sources(sources)
        self.chat_store.add_message(
            thread_id=thread_id,
            role="assistant",
            content=answer_text,
            sources=serialized_sources,
        )
        self.memory_service.maybe_update_summary(thread_id, prior_summary)

        return {
            "answer_text": answer_text,
            "sources": sources,
            "chunk_count": chunk_count,
        }
