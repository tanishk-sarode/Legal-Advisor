from __future__ import annotations

from core.memory import build_running_summary, compose_memory_context
from stores.chat_store import ChatStore


class MemoryService:
    """Handles memory-context construction and summary rollover policy."""

    def __init__(self, chat_store: ChatStore, recent_window: int = 8):
        self.chat_store = chat_store
        self.recent_window = recent_window

    def build_context(self, thread_id: str) -> tuple[str, str]:
        messages = self.chat_store.get_messages(thread_id)
        summary = self.chat_store.get_summary(thread_id)
        memory_context = compose_memory_context(
            summary=summary,
            messages=messages,
            recent_messages=self.recent_window,
        )
        return memory_context, summary

    def maybe_update_summary(self, thread_id: str, prior_summary: str) -> None:
        updated_messages = self.chat_store.get_messages(thread_id)
        if len(updated_messages) <= self.recent_window:
            return

        archived_messages = updated_messages[: -self.recent_window]
        new_summary = build_running_summary(prior_summary, archived_messages)
        self.chat_store.set_summary(thread_id, new_summary)
