from abc import ABC, abstractmethod
from typing import Any


class ChatStore(ABC):
    @abstractmethod
    def create_thread(
        self,
        user_id: str,
        thread_id: str,
        title: str = "New Chat",
        *,
        scope_act: str = "All",
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_threads(self, user_id: str, search: str = "") -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def set_thread_scope(self, user_id: str, thread_id: str, scope_act: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_thread_pinned(self, user_id: str, thread_id: str, pinned: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def rename_thread(self, user_id: str, thread_id: str, title: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_thread(self, user_id: str, thread_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def touch_thread(self, thread_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_summary(self, thread_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_summary(self, thread_id: str, summary: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        raise NotImplementedError