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
        """Creates a new thread for the user with the given thread_id and title. Returns thread metadata."""
        raise NotImplementedError

    @abstractmethod
    def list_threads(self, user_id: str, search: str = "") -> list[dict[str, Any]]:
        """Returns a list of thread metadata for the user, optionally filtered by search string in title."""
        raise NotImplementedError

    @abstractmethod
    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        """Returns thread metadata for the given thread_id if it belongs to the user, else None."""
        raise NotImplementedError

    @abstractmethod
    def set_thread_scope(self, user_id: str, thread_id: str, scope_act: str) -> None:
        """Sets the scope_act for the thread, which can be used to guide retrieval and answer generation."""
        raise NotImplementedError

    @abstractmethod
    def set_thread_pinned(self, user_id: str, thread_id: str, pinned: bool) -> None:
        """Sets the pinned status for the thread."""

        raise NotImplementedError

    @abstractmethod
    def rename_thread(self, user_id: str, thread_id: str, title: str) -> None:
        """Renames the thread."""
        raise NotImplementedError

    @abstractmethod
    def delete_thread(self, user_id: str, thread_id: str) -> None:
        """Deletes the thread and all associated messages."""
        raise NotImplementedError

    @abstractmethod
    def touch_thread(self, thread_id: str) -> None:
        """Updates the last_updated timestamp for the thread."""
        raise NotImplementedError

    @abstractmethod
    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Adds a message to the thread."""
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        """Returns a list of messages for the given thread_id."""
        raise NotImplementedError

    @abstractmethod
    def get_summary(self, thread_id: str) -> str:
        """Returns the summary for the given thread_id."""
        raise NotImplementedError

    @abstractmethod
    def set_summary(self, thread_id: str, summary: str) -> None:
        """Sets the summary for the given thread_id."""
        raise NotImplementedError

    @abstractmethod
    def export_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        """Exports the thread data for backup or transfer. Returns None if thread not found or not owned by user."""
        raise NotImplementedError