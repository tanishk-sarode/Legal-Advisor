from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st

from stores.chat_store import ChatStore


class SessionManager:
    """Encapsulates Streamlit session/query state and local device id persistence."""

    @staticmethod
    def _device_id_file() -> Path:
        root = Path(__file__).resolve().parents[1]
        return root / ".streamlit" / "device_id.txt"

    @classmethod
    def _load_persisted_device_id(cls) -> str | None:
        try:
            path = cls._device_id_file()
            if not path.exists():
                return None
            value = path.read_text(encoding="utf-8").strip()
            return value or None
        except OSError:
            return None

    @classmethod
    def _save_persisted_device_id(cls, device_id: str) -> None:
        try:
            path = cls._device_id_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(device_id, encoding="utf-8")
        except OSError:
            return

    @classmethod
    def ensure_state(cls) -> None:
        if "device_id" not in st.session_state:
            uid = st.query_params.get("uid")
            if not uid:
                uid = cls._load_persisted_device_id()
            if not uid:
                uid = str(uuid4())
                cls._save_persisted_device_id(uid)
            st.query_params["uid"] = uid
            st.session_state.device_id = uid

        if "active_thread_id" not in st.session_state:
            st.session_state.active_thread_id = None

        if "thread_search" not in st.session_state:
            st.session_state.thread_search = ""

        if "prefill_query" not in st.session_state:
            st.session_state.prefill_query = ""


    @staticmethod
    def queue_prompt(prompt: str) -> None:
        st.session_state.prefill_query = prompt
        st.rerun()

    @staticmethod
    def pop_prefill_query() -> str:
        query = st.session_state.prefill_query
        st.session_state.prefill_query = ""
        return query

    @staticmethod
    def ensure_default_thread(chat_store: ChatStore, user_id: str) -> str:
        threads = chat_store.list_threads(user_id=user_id)
        if not threads:
            thread_id = str(uuid4())
            chat_store.create_thread(user_id=user_id, thread_id=thread_id, scope_act="All")
            return thread_id
        return threads[0]["thread_id"]
