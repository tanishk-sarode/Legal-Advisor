from pathlib import Path
import logging
import time
from uuid import uuid4

import streamlit as st

from core.acts import get_act_sources, get_constitution_source
from infra.session_manager import SessionManager
from services.conversation_service import ConversationService
from services.memory_service import MemoryService
from stores.chat_store import ChatStore
from ui.chat_view import (
    render_empty_state,
    render_message_history,
    render_thread_context_bar,
    render_user_message,
)
from ui.layout import inject_styles, render_app_header
from ui.sidebar import render_sidebar
from ui.sources_view import render_sources

logger = logging.getLogger(__name__)


class LegalAdvisorUI:
    def __init__(self, chain, chat_store: ChatStore):
        self.chain = chain
        self.chat_store = chat_store
        self.session_manager = SessionManager()
        self.memory_service = MemoryService(chat_store)
        self.conversation_service = ConversationService(chain, chat_store, self.memory_service)

    @staticmethod
    def _build_act_options() -> list[tuple[str, str]]:
        root = Path(__file__).resolve().parents[1]
        act_sources = [get_constitution_source(root)] + get_act_sources(root)
        options = [("All", "All Acts")]
        for act in act_sources:
            options.append((act.act_abbrev, f"{act.act_abbrev} - {act.act}"))
        return options

    def render(self) -> None:
        st.set_page_config(page_title="Indian Legal Advisor", page_icon="⚖️", layout="wide")
        inject_styles()
        self.session_manager.ensure_state()

        user_id = st.session_state.device_id
        if not st.session_state.active_thread_id:
            st.session_state.active_thread_id = self.session_manager.ensure_default_thread(self.chat_store, user_id)

        options = self._build_act_options()
        render_sidebar(
            chat_store=self.chat_store,
            session_manager=self.session_manager,
            user_id=user_id,
            options=options,
        )

        active_thread_id = st.session_state.active_thread_id
        active_thread = self.chat_store.get_thread(user_id=user_id, thread_id=active_thread_id)
        act_abbrev = active_thread.get("scope_act") if active_thread else "All"
        messages = self.chat_store.get_messages(active_thread_id)

        render_app_header()
        render_thread_context_bar(active_thread, len(messages))

        if not messages:
            render_empty_state(self.session_manager.queue_prompt)

        render_message_history(messages, render_sources)

        query = st.chat_input("Ask about an article, section, procedure, or legal risk")
        if not query and st.session_state.prefill_query:
            query = self.session_manager.pop_prefill_query()
        if not query:
            return

        request_id = str(uuid4())[:8]
        turn_start = time.perf_counter()
        logger.info(
            "Turn started: request_id=%s user_id=%s thread_id=%s act=%s",
            request_id,
            user_id,
            active_thread_id,
            act_abbrev or "All",
        )

        render_user_message(query)

        with st.chat_message("assistant"):
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>Assistant</div>", unsafe_allow_html=True)
                placeholder = st.empty()
                answer_text = ""
                sources = []

                with st.spinner("Searching the most relevant provisions..."):

                    def on_token(partial_answer: str, chunk_count: int) -> None:
                        placeholder.markdown(partial_answer + "▌")
                        if chunk_count == 1:
                            logger.info("UI received first stream token: request_id=%s", request_id)

                    result = self.conversation_service.process_turn(
                        user_id=user_id,
                        thread_id=active_thread_id,
                        query=query,
                        act_abbrev=act_abbrev,
                        request_id=request_id,
                        on_token=on_token,
                    )
                    answer_text = result["answer_text"]
                    sources = result["sources"]
                    chunk_count = result["chunk_count"]

                    logger.info(
                        "UI stream completed: request_id=%s chunks=%s answer_chars=%s sources=%s",
                        request_id,
                        chunk_count,
                        len(answer_text),
                        len(sources),
                    )

                placeholder.markdown(answer_text)

            if sources:
                render_sources(sources, key_prefix=f"current_{active_thread_id}")

        logger.info(
            "Turn finished: request_id=%s elapsed=%.2fs",
            request_id,
            time.perf_counter() - turn_start,
        )
        st.rerun()
