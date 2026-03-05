from pathlib import Path
from uuid import uuid4
from typing import Any
import json
import streamlit as st

from common.chat_store import ChatStore
from core.acts import get_act_sources, get_constitution_source
from core.memory import build_running_summary, compose_memory_context

class LegalAdvisorUI:
    def __init__(self, chain, chat_store: ChatStore):
        self.chain = chain
        self.chat_store = chat_store

    @staticmethod
    def _ensure_state():
        if "device_id" not in st.session_state:
            uid = st.query_params.get("uid")
            if not uid:
                uid = str(uuid4())
                st.query_params["uid"] = uid
            st.session_state.device_id = uid

        if "active_thread_id" not in st.session_state:
            st.session_state.active_thread_id = None

        if "thread_search" not in st.session_state:
            st.session_state.thread_search = ""

    def _ensure_default_thread(self, user_id: str) -> str:
        threads = self.chat_store.list_threads(user_id=user_id)
        if not threads:
            thread_id = str(uuid4())
            self.chat_store.create_thread(user_id=user_id, thread_id=thread_id, scope_act="All")
            return thread_id
        return threads[0]["thread_id"]

    @staticmethod
    def _extract_source(source: Any) -> tuple[dict[str, Any], str]:
        if isinstance(source, dict):
            metadata = source.get("metadata", {}) or {}
            content = source.get("page_content", "")
            return metadata, content

        metadata = getattr(source, "metadata", {}) or {}
        content = getattr(source, "page_content", "")
        return metadata, content

    @staticmethod
    def _serialize_sources(sources: list[Any]) -> list[dict[str, Any]]:
        serialized = []
        for source in sources:
            metadata, content = LegalAdvisorUI._extract_source(source)
            serialized.append(
                {
                    "metadata": metadata,
                    "page_content": content,
                }
            )
        return serialized

    @staticmethod
    def _render_sources(sources, key_prefix: str):
        if not sources:
            return

        normalized_sources = []
        for source in sources:
            metadata, content = LegalAdvisorUI._extract_source(source)
            normalized_sources.append({"metadata": metadata, "page_content": content})

        def preview_text(text: str, limit: int = 320) -> str:
            compact = " ".join((text or "").split())
            if len(compact) <= limit:
                return compact
            return compact[:limit].rstrip() + "…"

        with st.expander(f"📚 Sources ({len(sources)} retrieved)", expanded=False):
            acts = sorted({s["metadata"].get("act_abbrev") or "N/A" for s in normalized_sources})
            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieved", len(normalized_sources))
            col2.metric("Acts", len(acts))
            col3.metric("With Citation", sum(1 for s in normalized_sources if s["metadata"].get("citation")))

            sort_choice = st.selectbox(
                "Sort sources by",
                ["Retrieved order", "Act", "Citation"],
                index=0,
                key=f"{key_prefix}_sort_sources",
            )

            if sort_choice == "Act":
                ordered_sources = sorted(
                    normalized_sources,
                    key=lambda s: (
                        s["metadata"].get("act_abbrev") or "",
                        s["metadata"].get("citation") or "",
                    ),
                )
            elif sort_choice == "Citation":
                ordered_sources = sorted(
                    normalized_sources,
                    key=lambda s: (
                        s["metadata"].get("citation") or "",
                        s["metadata"].get("act_abbrev") or "",
                    ),
                )
            else:
                ordered_sources = list(normalized_sources)

            overview_tab, detailed_tab = st.tabs(["Quick View", "Detailed View"])

            with overview_tab:
                for idx, source in enumerate(ordered_sources, 1):
                    metadata, content = source["metadata"], source["page_content"]
                    citation = metadata.get("citation", "Unknown source")
                    act_name = metadata.get("act", "")
                    act_abbrev = metadata.get("act_abbrev", "")
                    chapter = metadata.get("chapter", "")
                    source_type = metadata.get("source_type", "")
                    jurisdiction = metadata.get("jurisdiction", "")

                    with st.container(border=True):
                        st.markdown(f"**{idx}. {citation}**")
                        metadata_line = " • ".join(
                            [
                                value
                                for value in [act_abbrev, act_name, chapter, source_type, jurisdiction]
                                if value
                            ]
                        )
                        if metadata_line:
                            st.caption(metadata_line)
                        st.write(preview_text(content))

            with detailed_tab:
                for idx, source in enumerate(ordered_sources, 1):
                    metadata, content = source["metadata"], source["page_content"]
                    citation = metadata.get("citation", "Unknown source")
                    act_name = metadata.get("act", "")
                    header = f"{idx}. {citation}"
                    if act_name and citation and act_name not in citation:
                        header += f" - {act_name}"

                    with st.expander(header, expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            if metadata.get("act"):
                                st.markdown(f"**Act:** {metadata.get('act')}")
                            if metadata.get("chapter"):
                                st.markdown(f"**Chapter:** {metadata.get('chapter')}")
                        with col2:
                            if metadata.get("jurisdiction"):
                                st.markdown(f"**Jurisdiction:** {metadata.get('jurisdiction')}")
                            if metadata.get("source_type"):
                                st.markdown(f"**Type:** {metadata.get('source_type')}")
                        st.markdown("---")
                        st.write(content)

    def render(self):
        st.set_page_config(page_title="Indian Legal Advisor", page_icon="⚖️", layout="wide")
        st.title("⚖️ Indian Legal Advisor")
        st.caption("Ask questions about Indian law with retrieval-grounded answers and cited sources.")

        self._ensure_state()
        user_id = st.session_state.device_id

        if not st.session_state.active_thread_id:
            st.session_state.active_thread_id = self._ensure_default_thread(user_id)

        root = Path(__file__).resolve().parents[1]
        act_sources = [get_constitution_source(root)] + get_act_sources(root)
        options = [("All", "All Acts")]
        for act in act_sources:
            options.append((act.act_abbrev, f"{act.act_abbrev} - {act.act}"))

        with st.sidebar:
            st.subheader("Settings")
            act_labels = [label for _, label in options]
            act_map = {label: abbrev for abbrev, label in options}

            st.markdown("---")
            st.subheader("Threads")

            st.text_input(
                "Search threads",
                key="thread_search",
                placeholder="Type to filter by title",
            )

            all_threads = self.chat_store.list_threads(
                user_id=user_id,
                search=st.session_state.thread_search,
            )

            if st.button("New Thread", use_container_width=True):
                new_thread_id = str(uuid4())
                self.chat_store.create_thread(user_id=user_id, thread_id=new_thread_id, scope_act="All")
                st.session_state.active_thread_id = new_thread_id
                st.rerun()

            if not all_threads:
                st.info("No threads found for current filter.")
            else:
                current_ids = [thread["thread_id"] for thread in all_threads]
                if st.session_state.active_thread_id not in current_ids:
                    st.session_state.active_thread_id = current_ids[0]

                selected_thread_id = st.radio(
                    "Choose thread",
                    options=current_ids,
                    index=current_ids.index(st.session_state.active_thread_id),
                    format_func=lambda thread_id: next(
                        (
                            ("📌 " if thread.get("pinned") else "") + thread["title"]
                            for thread in all_threads
                            if thread["thread_id"] == thread_id
                        ),
                        "Untitled",
                    ),
                    key="thread_selector",
                )

                if selected_thread_id != st.session_state.active_thread_id:
                    st.session_state.active_thread_id = selected_thread_id
                    st.rerun()

                active_thread = self.chat_store.get_thread(
                    user_id=user_id,
                    thread_id=st.session_state.active_thread_id,
                )

                if active_thread:
                    current_scope = active_thread.get("scope_act") or "All"
                    current_scope_label = next(
                        (label for abbr, label in options if abbr == current_scope),
                        "All Acts",
                    )
                    selected_scope_label = st.selectbox(
                        "Thread Scope",
                        act_labels,
                        index=act_labels.index(current_scope_label),
                        key=f"scope_{active_thread['thread_id']}",
                    )
                    selected_scope = act_map.get(selected_scope_label, "All")
                    if selected_scope != current_scope:
                        self.chat_store.set_thread_scope(
                            user_id=user_id,
                            thread_id=active_thread["thread_id"],
                            scope_act=selected_scope,
                        )
                        st.rerun()

                    pin_label = "Unpin Thread" if active_thread.get("pinned") else "Pin Thread"
                    if st.button(pin_label, use_container_width=True):
                        self.chat_store.set_thread_pinned(
                            user_id=user_id,
                            thread_id=active_thread["thread_id"],
                            pinned=not bool(active_thread.get("pinned")),
                        )
                        st.rerun()

                    rename_text = st.text_input(
                        "Rename active thread",
                        value=active_thread["title"],
                        key=f"rename_{active_thread['thread_id']}",
                    )
                    rename_col, delete_col = st.columns(2)
                    with rename_col:
                        if st.button("Save Name", use_container_width=True):
                            cleaned = rename_text.strip() or "New Chat"
                            self.chat_store.rename_thread(
                                user_id=user_id,
                                thread_id=active_thread["thread_id"],
                                title=cleaned,
                            )
                            st.rerun()

                    with delete_col:
                        if st.button("Delete", use_container_width=True):
                            self.chat_store.delete_thread(
                                user_id=user_id,
                                thread_id=active_thread["thread_id"],
                            )
                            st.session_state.active_thread_id = self._ensure_default_thread(user_id)
                            st.rerun()

                    export_payload = self.chat_store.export_thread(
                        user_id=user_id,
                        thread_id=active_thread["thread_id"],
                    )
                    if export_payload:
                        file_title = (active_thread["title"] or "thread").strip().replace(" ", "_")[:40]
                        st.download_button(
                            "Export Chat (JSON)",
                            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                            file_name=f"{file_title}_{active_thread['thread_id'][:8]}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

        active_thread_id = st.session_state.active_thread_id
        active_thread = self.chat_store.get_thread(user_id=user_id, thread_id=active_thread_id)
        act_abbrev = active_thread.get("scope_act") if active_thread else "All"
        messages = self.chat_store.get_messages(active_thread_id)

        for index, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    self._render_sources(message["sources"], key_prefix=f"hist_{index}")

        query = st.chat_input("Ask a legal question")
        if not query:
            return

        active_thread = self.chat_store.get_thread(user_id=user_id, thread_id=active_thread_id)
        if active_thread and active_thread["title"] == "New Chat":
            auto_title = " ".join(query.split())[:60].strip() or "New Chat"
            self.chat_store.rename_thread(
                user_id=user_id,
                thread_id=active_thread_id,
                title=auto_title,
            )

        self.chat_store.add_message(
            thread_id=active_thread_id,
            role="user",
            content=query,
            sources=[],
        )

        messages = self.chat_store.get_messages(active_thread_id)
        summary = self.chat_store.get_summary(active_thread_id)
        memory_context = compose_memory_context(
            summary=summary,
            messages=messages,
            recent_messages=8,
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_text = ""
            sources = []

            with st.spinner("Analyzing relevant provisions…"):
                for event in self.chain.stream(
                    {
                        "query": query,
                        "act": act_abbrev,
                        "chat_history": memory_context,
                    }
                ):
                    event_type = event.get("type")
                    if event_type == "token":
                        answer_text += event.get("content", "")
                        placeholder.markdown(answer_text + "▌")
                    elif event_type == "done":
                        answer_text = event.get("content", answer_text)
                        sources = event.get("sources", [])

            placeholder.markdown(answer_text)
            if sources:
                self._render_sources(sources, key_prefix=f"current_{active_thread_id}")

        serialized_sources = self._serialize_sources(sources)
        self.chat_store.add_message(
            thread_id=active_thread_id,
            role="assistant",
            content=answer_text,
            sources=serialized_sources,
        )

        updated_messages = self.chat_store.get_messages(active_thread_id)
        if len(updated_messages) > 8:
            archived_messages = updated_messages[:-8]
            new_summary = build_running_summary(summary, archived_messages)
            self.chat_store.set_summary(active_thread_id, new_summary)

        st.rerun()

