from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import streamlit as st

from core.acts import get_act_sources, get_constitution_source
from core.memory import build_running_summary, compose_memory_context
from stores.chat_store import ChatStore

logger = logging.getLogger(__name__)


class LegalAdvisorUI:
    def __init__(self, chain, chat_store: ChatStore):
        self.chain = chain
        self.chat_store = chat_store

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
    def _ensure_state(cls):
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
    def _format_relative_time(value: str | None) -> str:
        if not value:
            return "Updated recently"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - parsed
        except ValueError:
            return "Updated recently"

        total_seconds = max(int(delta.total_seconds()), 0)
        if total_seconds < 60:
            return "Updated just now"
        if total_seconds < 3600:
            return f"Updated {total_seconds // 60}m ago"
        if total_seconds < 86400:
            return f"Updated {total_seconds // 3600}h ago"
        return f"Updated {total_seconds // 86400}d ago"

    @staticmethod
    def _thread_title_preview(title: str, limit: int = 46) -> str:
        compact = " ".join((title or "Untitled").split()).strip() or "Untitled"
        if len(compact) <= limit:
            return compact
        return compact[:limit].rstrip() + "..."

    @staticmethod
    def _preview_text(text: str, limit: int = 340) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit].rstrip() + "..."

    def _queue_prompt(self, prompt: str) -> None:
        st.session_state.prefill_query = prompt
        st.rerun()

    def _render_thread_context_bar(self, thread: dict[str, Any] | None, message_count: int) -> None:
        if not thread:
            return

        title = self._thread_title_preview(thread.get("title") or "New Chat", limit=72)
        scope = thread.get("scope_act") or "All"
        updated = self._format_relative_time(thread.get("updated_at"))
        st.markdown(
            """
            <div class="thread-context-bar">
                <div>
                    <div class="eyebrow">Current conversation</div>
                    <div class="thread-context-title">{title}</div>
                </div>
                <div class="thread-context-meta">
                    <span class="scope-chip">{scope}</span>
                    <span class="meta-chip">{message_count} messages</span>
                    <span class="meta-chip">{updated}</span>
                </div>
            </div>
            """.format(
                title=title,
                scope=scope,
                message_count=message_count,
                updated=updated,
            ),
            unsafe_allow_html=True,
        )

    def _render_thread_item(self, thread: dict[str, Any], *, is_active: bool) -> None:
        title = self._thread_title_preview(thread.get("title") or "Untitled")
        scope = thread.get("scope_act") or "All"
        updated = self._format_relative_time(thread.get("updated_at"))
        pin_label = "Pinned" if thread.get("pinned") else "Conversation"
        active_class = " active" if is_active else ""

        st.markdown(
            """
            <div class="thread-item{active_class}">
                <div class="thread-item-title">{title}</div>
                <div class="thread-item-meta">{pin_label} <span class="thread-sep">/</span> {scope} <span class="thread-sep">/</span> {updated}</div>
            </div>
            """.format(
                active_class=active_class,
                title=title,
                pin_label=pin_label,
                scope=scope,
                updated=updated,
            ),
            unsafe_allow_html=True,
        )
        if is_active:
            st.markdown("<div class='active-pill'>Active conversation</div>", unsafe_allow_html=True)
        else:
            if st.button("Open conversation", key=f"thread_open_{thread['thread_id']}", use_container_width=True):
                st.session_state.active_thread_id = thread["thread_id"]
                st.rerun()

    def _render_thread_section(
        self,
        *,
        title: str,
        threads: list[dict[str, Any]],
        active_thread_id: str | None,
    ) -> None:
        if not threads:
            return

        st.markdown(f"<div class='sidebar-label'>{title}</div>", unsafe_allow_html=True)
        for thread in threads:
            self._render_thread_item(thread, is_active=thread.get("thread_id") == active_thread_id)

    def _render_empty_state(self) -> None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="eyebrow">Start here</div>
                <div class="empty-state-title">Ask a legal question with grounded sources</div>
                <div class="empty-state-copy">Use the assistant for sections, articles, procedures, compliance risks, and plain-English explanations of Indian law.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Explain Article 21", key="starter_article_21", use_container_width=True):
                self._queue_prompt("Explain Article 21 of the Constitution in simple terms.")
        with col2:
            if st.button("Find Section 420 IPC", key="starter_420_ipc", use_container_width=True):
                self._queue_prompt("What does Section 420 IPC say, and when does it apply?")
        with col3:
            if st.button("Check bail procedure", key="starter_bail", use_container_width=True):
                self._queue_prompt("What is the general bail procedure in India for a non-bailable offence?")

    def _render_sources(self, sources, key_prefix: str):
        if not sources:
            return

        normalized_sources = []
        for source in sources:
            metadata, content = LegalAdvisorUI._extract_source(source)
            normalized_sources.append({"metadata": metadata, "page_content": content})

        acts = sorted({s["metadata"].get("act_abbrev") or "N/A" for s in normalized_sources})
        source_count = len(normalized_sources)
        cited_count = sum(1 for s in normalized_sources if s["metadata"].get("citation"))

        st.markdown(
            """
            <div class="sources-header">
                <div>
                    <div class="eyebrow">Supporting material</div>
                    <div class="sources-title">Retrieved sources for this answer</div>
                </div>
                <div class="sources-stats">
                    <span class="meta-chip">{source_count} sources</span>
                    <span class="meta-chip">{act_count} acts</span>
                    <span class="meta-chip">{cited_count} with citation</span>
                </div>
            </div>
            """.format(source_count=source_count, act_count=len(acts), cited_count=cited_count),
            unsafe_allow_html=True,
        )

        with st.expander(f"View sources ({source_count})", expanded=False):
            control_col1, control_col2 = st.columns([1.15, 1])
            with control_col1:
                sort_choice = st.selectbox(
                    "Order sources",
                    ["Retrieved order", "Act", "Citation"],
                    index=0,
                    key=f"{key_prefix}_sort_sources",
                )
            with control_col2:
                st.caption("Use Highlights for quick scan or Full text for detailed reading.")

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

            compact_tab, detailed_tab = st.tabs(["Highlights", "Full text"])

            with compact_tab:
                for index, source in enumerate(ordered_sources, 1):
                    metadata, content = source["metadata"], source["page_content"]
                    citation = metadata.get("citation") or f"Source {index}"
                    act_name = metadata.get("act") or ""
                    act_abbrev = metadata.get("act_abbrev") or ""
                    source_type = metadata.get("source_type") or ""
                    jurisdiction = metadata.get("jurisdiction") or ""
                    chapter = metadata.get("chapter") or ""

                    with st.container(border=True):
                        st.markdown(f"#### {citation}")
                        metadata_line = "  |  ".join(
                            value
                            for value in [act_abbrev, act_name, chapter, source_type, jurisdiction]
                            if value
                        )
                        if metadata_line:
                            st.caption(metadata_line)
                        st.write(self._preview_text(content))

            with detailed_tab:
                expand_all = st.checkbox(
                    "Expand all detailed sources",
                    value=False,
                    key=f"{key_prefix}_expand_all_sources",
                )
                for index, source in enumerate(ordered_sources, 1):
                    metadata, content = source["metadata"], source["page_content"]
                    citation = metadata.get("citation") or f"Source {index}"
                    act_name = metadata.get("act") or ""
                    header = citation if not act_name or act_name in citation else f"{citation} - {act_name}"

                    with st.expander(header, expanded=expand_all):
                        meta_left, meta_right = st.columns(2)
                        with meta_left:
                            if metadata.get("act"):
                                st.markdown(f"**Act:** {metadata.get('act')}")
                            if metadata.get("chapter"):
                                st.markdown(f"**Chapter:** {metadata.get('chapter')}")
                            if metadata.get("title"):
                                st.markdown(f"**Title:** {metadata.get('title')}")
                        with meta_right:
                            if metadata.get("jurisdiction"):
                                st.markdown(f"**Jurisdiction:** {metadata.get('jurisdiction')}")
                            if metadata.get("source_type"):
                                st.markdown(f"**Type:** {metadata.get('source_type')}")
                            if metadata.get("act_abbrev"):
                                st.markdown(f"**Act Code:** {metadata.get('act_abbrev')}")
                        st.divider()
                        st.write(content)

    @staticmethod
    def _inject_styles() -> None:
        st.markdown(
            """
            <style>
            :root {
                --bg: #f4f7f7;
                --sidebar-bg: #e8eeef;
                --surface: #ffffff;
                --surface-soft: #f8fbfb;
                --border: #d6e0e0;
                --border-strong: #c6d3d3;
                --text: #162126;
                --muted: #5d6870;
                --accent: #0f6f78;
                --accent-soft: #d9eff0;
                --shadow: 0 12px 30px rgba(15, 23, 31, 0.05);
            }

            html, body, [class*="css"] {
                color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            [data-testid="stAppViewContainer"] {
                background: var(--bg);
            }

            .stApp,
            .stApp > div,
            [data-testid="stMain"],
            section.main {
                background: var(--bg) !important;
            }

            [data-testid="stHeader"] {
                background: rgba(244, 247, 247, 0.92);
            }

            [data-testid="stSidebar"] {
                background: var(--sidebar-bg);
                border-right: 1px solid var(--border);
            }

            [data-testid="stBottomBlockContainer"],
            [data-testid="stBottom"] {
                background: var(--bg) !important;
                border-top: 1px solid var(--border);
            }

            [data-testid="stChatFloatingInputContainer"],
            [data-testid="stChatInputContainer"] {
                background: var(--bg) !important;
                border-top: 1px solid var(--border);
                box-shadow: none !important;
            }

            div[class*="stBottom"],
            div[class*="stBottomBlockContainer"],
            div[class*="stChatFloatingInputContainer"],
            div[class*="stChatInputContainer"],
            footer {
                background: var(--bg) !important;
                border-top-color: var(--border) !important;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1.35rem;
                padding-bottom: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .block-container {
                max-width: 1240px;
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }

            h1,
            [data-testid="stMarkdownContainer"] h1 {
                margin: 0;
                font-size: 2.55rem;
                line-height: 1.05;
                letter-spacing: -0.03em;
                color: var(--text) !important;
            }

            h2, h3, h4,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4 {
                color: var(--text) !important;
                letter-spacing: -0.02em;
            }

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] div,
            [data-testid="stCaptionContainer"] {
                color: var(--text);
            }

            .app-header {
                padding: 0.2rem 0 0.6rem;
                border-bottom: 1px solid var(--border);
                margin-bottom: 1rem;
            }

            .app-subtitle {
                margin-top: 0.45rem;
                max-width: 760px;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.6;
            }

            .eyebrow {
                font-size: 0.74rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
                margin-bottom: 0.45rem;
            }

            .thread-context-bar {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1rem 1.15rem;
                box-shadow: var(--shadow);
                margin: 0.9rem 0 1.35rem;
            }

            .thread-context-title {
                font-size: 1.1rem;
                font-weight: 600;
                line-height: 1.35;
                color: var(--text);
            }

            .thread-context-meta {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.45rem;
            }

            .scope-chip,
            .meta-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.34rem 0.7rem;
                font-size: 0.82rem;
                line-height: 1;
                white-space: nowrap;
            }

            .scope-chip {
                background: var(--accent-soft);
                color: var(--accent);
                font-weight: 600;
            }

            .meta-chip {
                background: var(--surface-soft);
                color: var(--muted);
                border: 1px solid var(--border);
            }

            .sidebar-brand {
                margin-bottom: 1rem;
            }

            .sidebar-title {
                font-size: 1.15rem;
                font-weight: 600;
                letter-spacing: -0.02em;
                color: var(--text);
                margin-bottom: 0.15rem;
            }

            .sidebar-copy {
                font-size: 0.9rem;
                line-height: 1.55;
                color: var(--muted);
                margin-bottom: 1rem;
            }

            .sidebar-label {
                font-size: 0.74rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
                margin: 1rem 0 0.55rem;
            }

            .sidebar-summary {
                font-size: 0.88rem;
                color: var(--muted);
                margin-top: 0.35rem;
                margin-bottom: 0.35rem;
            }

            .thread-item {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 0.85rem 0.9rem;
                margin-bottom: 0.45rem;
            }

            .thread-item.active {
                background: var(--surface);
                border-color: rgba(15, 111, 120, 0.28);
                box-shadow: inset 3px 0 0 var(--accent);
            }

            .thread-item-title {
                font-size: 0.95rem;
                font-weight: 600;
                line-height: 1.42;
                color: var(--text);
            }

            .thread-item-meta {
                margin-top: 0.28rem;
                font-size: 0.8rem;
                color: var(--muted);
                line-height: 1.4;
            }

            .thread-sep {
                color: #9cadb2;
                padding: 0 0.18rem;
            }

            .active-pill {
                display: inline-flex;
                margin-top: 0.35rem;
                margin-bottom: 0.9rem;
                padding: 0.24rem 0.62rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                color: var(--accent);
                background: var(--accent-soft);
            }

            .empty-state {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 251, 251, 0.96));
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 2rem;
                box-shadow: var(--shadow);
                margin: 1.2rem 0 1rem;
            }

            .empty-state-title {
                font-size: 1.5rem;
                font-weight: 600;
                line-height: 1.2;
                color: var(--text);
            }

            .empty-state-copy {
                margin-top: 0.65rem;
                max-width: 720px;
                font-size: 1rem;
                line-height: 1.7;
                color: var(--muted);
            }

            .sources-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
                margin-top: 0.8rem;
                margin-bottom: 0.7rem;
            }

            .sources-title {
                font-size: 1rem;
                font-weight: 600;
                line-height: 1.35;
                color: var(--text);
            }

            .sources-stats {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.4rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--surface);
                border-color: var(--border);
                border-radius: 20px;
                box-shadow: var(--shadow);
            }

            [data-testid="stChatMessage"] {
                background: transparent;
                padding-top: 0.2rem;
                padding-bottom: 0.2rem;
            }

            [data-testid="stChatMessageContent"],
            [data-testid="stChatMessageContent"] p,
            [data-testid="stChatMessageContent"] li,
            [data-testid="stChatMessageContent"] div,
            [data-testid="stChatMessageContent"] span {
                color: var(--text) !important;
                opacity: 1 !important;
            }

            [data-testid="stChatInput"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 18px;
                box-shadow: var(--shadow);
                padding: 0.15rem 0.2rem;
                max-width: 1120px;
                margin: 0 auto;
            }

            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] [data-baseweb="textarea"],
            [data-testid="stChatInput"] [data-baseweb="base-input"],
            [data-testid="stChatInput"] div:has(> textarea[data-testid="stChatInputTextArea"]),
            [data-testid="stChatInput"] div:has(> button[data-testid="stChatInputSubmitButton"]) {
                background: var(--surface) !important;
                border-color: var(--border) !important;
            }

            [data-testid="stChatInput"] textarea,
            [data-testid="stTextInput"] input,
            [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
            [data-testid="stTextArea"] textarea {
                background: var(--surface) !important;
                color: var(--text) !important;
                border-radius: 14px !important;
            }

            [data-testid="stChatInputSubmitButton"] {
                background: var(--surface-soft) !important;
                border: 1px solid var(--border) !important;
                color: var(--muted) !important;
                box-shadow: none !important;
            }

            [data-testid="stChatInputSubmitButton"]:hover {
                border-color: var(--accent) !important;
                color: var(--accent) !important;
            }

            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stChatInput"] textarea::placeholder {
                color: #8c989f !important;
            }

            [data-testid="stButton"] button,
            [data-testid="stDownloadButton"] button {
                border-radius: 12px;
                border: 1px solid var(--border-strong);
                background: var(--surface);
                color: var(--text);
                font-weight: 600;
                min-height: 2.7rem;
            }

            [data-testid="stButton"] button:hover,
            [data-testid="stDownloadButton"] button:hover {
                border-color: var(--accent);
                color: var(--accent);
            }

            [data-testid="stButton"] button[kind="primary"] {
                background: var(--accent);
                color: #ffffff;
                border-color: var(--accent);
            }

            [data-testid="stButton"] button[kind="primary"]:hover {
                background: #0b626a;
                color: #ffffff;
            }

            [data-testid="stExpander"] details {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 16px;
            }

            [data-testid="stExpander"] summary {
                background: var(--surface) !important;
                color: var(--text) !important;
                border-radius: 16px;
            }

            [data-testid="stExpander"] summary * {
                color: var(--text) !important;
                fill: var(--text) !important;
            }

            [data-testid="stTabs"] button[role="tab"] {
                border-radius: 999px;
                background: transparent;
                color: var(--muted);
                font-weight: 600;
                padding: 0.35rem 0.8rem;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                background: var(--accent-soft);
                color: var(--accent);
            }

            hr {
                border-color: var(--border);
            }

            @media (max-width: 1100px) {
                .thread-context-bar,
                .sources-header {
                    flex-direction: column;
                }

                .thread-context-meta,
                .sources-stats {
                    justify-content: flex-start;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render(self):
        st.set_page_config(page_title="Indian Legal Advisor", page_icon="⚖️", layout="wide")
        self._inject_styles()
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
            act_labels = [label for _, label in options]
            act_map = {label: abbrev for abbrev, label in options}

            st.markdown(
                """
                <div class="sidebar-brand">
                    <div class="eyebrow">Workspace</div>
                    <div class="sidebar-title">Indian Legal Advisor</div>
                    <div class="sidebar-copy">A calmer workspace for research, grounded answers, and saved conversations.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("New Conversation", type="primary", use_container_width=True):
                new_thread_id = str(uuid4())
                self.chat_store.create_thread(user_id=user_id, thread_id=new_thread_id, scope_act="All")
                st.session_state.active_thread_id = new_thread_id
                st.rerun()

            st.markdown("<div class='sidebar-label'>Browse conversations</div>", unsafe_allow_html=True)
            st.text_input(
                "Search conversations",
                key="thread_search",
                placeholder="Search saved chats",
                label_visibility="collapsed",
            )

            all_threads = self.chat_store.list_threads(
                user_id=user_id,
                search=st.session_state.thread_search,
            )

            if not all_threads:
                st.info("No conversations match the current filter.")
            else:
                current_ids = [thread["thread_id"] for thread in all_threads]
                if st.session_state.active_thread_id not in current_ids:
                    st.session_state.active_thread_id = current_ids[0]

                pinned_threads = [thread for thread in all_threads if thread.get("pinned")]
                recent_threads = [thread for thread in all_threads if not thread.get("pinned")]
                pinned_suffix = f", {len(pinned_threads)} pinned" if pinned_threads else ""
                st.markdown(
                    f"<div class='sidebar-summary'>{len(all_threads)} saved conversations{pinned_suffix}</div>",
                    unsafe_allow_html=True,
                )
                self._render_thread_section(
                    title="Pinned",
                    threads=pinned_threads,
                    active_thread_id=st.session_state.active_thread_id,
                )
                self._render_thread_section(
                    title="Recent",
                    threads=recent_threads,
                    active_thread_id=st.session_state.active_thread_id,
                )

            active_thread = self.chat_store.get_thread(
                user_id=user_id,
                thread_id=st.session_state.active_thread_id,
            )

            if active_thread:
                st.divider()
                st.markdown("<div class='sidebar-label'>Manage current conversation</div>", unsafe_allow_html=True)

                with st.expander("Context", expanded=True):
                    current_scope = active_thread.get("scope_act") or "All"
                    current_scope_label = next(
                        (label for abbr, label in options if abbr == current_scope),
                        "All Acts",
                    )
                    selected_scope_label = st.selectbox(
                        "Conversation scope",
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

                    pin_label = "Unpin conversation" if active_thread.get("pinned") else "Pin conversation"
                    if st.button(pin_label, key=f"pin_{active_thread['thread_id']}", use_container_width=True):
                        self.chat_store.set_thread_pinned(
                            user_id=user_id,
                            thread_id=active_thread["thread_id"],
                            pinned=not bool(active_thread.get("pinned")),
                        )
                        st.rerun()

                with st.expander("Settings", expanded=False):
                    rename_text = st.text_input(
                        "Rename conversation",
                        value=active_thread["title"],
                        key=f"rename_{active_thread['thread_id']}",
                    )
                    rename_col, delete_col = st.columns(2)
                    with rename_col:
                        if st.button("Save name", key=f"save_{active_thread['thread_id']}", type="primary", use_container_width=True):
                            cleaned = (rename_text or "").strip() or "New Chat"
                            self.chat_store.rename_thread(
                                user_id=user_id,
                                thread_id=active_thread["thread_id"],
                                title=cleaned,
                            )
                            st.rerun()
                    with delete_col:
                        if st.button("Delete", key=f"delete_{active_thread['thread_id']}", use_container_width=True):
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
                            "Export conversation",
                            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                            file_name=f"{file_title}_{active_thread['thread_id'][:8]}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

        active_thread_id = st.session_state.active_thread_id
        active_thread = self.chat_store.get_thread(user_id=user_id, thread_id=active_thread_id)
        act_abbrev = active_thread.get("scope_act") if active_thread else "All"
        messages = self.chat_store.get_messages(active_thread_id)

        st.markdown(
            """
            <div class="app-header">
                <div class="eyebrow">Grounded legal research</div>
                <h1>Indian Legal Advisor</h1>
                <div class="app-subtitle">Ask about sections, articles, procedures, and risks with retrieval-grounded answers and supporting sources.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        self._render_thread_context_bar(active_thread, len(messages))

        if not messages:
            self._render_empty_state()

        for index, message in enumerate(messages):
            role = message["role"]
            label = "You" if role == "user" else "Assistant"
            with st.chat_message(role):
                with st.container(border=True):
                    st.markdown(f"<div class='eyebrow'>{label}</div>", unsafe_allow_html=True)
                    st.markdown(message["content"])
                if message.get("sources"):
                    self._render_sources(message["sources"], key_prefix=f"hist_{index}")

        query = st.chat_input("Ask about an article, section, procedure, or legal risk")
        if not query and st.session_state.prefill_query:
            query = st.session_state.prefill_query
            st.session_state.prefill_query = ""
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
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>You</div>", unsafe_allow_html=True)
                st.markdown(query)

        with st.chat_message("assistant"):
            with st.container(border=True):
                st.markdown("<div class='eyebrow'>Assistant</div>", unsafe_allow_html=True)
                placeholder = st.empty()
                answer_text = ""
                sources = []

                with st.spinner("Searching the most relevant provisions..."):
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
                            placeholder.markdown(answer_text + "▌")
                            if chunk_count == 1:
                                logger.info("UI received first stream token: request_id=%s", request_id)
                        elif event_type == "done":
                            answer_text = event.get("content", answer_text)
                            sources = event.get("sources", [])

                    logger.info(
                        "UI stream completed: request_id=%s chunks=%s answer_chars=%s sources=%s",
                        request_id,
                        chunk_count,
                        len(answer_text),
                        len(sources),
                    )

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

        logger.info(
            "Turn finished: request_id=%s elapsed=%.2fs",
            request_id,
            time.perf_counter() - turn_start,
        )
        st.rerun()
