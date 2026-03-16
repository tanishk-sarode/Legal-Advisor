from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import streamlit as st


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


def _thread_title_preview(title: str, limit: int = 46) -> str:
    compact = " ".join((title or "Untitled").split()).strip() or "Untitled"
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def render_thread_context_bar(thread: dict[str, Any] | None, message_count: int) -> None:
    if not thread:
        return

    title = _thread_title_preview(thread.get("title") or "New Chat", limit=72)
    scope = thread.get("scope_act") or "All"
    updated = _format_relative_time(thread.get("updated_at"))
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


def render_empty_state(queue_prompt: Callable[[str], None]) -> None:
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
            queue_prompt("Explain Article 21 of the Constitution in simple terms.")
    with col2:
        if st.button("Find Section 420 IPC", key="starter_420_ipc", use_container_width=True):
            queue_prompt("What does Section 420 IPC say, and when does it apply?")
    with col3:
        if st.button("Check bail procedure", key="starter_bail", use_container_width=True):
            queue_prompt("What is the general bail procedure in India for a non-bailable offence?")


def render_message_history(
    messages: list[dict[str, Any]],
    render_sources: Callable[[list[Any], str], None],
) -> None:
    for index, message in enumerate(messages):
        role = message["role"]
        label = "You" if role == "user" else "Assistant"
        with st.chat_message(role):
            with st.container(border=True):
                st.markdown(f"<div class='eyebrow'>{label}</div>", unsafe_allow_html=True)
                st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], f"hist_{index}")


def render_user_message(query: str) -> None:
    with st.chat_message("user"):
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>You</div>", unsafe_allow_html=True)
            st.markdown(query)
