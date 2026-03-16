from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import streamlit as st

from infra.session_manager import SessionManager
from stores.chat_store import ChatStore


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


def _render_thread_item(thread: dict[str, Any], *, is_active: bool) -> None:
    title = _thread_title_preview(thread.get("title") or "Untitled")
    scope = thread.get("scope_act") or "All"
    updated = _format_relative_time(thread.get("updated_at"))
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
    *,
    title: str,
    threads: list[dict[str, Any]],
    active_thread_id: str | None,
) -> None:
    if not threads:
        return

    st.markdown(f"<div class='sidebar-label'>{title}</div>", unsafe_allow_html=True)
    for thread in threads:
        _render_thread_item(thread, is_active=thread.get("thread_id") == active_thread_id)


def render_sidebar(
    *,
    chat_store: ChatStore,
    session_manager: SessionManager,
    user_id: str,
    options: list[tuple[str, str]],
) -> dict[str, Any] | None:
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
            chat_store.create_thread(user_id=user_id, thread_id=new_thread_id, scope_act="All")
            st.session_state.active_thread_id = new_thread_id
            st.rerun()

        st.markdown("<div class='sidebar-label'>Browse conversations</div>", unsafe_allow_html=True)
        st.text_input(
            "Search conversations",
            key="thread_search",
            placeholder="Search saved chats",
            label_visibility="collapsed",
        )

        all_threads = chat_store.list_threads(
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
            _render_thread_section(
                title="Pinned",
                threads=pinned_threads,
                active_thread_id=st.session_state.active_thread_id,
            )
            _render_thread_section(
                title="Recent",
                threads=recent_threads,
                active_thread_id=st.session_state.active_thread_id,
            )

        active_thread = chat_store.get_thread(
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
                    chat_store.set_thread_scope(
                        user_id=user_id,
                        thread_id=active_thread["thread_id"],
                        scope_act=selected_scope,
                    )
                    st.rerun()

                pin_label = "Unpin conversation" if active_thread.get("pinned") else "Pin conversation"
                if st.button(pin_label, key=f"pin_{active_thread['thread_id']}", use_container_width=True):
                    chat_store.set_thread_pinned(
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
                        chat_store.rename_thread(
                            user_id=user_id,
                            thread_id=active_thread["thread_id"],
                            title=cleaned,
                        )
                        st.rerun()
                with delete_col:
                    if st.button("Delete", key=f"delete_{active_thread['thread_id']}", use_container_width=True):
                        chat_store.delete_thread(
                            user_id=user_id,
                            thread_id=active_thread["thread_id"],
                        )
                        st.session_state.active_thread_id = session_manager.ensure_default_thread(chat_store, user_id)
                        st.rerun()

                export_payload = chat_store.export_thread(
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

        return active_thread
