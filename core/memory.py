from __future__ import annotations

from typing import Any


def _normalize_text(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "…"


def _to_line(message: dict[str, Any]) -> str:
    role = message.get("role", "user")
    content = _normalize_text(message.get("content", ""), 260)
    return f"{role.capitalize()}: {content}"


def build_running_summary(
    existing_summary: str,
    archived_messages: list[dict[str, Any]],
    *,
    max_chars: int = 2500,
) -> str:
    if not archived_messages:
        return existing_summary.strip()

    recent_archived = archived_messages[-8:]
    archived_lines = [_to_line(message) for message in recent_archived]

    sections = []
    if existing_summary.strip():
        sections.append(existing_summary.strip())
    sections.append("Recent archived conversation:\n" + "\n".join(f"- {line}" for line in archived_lines))

    merged = "\n\n".join(sections)
    return _normalize_text(merged, max_chars)


def compose_memory_context(
    summary: str,
    messages: list[dict[str, Any]],
    *,
    recent_messages: int = 8,
) -> str:
    tail = messages[-recent_messages:] if recent_messages > 0 else []
    tail_lines = [_to_line(message) for message in tail]

    parts = []
    if summary.strip():
        parts.append(f"Conversation summary:\n{summary.strip()}")
    if tail_lines:
        parts.append("Recent messages:\n" + "\n".join(tail_lines))

    if not parts:
        return "No prior conversation."
    return "\n\n".join(parts)
