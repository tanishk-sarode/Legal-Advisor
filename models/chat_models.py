from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Thread:
    thread_id: str
    user_id: str
    title: str
    scope_act: str
    pinned: bool
    created_at: str
    updated_at: str


@dataclass
class Message:
    thread_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]]
    created_at: str


@dataclass
class ThreadMemory:
    thread_id: str
    summary: str
    updated_at: str