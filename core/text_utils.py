import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


CLAUSE_SPLIT_REGEX = re.compile(
    r"\n\s*(?:\([0-9]+\)|\([a-zA-Z]\)|[0-9]+\.|[a-zA-Z]\.)\s+"
)


def normalize_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_clauses(text: str) -> List[str]:
    if not text:
        return []
    parts = CLAUSE_SPLIT_REGEX.split(text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". "]
    )
    return splitter.split_text(text)
