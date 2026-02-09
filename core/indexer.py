from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.acts import get_act_sources, get_constitution_source
from core.ingestion_json import ingest_act_json
from core.ingestion_constitution import ingest_constitution


def build_all_documents(root: Path) -> List[Document]:
    docs: List[Document] = []

    constitution = get_constitution_source(root)
    if constitution.file_path.exists():
        docs.extend(ingest_constitution(constitution.file_path))

    for act in get_act_sources(root):
        if act.file_path.exists():
            docs.extend(ingest_act_json(act.act, act.act_abbrev, act.file_path))

    return docs


def ingest_all(vectorstore, docs: List[Document], batch_size: int = 128) -> int:
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        total += len(batch)
    return total
