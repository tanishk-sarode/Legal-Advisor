import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.acts import get_act_sources, get_constitution_source
from core.schema import build_metadata


class Indexer:
    def __init__(self, chunk_size: int = 900, overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". "]
        )

    def build_all_documents(self, root: Path) -> List[Document]:
        docs: List[Document] = []

        constitution = get_constitution_source(root)
        if constitution.file_path.exists():
            docs.extend(self._load_constitution_documents(constitution.file_path))

        for act in get_act_sources(root):
            if act.file_path.exists():
                docs.extend(self._load_act_documents(act.act, act.act_abbrev, act.file_path))

        return docs

    def ingest_all(self, vectorstore, docs: List[Document], batch_size: int = 128) -> int:
        total = 0
        max_docs_len = 0
        for doc in docs:
            doc_len = len(doc.page_content)
            if doc_len > max_docs_len:
                max_docs_len = doc_len
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            vectorstore.add_documents(batch)
            total += len(batch)
        print(f"Ingested {total} documents. Max document length: {max_docs_len} characters.")
        return total

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        split_docs: List[Document] = []
        for doc in docs:
            if len(doc.page_content) <= 1200:
                split_docs.append(doc)
                continue
            chunks = self._splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks):
                chunk.metadata = dict(chunk.metadata)
                chunk.metadata["source_type"] = "clause"
                chunk.metadata["chunk_index"] = str(idx)
            split_docs.extend(chunks)
        return split_docs

    def _load_act_documents(self, act: str, act_abbrev: str, json_path: Path) -> List[Document]:
        def _metadata_func(record: dict, metadata: dict) -> dict:
            section = record.get("section")
            if section is None:
                section = record.get("Section")
            section_id = str(section).strip() if section is not None else None
            title = (record.get("section_title") or "").strip()
            chapter = record.get("chapter")
            chapter_title = record.get("chapter_title")
            description = (record.get("section_desc") or "").strip()

            return build_metadata(
                act=act,
                act_abbrev=act_abbrev,
                jurisdiction="India",
                source_type="section",
                title=title or None,
                chapter=str(chapter) if chapter is not None else None,
                chapter_title=(str(chapter_title).strip() if chapter_title else None),
                section_id=section_id,
                raw_text=description or None,
            )

        loader = JSONLoader(
            file_path=str(json_path),
            jq_schema=(
                ".[] | {"
                "section_desc: (.section_desc // .Section_desc // \"\"),"
                "section: (.section // .Section),"
                "section_title: (.section_title // \"\"),"
                "chapter: .chapter,"
                "chapter_title: .chapter_title"
                "}"
            ),
            content_key="section_desc",
            metadata_func=_metadata_func,
        )

        docs = []
        for doc in loader.load():
            description = (doc.page_content or "").strip()
            if not description:
                continue
            title = (doc.metadata.get("title") or "").strip()
            section_id = doc.metadata.get("section_id")
            normalized = self._normalize_text(description)
            heading = f"Section {section_id}. {title}" if section_id else title
            doc.page_content = f"{heading}\n{normalized}" if heading else normalized
            docs.append(doc)

        return self._split_documents(docs)

    def _load_constitution_documents(self, json_path: Path) -> List[Document]:
        def _metadata_func(record: dict, metadata: dict) -> dict:
            article = str(record.get("article", "")).strip()
            title = (record.get("title") or "").strip()
            description = (record.get("description") or "").strip()

            return build_metadata(
                act="Constitution of India",
                act_abbrev="COI",
                jurisdiction="India",
                source_type="article",
                title=title or None,
                article_id=article or None,
                raw_text=description or None,
            )

        loader = JSONLoader(
            file_path=str(json_path),
            jq_schema=".[]",
            content_key="description",
            metadata_func=_metadata_func,
        )

        docs = []
        for doc in loader.load():
            description = (doc.page_content or "").strip()
            if not description:
                continue
            title = (doc.metadata.get("title") or "").strip()
            article_id = doc.metadata.get("article_id")
            normalized = self._normalize_text(description)
            heading = f"Article {article_id}. {title}" if article_id else title
            doc.page_content = f"{heading}\n{normalized}" if heading else normalized
            docs.append(doc)

        return self._split_documents(docs)


_DEFAULT_INDEXER = Indexer()


def build_all_documents(root: Path) -> List[Document]:
    return _DEFAULT_INDEXER.build_all_documents(root)


def ingest_all(vectorstore, docs: List[Document], batch_size: int = 128) -> int:
    return _DEFAULT_INDEXER.ingest_all(vectorstore, docs, batch_size=batch_size)
