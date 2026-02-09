from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
META_PREFIX = "metadata."


def _term_field(field: str) -> str:
    return f"{META_PREFIX}{field}"


def _keyword_field(field: str) -> str:
    return f"{META_PREFIX}{field}.keyword"


def _term_or_keyword(field: str, value: str) -> Dict:
    return {
        "bool": {
            "should": [
                {"term": {_term_field(field): value}},
                {"term": {_keyword_field(field): value}},
            ],
            "minimum_should_match": 1
        }
    }


def _act_filter(act_abbrev: str) -> Dict:
    if not act_abbrev or act_abbrev == "All":
        return {}
    return _term_or_keyword("act_abbrev", act_abbrev)


def _build_filter(base_terms: List[Dict]) -> Dict:
    return {"bool": {"filter": base_terms}}


def _source_type_filter(values: List[str]) -> Dict:
    should = [_term_or_keyword("source_type", value) for value in values]
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _lexical_query(query: str, filters: List[Dict], size: int) -> Dict:
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": query,
                            "fields": [
                                "text",
                                "metadata.title",
                                "metadata.chapter_title",
                                "metadata.citation",
                                "metadata.raw_text",
                            ],
                            "default_operator": "AND"
                        }
                    }
                ],
                "filter": filters
            }
        }
    }


def _hits_to_docs(hits: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for hit in hits:
        source = hit.get("_source", {})
        docs.append(
            Document(
                page_content=source.get("text", ""),
                metadata=source.get("metadata", {})
            )
        )
    return docs


def build_retrievers(vectorstore, semantic_retriever):
    def _lexical_retriever(x: Dict) -> List[Document]:
        intents = x.get("intents", {})
        act_abbrev = intents.get("act", "All")
        docs: List[Document] = []

        for item in intents.get("article_lookups", []):
            key = item.get("id")
            act = item.get("act") or act_abbrev
            if not key:
                continue
            filters = [_source_type_filter(["article", "clause"])]
            if act and act != "All":
                filters.append(_term_or_keyword("act_abbrev", act))

            body = _lexical_query(f"\"Article {key}\"", filters, size=8)
            hits = vectorstore.client.search(index=vectorstore.index_name, body=body).get("hits", {}).get("hits", [])
            docs.extend(_hits_to_docs(hits))

        for item in intents.get("section_lookups", []):
            key = item.get("id")
            act = item.get("act") or act_abbrev
            if not key:
                continue
            filters = [_source_type_filter(["section", "clause"])]
            if act and act != "All":
                filters.append(_term_or_keyword("act_abbrev", act))

            body = _lexical_query(f"\"Section {key}\"", filters, size=8)
            hits = vectorstore.client.search(index=vectorstore.index_name, body=body).get("hits", {}).get("hits", [])
            docs.extend(_hits_to_docs(hits))

        trimmed = docs[:10]
        for idx, d in enumerate(trimmed):
            d.metadata["retriever"] = "lexical"
            d.metadata["retriever_rank"] = idx
        return trimmed

    def _article_retriever(x: Dict) -> List[Document]:
        intents = x.get("intents", {})
        act_abbrev = intents.get("act", "All")
        lookups = intents.get("article_lookups", [])
        docs: List[Document] = []

        base_filters = [_term_or_keyword("source_type", "article")]
        act_term = _act_filter(act_abbrev)
        if act_term:
            base_filters.append(act_term)

        if lookups:
            for item in lookups:
                key = item.get("id")
                act = item.get("act") or act_abbrev
                if not key:
                    continue
                filters = base_filters + [_term_or_keyword("article_id", key)]
                if act and act != "All":
                    filters.append(_term_or_keyword("act_abbrev", act))

                docs.extend(
                    vectorstore.similarity_search(
                        f"Article {key}",
                        k=6,
                        filter=_build_filter(filters)
                    )
                )
        else:
            for q in intents.get("semantic_queries", []):
                docs.extend(vectorstore.similarity_search(q, k=8, filter=_build_filter(base_filters)))

        trimmed = docs[:10]
        for idx, d in enumerate(trimmed):
            d.metadata["retriever"] = "article"
            d.metadata["retriever_rank"] = idx
        return trimmed

    def _section_retriever(x: Dict) -> List[Document]:
        intents = x.get("intents", {})
        act_abbrev = intents.get("act", "All")
        lookups = intents.get("section_lookups", [])
        docs: List[Document] = []

        base_filters = [_term_or_keyword("source_type", "section")]
        act_term = _act_filter(act_abbrev)
        if act_term:
            base_filters.append(act_term)

        if lookups:
            for item in lookups:
                key = item.get("id")
                act = item.get("act") or act_abbrev
                if not key:
                    continue
                filters = base_filters + [_term_or_keyword("section_id", key)]
                if act and act != "All":
                    filters.append(_term_or_keyword("act_abbrev", act))

                docs.extend(
                    vectorstore.similarity_search(
                        f"Section {key}",
                        k=5,
                        filter=_build_filter(filters)
                    )
                )
        else:
            for q in intents.get("semantic_queries", []):
                docs.extend(vectorstore.similarity_search(q, k=8, filter=_build_filter(base_filters)))

        trimmed = docs[:10]
        for idx, d in enumerate(trimmed):
            d.metadata["retriever"] = "section"
            d.metadata["retriever_rank"] = idx
        return trimmed

    def _semantic_retriever(x: Dict) -> List[Document]:
        intents = x.get("intents", {})
        act_abbrev = intents.get("act", "All")
        docs: List[Document] = []

        for q in intents.get("semantic_queries", []):
            retrieved = semantic_retriever.invoke(q)
            docs.extend(retrieved)

        if act_abbrev and act_abbrev != "All":
            docs = [d for d in docs if d.metadata.get("act_abbrev") == act_abbrev]

        trimmed = docs[:6]
        for idx, d in enumerate(trimmed):
            d.metadata["retriever"] = "semantic"
            d.metadata["retriever_rank"] = idx
        return trimmed

    return {
        "lexical": RunnableLambda(_lexical_retriever),
        "article": RunnableLambda(_article_retriever),
        "section": RunnableLambda(_section_retriever),
        "semantic": RunnableLambda(_semantic_retriever)
    }
