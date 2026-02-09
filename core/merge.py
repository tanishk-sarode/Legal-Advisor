from langchain_core.runnables import RunnableLambda

DEFAULT_RETRIEVER_WEIGHTS = {
    "lexical": 1.3,
    "article": 1.2,
    "section": 1.1,
    "semantic": 1.0
}

DEFAULT_RETRIEVER_CAPS = {
    "lexical": 10,
    "article": 12,
    "section": 12,
    "semantic": 18
}


def build_merger(weights=None, caps=None, rrf_k: int = 60):
    weights = weights or DEFAULT_RETRIEVER_WEIGHTS
    caps = caps or DEFAULT_RETRIEVER_CAPS

    def _merge(results: dict):
        seen = set()
        scored = []

        for retriever_name, docs in results.items():
            cap = caps.get(retriever_name)
            weight = weights.get(retriever_name, 1.0)
            for idx, d in enumerate(docs[:cap] if cap else docs):
                raw = d.metadata.get("raw_text") or d.metadata.get("raw_content") or ""
                fingerprint = raw.strip() or d.page_content[:400]
                key = (
                    d.metadata.get("act_abbrev"),
                    d.metadata.get("article_id") or d.metadata.get("section_id"),
                    hash(fingerprint)
                )
                if key in seen:
                    continue
                seen.add(key)

                rank = d.metadata.get("retriever_rank")
                rank = rank if isinstance(rank, int) else idx
                score = weight * (1.0 / (rrf_k + rank + 1))
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]

    return RunnableLambda(_merge)
