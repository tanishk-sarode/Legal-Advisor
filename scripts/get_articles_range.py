from __future__ import annotations

from typing import Dict, List, Tuple

from common.config import vectorstore


def _article_ids(start: int, end: int) -> List[str]:
    return [str(i) for i in range(start, end + 1)]


def _build_query(
    *,
    act_field: str,
    article_field: str,
    act_value: str,
    article_ids: List[str],
) -> Dict:
    return {
        "size": 5000,
        "query": {
            "bool": {
                "filter": [
                    {"term": {act_field: act_value}},
                    {"terms": {article_field: article_ids}},
                ]
            }
        },
    }


def main() -> None:
    start = 1
    end = 40
    article_ids = _article_ids(start, end)
    article_ids_lower = [value.lower() for value in article_ids]

    index_name = vectorstore.index_name
    queries: List[Tuple[str, Dict]] = [
        (
            "keyword_fields",
            _build_query(
                act_field="metadata.act_abbrev.keyword",
                article_field="metadata.article_id.keyword",
                act_value="COI",
                article_ids=article_ids,
            ),
        ),
        (
            "text_fields",
            _build_query(
                act_field="metadata.act_abbrev",
                article_field="metadata.article_id",
                act_value="coi",
                article_ids=article_ids_lower,
            ),
        ),
    ]

    hits: List[Dict] = []
    strategy = ""
    for label, query in queries:
        response = vectorstore.client.search(index=index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            strategy = label
            break

    print(f"Found {len(hits)} hits for COI articles {start}-{end} using {strategy or 'no match'}.")

    for hit in hits:
        metadata = hit.get("_source", {}).get("metadata", {})
        citation = metadata.get("citation")
        article_id = metadata.get("article_id")
        title = metadata.get("title")
        print(f"{article_id}: {citation} - {title}")


if __name__ == "__main__":
    main()
