import json
from pathlib import Path
from typing import List

from core.schema import build_metadata, make_document
from core.text_utils import chunk_text, normalize_text


def ingest_constitution(json_path: Path) -> List:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        article = str(item.get("article", "")).strip()
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        if not description:
            continue

        normalized = normalize_text(description)
        article_id = article if article else None
        full_text = f"Article {article}. {title}\n{normalized}" if article else f"{title}\n{normalized}"

        metadata = build_metadata(
            act="Constitution of India",
            act_abbrev="COI",
            jurisdiction="India",
            source_type="article",
            title=title or None,
            article_id=article_id,
            raw_text=description
        )
        docs.append(make_document(full_text, metadata))

        if len(full_text) > 1200:
            for idx, chunk in enumerate(chunk_text(normalized)):
                chunk_text_value = f"Article {article}. {title}\n{chunk}" if article else f"{title}\n{chunk}"
                chunk_meta = dict(metadata)
                chunk_meta["source_type"] = "clause"
                chunk_meta["chunk_index"] = str(idx)
                docs.append(make_document(chunk_text_value, chunk_meta))

    return docs
