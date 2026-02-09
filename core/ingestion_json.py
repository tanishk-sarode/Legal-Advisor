import json
from pathlib import Path
from typing import List

from core.schema import build_metadata, make_document
from core.text_utils import chunk_text, normalize_text


def _read_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ingest_act_json(act: str, act_abbrev: str, json_path: Path) -> List:
    data = _read_json(json_path)
    docs = []

    for item in data:
        section = item.get("section")
        if section is None:
            section = item.get("Section")
        section_id = str(section).strip() if section is not None else None
        title = (item.get("section_title") or "").strip()
        description = (item.get("section_desc") or "").strip()
        if not description:
            continue

        chapter = item.get("chapter")
        chapter_title = item.get("chapter_title")

        normalized = normalize_text(description)
        heading = f"Section {section_id}. {title}" if section_id else title
        full_text = f"{heading}\n{normalized}" if heading else normalized

        metadata = build_metadata(
            act=act,
            act_abbrev=act_abbrev,
            jurisdiction="India",
            source_type="section",
            title=title or None,
            chapter=str(chapter) if chapter is not None else None,
            chapter_title=(str(chapter_title).strip() if chapter_title else None),
            section_id=section_id,
            raw_text=description
        )
        docs.append(make_document(full_text, metadata))

        if len(full_text) > 1200:
            for idx, chunk in enumerate(chunk_text(normalized)):
                chunk_text_value = f"{heading}\n{chunk}" if heading else chunk
                chunk_meta = dict(metadata)
                chunk_meta["source_type"] = "clause"
                chunk_meta["chunk_index"] = str(idx)
                docs.append(make_document(chunk_text_value, chunk_meta))

    return docs
