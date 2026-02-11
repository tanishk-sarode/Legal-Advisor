from typing import Dict, Optional, TypedDict
from langchain_core.documents import Document


def build_metadata(
    *,
    act: str,
    act_abbrev: str,
    jurisdiction: str,
    source_type: str,
    title: Optional[str] = None,
    chapter: Optional[str] = None,
    chapter_title: Optional[str] = None,
    article_id: Optional[str] = None,
    section_id: Optional[str] = None,
    raw_text: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    citation = None
    if article_id:
        citation = f"Article {article_id}"
    if section_id:
        citation = f"Section {section_id}"
    if citation and act_abbrev:
        citation = f"{citation} ({act_abbrev})"

    return {
        "source": act,
        "act": act,
        "act_abbrev": act_abbrev,
        "jurisdiction": jurisdiction,
        "source_type": source_type,
        "title": title,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "article_id": article_id,
        "section_id": section_id,
        "citation": citation,
        "raw_text": raw_text,
    }


def make_document(text: str, metadata: Dict[str, Optional[str]]) -> Document:
    return Document(page_content=text, metadata=metadata)

from pydantic import BaseModel
from typing import List

class AnswerInput(TypedDict):
    docs: List[Document]
    query: str
    
class ExpandedQuery(BaseModel):
    primary_issue: str
    sub_queries: List[str]


class FinalAnswer(BaseModel):
    answer: str
    cited_sections: List[str]

class GraphState(TypedDict):
    query: str
    expanded_query: ExpandedQuery
    docs: List[Document]
    act: Optional[str]
