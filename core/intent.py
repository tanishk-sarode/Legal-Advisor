import re
from typing import Dict, Iterable, List, Optional

ARTICLE_REF_REGEX = re.compile(r"\b(?:Article|Art\.?|Art)\s+(\d+[A-Z]?)\b", re.I)
SECTION_REF_REGEX = re.compile(r"\b(?:Section|Sec\.?|Sec)\s+(\d+[A-Z]?)\b", re.I)

ACT_HINTS = {
    "IPC": re.compile(r"\b(indian penal code|ipc)\b", re.I),
    "CrPC": re.compile(r"\b(code of criminal procedure|crpc)\b", re.I),
    "CPC": re.compile(r"\b(civil procedure code|cpc)\b", re.I),
    "MVA": re.compile(r"\b(motor vehicles act|mva)\b", re.I),
    "IEA": re.compile(r"\b(indian evidence act|iea)\b", re.I),
    "HMA": re.compile(r"\b(hindu marriage act|hma)\b", re.I),
    "IDA": re.compile(r"\b(indian divorce act|ida)\b", re.I),
    "NIA": re.compile(r"\b(negotiable instruments act|nia)\b", re.I),
}

ACCIDENT_HINT = re.compile(r"\b(hit and run|hit-and-run|accident|rash driving|vehicle|motor vehicle)\b", re.I)

ACCIDENT_SECTION_HINTS = {
    "IPC": ["279", "304A"],
    "MVA": ["134", "187"],
}


def _normalize_sub_queries(sub_queries: Iterable[str]) -> List[str]:
    normalized = []
    seen = set()
    for q in sub_queries:
        if not q:
            continue
        q_clean = re.sub(r"\s+", " ", q).strip()
        if not q_clean or q_clean in seen:
            continue
        normalized.append(q_clean)
        seen.add(q_clean)
    return normalized


def _infer_act_from_text(text: str) -> Optional[str]:
    for act, pattern in ACT_HINTS.items():
        if pattern.search(text or ""):
            return act
    return None


def _infer_act(sub_queries: List[str], query: str, explicit_act: Optional[str]) -> Optional[str]:
    if explicit_act and explicit_act != "All":
        return explicit_act
    for q in sub_queries:
        act = _infer_act_from_text(q)
        if act:
            return act
    return _infer_act_from_text(query)


def _is_accident_query(sub_queries: List[str], query: str) -> bool:
    if ACCIDENT_HINT.search(query or ""):
        return True
    return any(ACCIDENT_HINT.search(q or "") for q in sub_queries)


def _dedupe_lookups(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("act"), item.get("id"), item.get("type"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_retrieval_intents(sub_queries: List[str], query: str, explicit_act: Optional[str]) -> Dict:
    normalized = _normalize_sub_queries(sub_queries)
    act = _infer_act(normalized, query, explicit_act)

    article_lookups: List[Dict[str, str]] = []
    section_lookups: List[Dict[str, str]] = []
    semantic_queries: List[str] = []

    for q in normalized:
        q_act = _infer_act_from_text(q) or act

        for match in ARTICLE_REF_REGEX.findall(q):
            article_lookups.append({"type": "article", "act": q_act or "All", "id": match.upper()})

        for match in SECTION_REF_REGEX.findall(q):
            section_lookups.append({"type": "section", "act": q_act or "All", "id": match.upper()})

        semantic_queries.append(q)

    if _is_accident_query(normalized, query):
        for act_hint, sections in ACCIDENT_SECTION_HINTS.items():
            for section_id in sections:
                section_lookups.append({"type": "section", "act": act_hint, "id": section_id})

    return {
        "act": act or explicit_act or "All",
        "article_lookups": _dedupe_lookups(article_lookups),
        "section_lookups": _dedupe_lookups(section_lookups),
        "semantic_queries": _normalize_sub_queries(semantic_queries),
    }
