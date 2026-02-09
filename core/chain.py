from datetime import datetime
from typing import Iterable, List, Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document


LOG_PATH = "debug_rag_trace.txt"
MAX_DOC_CHARS = 800
MAX_DOCS_LOGGED = 12

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a legal assistant grounded ONLY in the provided context.\n"
        "Rules:\n"
        "- Use only the provided context\n"
        "- Cite exact article/section citations (e.g., Article 21 (COI), Section 279 (IPC))\n"
        "- Some legal terms are composite; if the question is about a commonly used term that is not a named offence, answer by combining the relevant retrieved sections\n"
        "- Do not guess or use external knowledge\n"
        "- Be detailed, clear, and human in tone\n"
        "- Prefer 2-4 short paragraphs plus a compact bullet list of cited sections and penalties\n"
        "- If key details are missing in the context, briefly state what is missing in one short sentence at the end without boilerplate disclaimers"
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{query}")
])


def _append_log(title: str, lines: Iterable[str]) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== {title} | {timestamp} ===\n")
        for line in lines:
            f.write(f"{line}\n")


ChainState = Dict[str, Any]


def _ensure_input(x: Any) -> ChainState:
    if isinstance(x, dict):
        return x
    return {"query": str(x)}


def _log_query(x: ChainState) -> ChainState:
    _append_log("query", [x.get("query", "")])
    return x


def _log_retrieved(x: ChainState) -> ChainState:
    docs = x.get("retrieved", []) or []
    lines = []
    for d in docs[:MAX_DOCS_LOGGED]:
        citation = d.metadata.get("citation")
        retriever = d.metadata.get("retriever")
        snippet = (d.page_content or "")[:MAX_DOC_CHARS].replace("\n", " ")
        lines.append(f"- citation={citation} retriever={retriever} text={snippet}")
    _append_log("retrieved_docs", lines or ["<empty>"])
    return x


def _log_final(x):
    _append_log("final_result", [str(x)])
    return x


def _to_retrieval_query(x: ChainState) -> str:
    query = x.get("query", "") or ""
    act = x.get("act") or "All"
    if act and act != "All":
        return f"{query}\nAct scope: {act}".strip()
    return query.strip()


def _to_retrieval_input(x: ChainState) -> Dict[str, Any]:
    return {"act": x.get("act"), "query": x.get("query", "")}


def build_chain(retriever, answer_llm):
    def build_context(docs):
        return "\n\n".join(
            f"[{d.metadata.get('citation')}]\n{d.page_content}"
            for d in docs
        )

    retrieval_chain = (
        RunnableLambda(_to_retrieval_input)
        | RunnableLambda(_to_retrieval_query)
        | retriever
    )

    return (
        RunnableLambda(_ensure_input)
        | RunnableLambda(_log_query)
        | RunnablePassthrough.assign(retrieved=retrieval_chain)
        | RunnableLambda(_log_retrieved)
        | RunnablePassthrough.assign(docs=lambda x: x.get("retrieved", []))
        | RunnablePassthrough.assign(context=lambda x: build_context(x["docs"]))
        | ANSWER_PROMPT
        | answer_llm
        | RunnableLambda(_log_final)
    )
