from typing import List, Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document


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


ChainState = Dict[str, Any]


def _ensure_input(x: Any) -> ChainState:
    if isinstance(x, dict):
        return x
    return {"query": str(x)}


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
        | RunnablePassthrough.assign(retrieved=retrieval_chain)
        | RunnablePassthrough.assign(docs=lambda x: x.get("retrieved", []))
        | RunnablePassthrough.assign(context=lambda x: build_context(x["docs"]))
        | ANSWER_PROMPT
        | answer_llm
    )
