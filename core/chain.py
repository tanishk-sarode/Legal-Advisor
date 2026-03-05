from typing import Dict, Iterator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import OpenSearchVectorSearch

from core.prompts import QUERY_GENERATOR_PROMPT, ANSWER_PROMPT, ANSWER_STREAM_PROMPT
from core.schema import ExpandedQuery, FinalAnswer, GraphState


# ---------- Utilities ----------

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('citation')}]\n{d.page_content}"
        for d in docs
    )


def build_answer_input(state: GraphState) -> Dict[str, str]:
    # Format the search queries for context
    search_queries = ""
    if state.get("expanded_query") and state["expanded_query"].sub_queries:
        search_queries = "\n---\n\nSearch queries used to find relevant sections:\n" 
        for i, sq in enumerate(state["expanded_query"].sub_queries, 1): 
            search_queries += f"{i}. {sq}\n"
    
    context = format_docs(state["docs"])
    if search_queries:
        context = search_queries + "\n" + context
    
    return {
        "context": context,
        "query": state["query"],
    }


def stringify_history(chat_history: str | None) -> str:
    if chat_history and chat_history.strip():
        return chat_history
    return "No prior conversation."

def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (
            d.metadata.get("citation"),
            d.metadata.get("source"),
            d.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


class RetrievalLegalChain:
    def __init__(
        self,
        answer_llm,
        vectorstore: OpenSearchVectorSearch,
        answer_parser: PydanticOutputParser,
        query_parser: PydanticOutputParser,
        *,
        similarity_k: int = 12,
    ):
        self.answer_llm = answer_llm
        self.vectorstore = vectorstore
        self.answer_parser = answer_parser
        self.query_parser = query_parser
        self.similarity_k = similarity_k

        self.query_generator_chain = (
            QUERY_GENERATOR_PROMPT
            | self.answer_llm
            | self.query_parser
        )
        self.answer_chain = (
            RunnableLambda(build_answer_input)
            | ANSWER_PROMPT.partial(
                format_instructions=self.answer_parser.get_format_instructions()
            )
            | self.answer_llm
            | self.answer_parser
        )
        self.answer_stream_chain = ANSWER_STREAM_PROMPT | self.answer_llm

    def _retrieve(self, *, query: str, act: str | None, chat_history: str | None):
        expanded: ExpandedQuery = self.query_generator_chain.invoke(
            {
                "query": query,
                "chat_history": stringify_history(chat_history),
            }
        )

        queries = expanded.sub_queries if expanded.sub_queries else [query]

        docs: List[Document] = []
        for sub_query in queries:
            docs.extend(self.vectorstore.similarity_search(sub_query, k=self.similarity_k))

        docs = dedupe_docs(docs)
        if act and act != "All":
            filtered = [doc for doc in docs if doc.metadata.get("act_abbrev") == act]
            if filtered:
                docs = filtered

        return expanded, docs

    def invoke(self, inputs):
        query = inputs["query"]
        act = inputs.get("act")
        chat_history = inputs.get("chat_history")

        expanded_query, docs = self._retrieve(
            query=query,
            act=act,
            chat_history=chat_history,
        )

        state: GraphState = {
            "query": query,
            "chat_history": chat_history,
            "act": act,
            "expanded_query": expanded_query,
            "docs": docs,
            "answer": None,
        }

        answer: FinalAnswer = self.answer_chain.invoke(state)
        return {"answer": answer, "sources": docs}

    @staticmethod
    def _chunk_to_text(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            output = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    output.append(part["text"])
                elif hasattr(part, "text"):
                    output.append(getattr(part, "text"))
            return "".join(output)
        return ""

    def stream(self, inputs) -> Iterator[Dict[str, object]]:
        query = inputs["query"]
        act = inputs.get("act")
        chat_history = inputs.get("chat_history")

        _, docs = self._retrieve(
            query=query,
            act=act,
            chat_history=chat_history,
        )

        context = format_docs(docs)
        stream_inputs = {
            "chat_history": stringify_history(chat_history),
            "context": context,
            "query": query,
        }

        full_text = ""
        for chunk in self.answer_stream_chain.stream(stream_inputs):
            token_text = self._chunk_to_text(chunk)
            if not token_text:
                continue
            full_text += token_text
            yield {"type": "token", "content": token_text}

        yield {
            "type": "done",
            "content": full_text.strip(),
            "sources": docs,
        }


# ---------- Chain Builder ----------

def build_chain(
    answer_llm,
    vectorstore: OpenSearchVectorSearch,
    answer_parser: PydanticOutputParser,
    query_parser: PydanticOutputParser,
    *,
    similarity_k: int = 12,
):
    return RetrievalLegalChain(
        answer_llm=answer_llm,
        vectorstore=vectorstore,
        answer_parser=answer_parser,
        query_parser=query_parser,
        similarity_k=similarity_k,
    )
