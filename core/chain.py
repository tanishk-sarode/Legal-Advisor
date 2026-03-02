import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END, START

from core.prompts import QUERY_GENERATOR_PROMPT, ANSWER_PROMPT
from core.schema import ExpandedQuery, FinalAnswer, AnswerInput, GraphState


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


# ---------- Chain Builder ----------

def build_chain(
    answer_llm,
    vectorstore,
    answer_parser: PydanticOutputParser,
    query_parser: PydanticOutputParser,
    *,
    similarity_k: int = 12,
):


    query_generator_chain = (
        QUERY_GENERATOR_PROMPT
        | answer_llm
        | query_parser
    )

    def expand_query(state: GraphState):
        expanded: ExpandedQuery = query_generator_chain.invoke(
            {"query": state["query"]}
        )
        return {"expanded_query": expanded}

    def retrieve_docs(state: GraphState):
        expanded: ExpandedQuery = state["expanded_query"]

        queries = (
            expanded.sub_queries
            if expanded.sub_queries
            else [state["query"]]
        )

        docs: List[Document] = []

        # simple multi-query retrieval (can add reranker later)
        for q in queries:
            docs.extend(
                vectorstore.similarity_search(q, k=similarity_k)
            )

        return {"docs": docs}


    def deduplicate(state: GraphState):
        return {"docs": dedupe_docs(state["docs"])}

    answer_chain = (
        RunnableLambda(build_answer_input)
        | ANSWER_PROMPT.partial(
            format_instructions=answer_parser.get_format_instructions()
        )
        | answer_llm
        | answer_parser
    )

    def generate_answer(state: GraphState):
        answer: FinalAnswer = answer_chain.invoke(state)
        return {"answer": answer}

    def format_output(state: GraphState):
        """Format the final output with answer and sources"""
        return {
            "answer": state["answer"],
            "sources": state["docs"]
        }

    builder = StateGraph(GraphState)

    builder.add_edge(START, "expand_query")
    builder.add_node("expand_query", expand_query)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("deduplicate", deduplicate)
    builder.add_node("generate_answer", generate_answer)


    builder.add_edge("expand_query", "retrieve_docs")
    builder.add_edge("retrieve_docs", "deduplicate")
    builder.add_edge("deduplicate", "generate_answer")
    builder.add_edge("generate_answer", END)

    graph = builder.compile()
    
    # Wrap the graph to return formatted output
    def chain_with_sources(inputs):
        result = graph.invoke(inputs)
        return {
            "answer": result["answer"],
            "sources": result["docs"]
        }
    
    return RunnableLambda(chain_with_sources)
