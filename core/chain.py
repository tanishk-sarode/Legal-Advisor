from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.documents import Document

from core.prompts import QUERY_GENERATOR_PROMPT, ANSWER_PROMPT
from core.schema import ExpandedQuery, FinalAnswer, AnswerInput


ChainState = Dict[str, Any]


# ---------- Document Formatting (Runnable-native) ----------

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('citation')}]\n{d.page_content}"
        for d in docs
    )


# ---------- Prompt Input Builder (Typed) ----------

def build_answer_input(x: AnswerInput) -> Dict[str, str]:
    return {
        "context": format_docs(x["docs"]),
        "query": x["query"],
    }


# ---------- Chain Builder ----------

def build_chain(
    answer_llm,
    retriever,
    answer_parser: PydanticOutputParser,
    query_parser: PydanticOutputParser,
) -> Runnable:
    """
    Builds a LangChain RAG pipeline with:
    - LLM-based query expansion
    - Multi-query vector retrieval
    - Runnable-native context injection
    - Structured output parsing
    """

    # 1. Answer generation chain (NO deprecated APIs)
    answer_chain = (
        RunnableLambda(build_answer_input)
        | ANSWER_PROMPT.partial(
            format_instructions=answer_parser.get_format_instructions()
        )
        | answer_llm
        | answer_parser
    )

    # 2. Query generator chain
    query_generator_chain = (
        QUERY_GENERATOR_PROMPT
        | answer_llm
        # print llm response for debugging
        # | RunnableLambda(lambda x: print(f"Query generatorLLM response: {x}") or x)
        | query_parser
    )

    # 3. Expanded-query retrieval
    def retrieve_with_expansion(x: ExpandedQuery) -> List[Document]:
        queries = [x.primary_query, *x.similar_queries]
        docs: List[Document] = []
        for q in queries:
            docs.extend(retriever.invoke(q))
        return docs

    retrieval_chain = (
        query_generator_chain
        | RunnableLambda(retrieve_with_expansion)
        # | RunnableLambda(lambda docs: print(f"Retrieved documents \n \n {docs}") or docs)
    )

    # 4. Full pipeline
    chain = (
        RunnablePassthrough.assign(query=lambda x: x["query"])
        | RunnableParallel(
            docs=retrieval_chain,
            query=lambda x: x["query"],
        )
        | RunnableParallel(
            answer=answer_chain,
            sources=lambda x: x["docs"],
        )
    )

    return chain
