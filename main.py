from common.config import vectorstore, compressor

from core.chain import build_chain
from core.llm import get_answer_llm, get_retriever_llm
from core.retrievers import build_retrievers
from core.schema import ExpandedQuery, FinalAnswer

from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever


from ui.streamlit_app import LegalAdvisorUI


LOG_PATH = "debug_rag_trace.txt"


def build_app():
    # 1. Build retrievers
    vectorstore_ref, self_query_retriever = build_retrievers(
        vectorstore=vectorstore,
        llm=get_retriever_llm(),
    )


    retriever = vectorstore_ref

    # 3. LLMs
    answer_llm = get_answer_llm()

    # 4. Output parsers (REQUIRED by build_chain)
    query_parser = PydanticOutputParser(pydantic_object=ExpandedQuery)
    answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

    # 5. Build chain (custom human-readable logging only)
    chain = build_chain(
        answer_llm=answer_llm,
        vectorstore=retriever,
        answer_parser=answer_parser,
        query_parser=query_parser,
    )

    # 6. UI
    app = LegalAdvisorUI(chain)
    app.render()


if __name__ == "__main__":
    build_app()
