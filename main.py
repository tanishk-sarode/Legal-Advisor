from common.config import vectorstore, compressor

from core.chain import build_chain
from core.llm import get_answer_llm, get_retriever_llm
from core.retrievers import build_retriever
from core.schema import ExpandedQuery, FinalAnswer

from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.callbacks.file import FileCallbackHandler


from ui.streamlit_app import LegalAdvisorUI


LOG_PATH = "debug_rag_trace.txt"


def build_app():
    # 1. Build base retriever
    base_retriever = build_retriever(
        vectorstore=vectorstore,
        llm=get_retriever_llm(),
    )

    # 2. Wrap with contextual compression (optional but valid)
    # retriever = ContextualCompressionRetriever(
    #     base_retriever=base_retriever,
    #     base_compressor=compressor,
    # )
    retriever = base_retriever

    # 3. LLMs
    answer_llm = get_answer_llm()

    # 4. Output parsers (REQUIRED by build_chain)
    query_parser = PydanticOutputParser(pydantic_object=ExpandedQuery)
    answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

    # 5. Callbacks
    handler = FileCallbackHandler(LOG_PATH)

    # 6. Build chain (signature now matches)
    chain = build_chain(
        answer_llm=answer_llm,
        retriever=retriever,
        answer_parser=answer_parser,
        query_parser=query_parser,
    ).with_config({"callbacks": [handler]})

    # 7. UI
    app = LegalAdvisorUI(chain)
    app.render()


if __name__ == "__main__":
    build_app()
