from common.config import vectorstore
from core.chain import build_chain
from core.llm import get_answer_llm
from core.schema import ExpandedQuery, FinalAnswer

from langchain_core.output_parsers import PydanticOutputParser


from ui.streamlit_app import LegalAdvisorUI


def build_app():
    retriever = vectorstore

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

    """
    Order of operations in chain:
    1. User query -> Expanded query (with legal search queries)
    2. Expanded query -> Search retriever -> Retrieved docs
    3. Retrieved docs + search queries + user query -> Final answer
    """

    # 6. UI
    app = LegalAdvisorUI(chain)
    app.render()


if __name__ == "__main__":
    build_app()
