from pathlib import Path

from common.config import vectorstore
from core.chain import build_chain
from core.llm import get_answer_llm
from core.schema import ExpandedQuery, FinalAnswer
from stores.factory import build_chat_store

from langchain_core.output_parsers import PydanticOutputParser


from ui.streamlit_app import LegalAdvisorUI


def build_app():
    root = Path(__file__).resolve().parent
    retriever = vectorstore
    chat_store = build_chat_store(root)

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
    app = LegalAdvisorUI(chain, chat_store=chat_store)
    app.render()


if __name__ == "__main__":
    build_app()
