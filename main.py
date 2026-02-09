from pathlib import Path

from common.config import INGEST, vectorstore, compressor
from core.chain import build_chain
from core.indexer import build_all_documents, ingest_all
from core.llm import get_answer_llm, get_retriever_llm
from core.retrievers import build_retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from ui.streamlit_app import LegalAdvisorUI

def maybe_ingest():
    if not INGEST:
        return
    root = Path(__file__).resolve().parent
    docs = build_all_documents(root)
    ingest_all(vectorstore, docs)


def build_app():
    base_retriever = build_retriever(vectorstore, get_retriever_llm())
    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
    answer_llm = get_answer_llm()

    chain = build_chain(
        retriever=retriever,
        answer_llm=answer_llm
    )

    app = LegalAdvisorUI(chain)
    app.render()


if __name__ == "__main__":
    maybe_ingest()
    build_app()
