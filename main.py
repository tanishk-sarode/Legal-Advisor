from pathlib import Path

from common.config import INGEST, vectorstore, semantic_retriever, compressor
from core.chain import build_chain
from core.compression import build_compressor
from core.decomposer import build_decomposer
from core.indexer import build_all_documents, ingest_all
from core.llm import get_answer_llm, get_decomposer_llm
from core.merge import build_merger
from core.retrievers import build_retrievers
from ui.streamlit_app import LegalAdvisorUI
from pydantic.v1.fields import FieldInfo as FieldInfoV1

def maybe_ingest():
    if not INGEST:
        return
    root = Path(__file__).resolve().parent
    docs = build_all_documents(root)
    ingest_all(vectorstore, docs)


def build_app():
    decomposer = build_decomposer(get_decomposer_llm())
    retrievers = build_retrievers(vectorstore, semantic_retriever)
    merger = build_merger()
    comp = build_compressor(compressor)
    answer_llm = get_answer_llm()

    chain = build_chain(
        decomposer=decomposer,
        retrievers=retrievers,
        merger=merger,
        compressor=comp,
        answer_llm=answer_llm,
        use_compression=False
    )

    app = LegalAdvisorUI(chain)
    app.render()


if __name__ == "__main__":
    maybe_ingest()
    build_app()
