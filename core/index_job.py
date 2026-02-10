import sys
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from common.config import vectorstore
from core.indexer import Indexer


# # -------------------------------------------------
# # Utility functions
# # -------------------------------------------------
# def _max_doc_length(docs: List[Document]) -> int:
#     return max(len(doc.page_content) for doc in docs) if docs else 0


def _batched(docs: List[Document], batch_size: int) -> List[List[Document]]:
    return [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]


# -------------------------------------------------
# Threaded ingestion
# -------------------------------------------------
def _ingest_threaded(
    vectorstore_client,
    docs: List[Document],
    batch_size: int,
    max_workers: int = 4,
) -> int:
    batches = _batched(docs, batch_size)
    total_ingested = 0

    def upload_batch(batch: List[Document]) -> int:
        vectorstore_client.add_documents(batch)
        return len(batch)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(upload_batch, batch)
            for batch in batches
        ]

        for future in as_completed(futures):
            total_ingested += future.result()
        print(f"Completed ingestion of {total_ingested} documents using {max_workers} workers.")

    return total_ingested


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
def main() -> None:
    root = PROJECT_ROOT
    indexer = Indexer()

    docs = indexer.build_all_documents(root)

    # max_len = _max_doc_length(docs)

    total = _ingest_threaded(
        vectorstore_client=vectorstore,
        docs=docs,
        batch_size=128,
        max_workers=4,   # tune based on API limits / infra
    )

    print(
        f"Ingested {total} documents. "
        # f"Max document length: {max_len} characters."
    )


if __name__ == "__main__":
    main()
