from pathlib import Path
from core.indexer import build_all_documents


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = build_all_documents(root)
    print(f"Total documents to upload: {len(docs)}")


if __name__ == "__main__":
    main()
