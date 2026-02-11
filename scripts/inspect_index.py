from __future__ import annotations

from common.config import vectorstore


def main() -> None:
    index_name = vectorstore.index_name
    client = vectorstore.client

    print(f"Index: {index_name}")
    try:
        count = client.count(index=index_name)
        print(f"Total docs: {count.get('count')}")
    except Exception as exc:
        print(f"Count failed: {exc}")

    try:
        mapping = client.indices.get_mapping(index=index_name)
        print("Mapping:")
        print(mapping)
    except Exception as exc:
        print(f"Mapping fetch failed: {exc}")

    try:
        sample = client.search(index=index_name, body={"size": 3, "query": {"match_all": {}}})
        hits = sample.get("hits", {}).get("hits", [])
        print("Sample docs:")
        for hit in hits:
            src = hit.get("_source", {})
            print(src)
    except Exception as exc:
        print(f"Sample search failed: {exc}")


if __name__ == "__main__":
    main()
