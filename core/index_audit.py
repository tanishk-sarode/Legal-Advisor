from typing import Dict, List


def find_missing_keyword_fields(mapping: Dict, fields: List[str]) -> List[str]:
    props = mapping.get("mappings", {}).get("properties", {})
    metadata = props.get("metadata", {}).get("properties", {})
    missing = []

    for field in fields:
        field_props = metadata.get(field, {})
        keyword_props = field_props.get("fields", {}).get("keyword")
        if not keyword_props:
            missing.append(field)

    return missing


def audit_index_mapping(vectorstore) -> List[str]:
    mapping = vectorstore.client.indices.get_mapping(index=vectorstore.index_name)
    return find_missing_keyword_fields(
        mapping.get(vectorstore.index_name, {}),
        ["section_id", "article_id", "act_abbrev", "source_type"]
    )


def main() -> None:
    from common.config import vectorstore

    missing = audit_index_mapping(vectorstore)
    if missing:
        print("Missing keyword fields:")
        for field in missing:
            print(f"- metadata.{field}.keyword")
    else:
        print("All required keyword fields present.")


if __name__ == "__main__":
    main()
