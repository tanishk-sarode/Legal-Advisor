from __future__ import annotations

from typing import Any

import streamlit as st


def _extract_source(source: Any) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        metadata = source.get("metadata", {}) or {}
        content = source.get("page_content", "")
        return metadata, content

    metadata = getattr(source, "metadata", {}) or {}
    content = getattr(source, "page_content", "")
    return metadata, content


def _preview_text(text: str, limit: int = 340) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def render_sources(sources: list[Any], key_prefix: str) -> None:
    if not sources:
        return

    normalized_sources = []
    for source in sources:
        metadata, content = _extract_source(source)
        normalized_sources.append({"metadata": metadata, "page_content": content})

    acts = sorted({s["metadata"].get("act_abbrev") or "N/A" for s in normalized_sources})
    source_count = len(normalized_sources)
    cited_count = sum(1 for s in normalized_sources if s["metadata"].get("citation"))

    st.markdown(
        """
        <div class="sources-header">
            <div>
                <div class="eyebrow">Supporting material</div>
                <div class="sources-title">Retrieved sources for this answer</div>
            </div>
            <div class="sources-stats">
                <span class="meta-chip">{source_count} sources</span>
                <span class="meta-chip">{act_count} acts</span>
                <span class="meta-chip">{cited_count} with citation</span>
            </div>
        </div>
        """.format(source_count=source_count, act_count=len(acts), cited_count=cited_count),
        unsafe_allow_html=True,
    )

    with st.expander(f"View sources ({source_count})", expanded=False):
        control_col1, control_col2 = st.columns([1.15, 1])
        with control_col1:
            sort_choice = st.selectbox(
                "Order sources",
                ["Retrieved order", "Act", "Citation"],
                index=0,
                key=f"{key_prefix}_sort_sources",
            )
        with control_col2:
            st.caption("Use Highlights for quick scan or Full text for detailed reading.")

        if sort_choice == "Act":
            ordered_sources = sorted(
                normalized_sources,
                key=lambda s: (
                    s["metadata"].get("act_abbrev") or "",
                    s["metadata"].get("citation") or "",
                ),
            )
        elif sort_choice == "Citation":
            ordered_sources = sorted(
                normalized_sources,
                key=lambda s: (
                    s["metadata"].get("citation") or "",
                    s["metadata"].get("act_abbrev") or "",
                ),
            )
        else:
            ordered_sources = list(normalized_sources)

        compact_tab, detailed_tab = st.tabs(["Highlights", "Full text"])

        with compact_tab:
            for index, source in enumerate(ordered_sources, 1):
                metadata, content = source["metadata"], source["page_content"]
                citation = metadata.get("citation") or f"Source {index}"
                act_name = metadata.get("act") or ""
                act_abbrev = metadata.get("act_abbrev") or ""
                source_type = metadata.get("source_type") or ""
                jurisdiction = metadata.get("jurisdiction") or ""
                chapter = metadata.get("chapter") or ""

                with st.container(border=True):
                    st.markdown(f"#### {citation}")
                    metadata_line = "  |  ".join(
                        value
                        for value in [act_abbrev, act_name, chapter, source_type, jurisdiction]
                        if value
                    )
                    if metadata_line:
                        st.caption(metadata_line)
                    st.write(_preview_text(content))

        with detailed_tab:
            expand_all = st.checkbox(
                "Expand all detailed sources",
                value=False,
                key=f"{key_prefix}_expand_all_sources",
            )
            for index, source in enumerate(ordered_sources, 1):
                metadata, content = source["metadata"], source["page_content"]
                citation = metadata.get("citation") or f"Source {index}"
                act_name = metadata.get("act") or ""
                header = citation if not act_name or act_name in citation else f"{citation} - {act_name}"

                with st.expander(header, expanded=expand_all):
                    meta_left, meta_right = st.columns(2)
                    with meta_left:
                        if metadata.get("act"):
                            st.markdown(f"**Act:** {metadata.get('act')}")
                        if metadata.get("chapter"):
                            st.markdown(f"**Chapter:** {metadata.get('chapter')}")
                        if metadata.get("title"):
                            st.markdown(f"**Title:** {metadata.get('title')}")
                    with meta_right:
                        if metadata.get("jurisdiction"):
                            st.markdown(f"**Jurisdiction:** {metadata.get('jurisdiction')}")
                        if metadata.get("source_type"):
                            st.markdown(f"**Type:** {metadata.get('source_type')}")
                        if metadata.get("act_abbrev"):
                            st.markdown(f"**Act Code:** {metadata.get('act_abbrev')}")
                    st.divider()
                    st.write(content)
