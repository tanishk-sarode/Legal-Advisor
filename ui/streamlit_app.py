from pathlib import Path
import streamlit as st

from core.acts import get_act_sources, get_constitution_source

class LegalAdvisorUI:
    def __init__(self, chain):
        self.chain = chain

    def render(self):
        st.set_page_config(page_title="Indian Legal Advisor")
        st.title("Indian Legal Advisor")

        root = Path(__file__).resolve().parents[1]
        act_sources = [get_constitution_source(root)] + get_act_sources(root)
        options = [("All", "All Acts")]
        for act in act_sources:
            options.append((act.act_abbrev, f"{act.act_abbrev} - {act.act}"))

        act_labels = [label for _, label in options]
        selected_label = st.selectbox("Scope", act_labels, index=0)
        act_map = {label: abbrev for abbrev, label in options}
        act_abbrev = act_map.get(selected_label, "All")

        query = st.text_input("Ask a legal question")

        if st.button("Ask") and query:
            with st.spinner("Analyzing…"):
                result = self.chain.invoke({"query": query, "act": act_abbrev})
            
            st.markdown("### Answer")
            if isinstance(result, dict):
                answer_obj = result.get("answer")
                answer_text = answer_obj.answer if answer_obj else ""
            else:
                answer_text = result.answer
            st.write(answer_text)

            sources = result.get("sources") if isinstance(result, dict) else None
            if sources:
                st.markdown("---")
                st.markdown(f"### 📚 Sources ({len(sources)} retrieved)")
                st.caption("Click on each source to view the full content")
                
                for idx, source in enumerate(sources, 1):
                    citation = source.metadata.get("citation", "Unknown source")
                    act_name = source.metadata.get("act", "")
                    text = source.page_content
                    
                    # Create a more detailed header for the expander
                    expander_header = f"{idx}. {citation}"
                    if act_name and citation and act_name not in citation:
                        expander_header += f" - {act_name}"
                    
                    with st.expander(expander_header, expanded=False):
                        # Display metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            if source.metadata.get("act"):
                                st.markdown(f"**Act:** {source.metadata.get('act')}")
                            if source.metadata.get("chapter"):
                                st.markdown(f"**Chapter:** {source.metadata.get('chapter')}")
                        with col2:
                            if source.metadata.get("jurisdiction"):
                                st.markdown(f"**Jurisdiction:** {source.metadata.get('jurisdiction')}")
                            if source.metadata.get("source_type"):
                                st.markdown(f"**Type:** {source.metadata.get('source_type')}")
                        
                        st.markdown("---")
                        st.markdown("**Content:**")
                        st.write(text)

