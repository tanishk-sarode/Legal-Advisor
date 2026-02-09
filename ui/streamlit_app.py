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
            st.write(result.content)
